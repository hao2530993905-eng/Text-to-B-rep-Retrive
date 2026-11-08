from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from .config import DATASET_LEVELS, INSTRUCTION


def build_parser():
    parser = argparse.ArgumentParser(prog="cadret encode-text")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_LEVELS), required=True)
    parser.add_argument("--levels")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=384)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.cache.is_file():
        raise FileNotFoundError(f"cache not found: {args.cache}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from transformers import AutoModel, AutoTokenizer

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache["dataset"] != args.dataset:
        raise ValueError("cache dataset does not match --dataset")
    levels = tuple(
        value.strip()
        for value in (args.levels or ",".join(cache["levels"])).split(",")
        if value.strip()
    )
    texts = sorted(
        {
            values[level].strip()
            for split in cache["splits"].values()
            for values in split["texts"]
            for level in levels
            if isinstance(values.get(level), str) and values[level].strip()
        }
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = (
        AutoModel.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    lower = rank * len(texts) // world_size
    upper = (rank + 1) * len(texts) // world_size
    local_input_indices = np.arange(lower, upper)
    local_order = np.argsort(
        [len(texts[int(index)]) for index in local_input_indices],
        kind="stable",
    )
    ordered_indices = local_input_indices[local_order]
    local_features = torch.zeros(
        len(local_input_indices),
        model.config.hidden_size,
        dtype=torch.float32,
    )
    with torch.no_grad():
        iterator = range(0, len(ordered_indices), args.batch_size)
        if rank == 0:
            iterator = tqdm(iterator, desc="encode-text")
        for start in iterator:
            selected_positions = np.arange(
                start,
                min(start + args.batch_size, len(ordered_indices)),
            )
            selected = ordered_indices[selected_positions]
            tokenized = tokenizer(
                [INSTRUCTION + texts[int(index)] for index in selected],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
            ).last_hidden_state
            mask = tokenized.attention_mask
            indexes = mask.size(1) - 1 - mask.flip(1).float().argmax(1)
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), indexes]
            output_positions = local_order[selected_positions]
            local_features[torch.from_numpy(output_positions).long()] = pooled.float().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shard_path = args.output.parent / f".{args.output.stem}.rank{rank}.pt"
    torch.save(local_features, shard_path)
    dist.barrier()
    if rank == 0:
        merged = torch.cat(
            [
                torch.load(
                    args.output.parent / f".{args.output.stem}.rank{shard_rank}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for shard_rank in range(world_size)
            ]
        )
        temporary = args.output.with_suffix(args.output.suffix + ".part")
        torch.save(
            {
                "dataset": args.dataset,
                "levels": list(levels),
                "model": args.model,
                "max_length": args.max_length,
                "texts": texts,
                "features": merged,
            },
            temporary,
        )
        temporary.replace(args.output)
        print(
            f"saved {len(texts)} text features with dimension {merged.shape[1]}",
            flush=True,
        )
    dist.barrier()
    if rank == 0:
        for shard_rank in range(world_size):
            shard = args.output.parent / f".{args.output.stem}.rank{shard_rank}.pt"
            shard.unlink(missing_ok=True)
    dist.destroy_process_group()
