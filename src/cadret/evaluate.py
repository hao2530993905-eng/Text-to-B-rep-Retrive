from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import DATASET_LEVELS, INSTRUCTION
from .data import load_split
from .model import CADTower, TextEncoder, TextProjection
from .runtime import gather_variable, load_checkpoint, pooled_recall, restore_text_heads


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def build_parser():
    parser = argparse.ArgumentParser(prog="cadret evaluate")
    parser.add_argument("--dataset", choices=tuple(DATASET_LEVELS), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--text-features", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--levels")
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--pool-size", type=positive_int, default=1000)
    parser.add_argument("--trials", type=positive_int, default=1000)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.cache.is_file():
        raise FileNotFoundError(f"cache not found: {args.cache}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    if config["dataset"] != args.dataset:
        raise ValueError("checkpoint dataset does not match --dataset")
    levels = tuple(
        value.strip() for value in (args.levels or config["levels"]).split(",") if value.strip()
    )
    data = load_split(args.cache, args.split, levels)
    if data["dataset"] != args.dataset:
        raise ValueError("cache dataset does not match --dataset")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        tuning = config["tuning"]
        dtype = torch.float32 if config["precision"] == "fp32" else torch.bfloat16
        tokenizer = None
        text_features = None
        feature_lookup = None
        if tuning == "lora":
            from transformers import AutoTokenizer

            adapter = args.adapter or args.checkpoint.parent / "adapter"
            if not adapter.is_dir():
                raise FileNotFoundError(f"adapter not found: {adapter}")
            tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
            tokenizer.padding_side = "left"
            text_model = TextEncoder(
                args.model or config["model"],
                lora_rank=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                adapter=adapter,
                dtype=dtype,
            ).to(device)
        else:
            text_features_path = args.text_features or Path(config["text_features"])
            cached_features = torch.load(
                text_features_path,
                map_location="cpu",
                weights_only=False,
            )
            text_features = cached_features["features"].float().to(device)
            feature_lookup = {text: index for index, text in enumerate(cached_features["texts"])}
            text_model = TextProjection(text_features.shape[1]).to(device)
        restore_text_heads(text_model, checkpoint)
        cad_tower = CADTower().to(device)
        cad_tower.load_state_dict(checkpoint["cad_tower"])
        text_model.eval()
        cad_tower.eval()
        values = (data["vectors"] - checkpoint["mean"]) / checkpoint["std"]
        lower = rank * len(data["rows"]) // dist.get_world_size()
        upper = (rank + 1) * len(data["rows"]) // dist.get_world_size()
        local_rows = data["rows"][lower:upper]
        local_queries = torch.zeros(len(local_rows), 512, device=device)
        with torch.no_grad():
            gallery = F.normalize(cad_tower(values.to(device)), dim=-1).cpu()
            if tuning == "lora":
                for start in range(0, len(local_rows), args.batch_size):
                    selected = list(range(start, min(start + args.batch_size, len(local_rows))))
                    tokenized = tokenizer(
                        [INSTRUCTION + local_rows[index][2] for index in selected],
                        padding=True,
                        truncation=True,
                        max_length=config["max_length"],
                        return_tensors="pt",
                    ).to(device)
                    encoded = text_model(
                        tokenized.input_ids,
                        tokenized.attention_mask,
                    )
                    local_queries[selected] = F.normalize(encoded, dim=-1)
            else:
                feature_indices = torch.tensor(
                    [feature_lookup[row[2]] for row in local_rows],
                    dtype=torch.long,
                    device=device,
                )
                encoded = text_model(text_features[feature_indices])
                local_queries[:] = F.normalize(encoded, dim=-1)
        queries = gather_variable(local_queries).cpu()
        if rank == 0:
            results, digest = pooled_recall(
                queries,
                gallery,
                data["rows"],
                levels,
                min(args.pool_size, len(gallery)),
                args.trials,
            )
            output = args.output or args.checkpoint.parent / f"{args.split}_metrics.json"
            payload = {
                "dataset": args.dataset,
                "split": args.split,
                "checkpoint": str(args.checkpoint),
                "pool_size": min(args.pool_size, len(gallery)),
                "trials": args.trials,
                "pool_sha256": digest,
                "results": results,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(results, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
