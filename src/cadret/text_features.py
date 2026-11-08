from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = (
        AutoModel.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .cuda()
        .eval()
    )
    order = np.argsort([len(value) for value in texts], kind="stable")
    features = torch.zeros(len(texts), model.config.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        for start in tqdm(range(0, len(order), args.batch_size), desc="encode-text"):
            selected = order[start : start + args.batch_size]
            tokenized = tokenizer(
                [INSTRUCTION + texts[int(index)] for index in selected],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).cuda()
            hidden = model(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
            ).last_hidden_state
            mask = tokenized.attention_mask
            indexes = mask.size(1) - 1 - mask.flip(1).float().argmax(1)
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), indexes]
            features[torch.from_numpy(selected).long()] = pooled.float().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    torch.save(
        {
            "dataset": args.dataset,
            "levels": list(levels),
            "model": args.model,
            "max_length": args.max_length,
            "texts": texts,
            "features": features,
        },
        temporary,
    )
    temporary.replace(args.output)
    print(
        f"saved {len(texts)} text features with dimension {features.shape[1]}",
        flush=True,
    )
