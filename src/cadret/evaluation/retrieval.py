from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..checkpoints import load_checkpoint, restore_text_heads
from ..config import DATA, INSTRUCT
from ..data.descriptors import prepare_cache
from ..data.text import load_data
from ..models.towers import CADTower, TextEncoder
from .metrics import pooled_recall


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cadret evaluate", description="Evaluate a complete retrieval checkpoint"
    )
    ap.add_argument("--data_dir", default=DATA, help="Text2CAD dataset directory")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint saved by cadret train")
    ap.add_argument("--cache", help="CAD descriptor cache file")
    ap.add_argument("--model", help="Override the base model identifier or directory")
    ap.add_argument("--adapter_dir", help="Directory containing the trained LoRA adapter")
    ap.add_argument("--levels", help="Comma-separated description levels")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--n_pools", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    checkpoint_path = Path(args.checkpoint)
    try:
        checkpoint = load_checkpoint(checkpoint_path, require_text=True)
    except ValueError as error:
        ap.error(str(error))
    config = checkpoint["config"]
    levels = (args.levels or config["levels"]).split(",")
    cache_path = Path(args.cache) if args.cache else checkpoint_path.parent / "spec_desc_cache.pt"
    prepare_cache(args.data_dir, cache_path, workers=args.workers)
    data = load_data(levels, args.data_dir, cache_path)["test"]
    device = torch.device(args.device)
    adapter_dir = (
        Path(args.adapter_dir) if args.adapter_dir else checkpoint_path.parent / "lora_text"
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    text = TextEncoder(
        args.model or config["model"], lora_r=config["lora_r"], adapter_dir=adapter_dir
    )
    restore_text_heads(text, checkpoint)
    text.to(device).eval()
    cad = CADTower().to(device).eval()
    cad.load_state_dict(checkpoint["cadt"])
    rows = data["rows"]
    order = np.argsort([len(row[2]) for row in rows], kind="stable")
    queries = torch.zeros(len(rows), 512)
    with torch.no_grad():
        values = (data["V"] - checkpoint["mu"]) / checkpoint["sd"]
        gallery = F.normalize(cad(values.to(device)), dim=-1).cpu()
        for start in tqdm(range(0, len(order), args.bs), desc="encode"):
            indices = order[start : start + args.bs]
            tokens = tokenizer(
                [INSTRUCT + rows[i][2] for i in indices],
                padding=True,
                truncation=True,
                max_length=config["max_len"],
                return_tensors="pt",
            ).to(device)
            encoded, _, _ = text(tokens.input_ids, tokens.attention_mask)
            queries[indices] = F.normalize(encoded, dim=-1).cpu()
    results = pooled_recall(queries, gallery, rows, levels, args.pool, args.n_pools)
    for level, metrics in results.items():
        print(level, " ".join((f"{name}={value:.4f}" for name, value in metrics.items())))
