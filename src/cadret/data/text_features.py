from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from ..config import DATA, INSTRUCT, LEVELS
from .text import gather_texts


def worker(rank, world, texts, model, bs, max_len, out_dir):
    torch.cuda.set_device(rank)
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tok.padding_side = "left"
    bb = (
        AutoModel.from_pretrained(model, dtype=torch.bfloat16, trust_remote_code=True)
        .to(rank)
        .eval()
    )
    lo = rank * len(texts) // world
    hi = (rank + 1) * len(texts) // world
    mine = texts[lo:hi]
    perm = np.argsort([len(t) for t in mine], kind="stable")
    out = torch.zeros(len(mine), bb.config.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        it = range(0, len(perm), bs)
        if rank == 0:
            it = tqdm(it, desc=f"encode (shard 0 of {world})")
        for i in it:
            sel = perm[i : i + bs]
            b = tok(
                [INSTRUCT + mine[j] for j in sel],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(rank)
            h = bb(**b).last_hidden_state
            m = b.attention_mask
            idx = m.size(1) - 1 - m.flip(dims=[1]).float().argmax(dim=1)
            out[sel] = h[torch.arange(h.size(0), device=h.device), idx].float().cpu()
    torch.save(out, f"{out_dir}/shard_{rank}.pt")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cadret encode-text", description="Cache frozen text features"
    )
    ap.add_argument("--data_dir", default=DATA, help="Text2CAD dataset directory")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--out", default="runs/textfeat")
    args = ap.parse_args(argv)
    levels = args.levels.split(",")
    os.makedirs(args.out, exist_ok=True)
    rows, uniq = gather_texts(levels, args.data_dir)
    print(
        f"{len(rows)} (uid, level) rows -> {len(uniq)} unique strings ({len(uniq) / len(rows):.1%})",
        flush=True,
    )
    meta = f"{args.out}/meta.pt"
    torch.save({"rows": rows, "uniq": uniq, "levels": levels}, meta)
    world = torch.cuda.device_count()
    if world == 0:
        raise RuntimeError("CUDA is required for text feature encoding")
    print(f"encoding on {world} GPUs", flush=True)
    mp.spawn(
        worker,
        args=(world, uniq, args.model, args.bs, args.max_len, args.out),
        nprocs=world,
        join=True,
    )
    F = torch.cat([torch.load(f"{args.out}/shard_{r}.pt") for r in range(world)])
    assert len(F) == len(uniq), f"{len(F)} != {len(uniq)}"
    torch.save(F, f"{args.out}/feats.pt")
    print(f"saved {tuple(F.shape)} to {args.out}/feats.pt", flush=True)
