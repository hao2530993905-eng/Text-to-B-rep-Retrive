from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ..checkpoints import save_checkpoint
from ..config import AUX_CLS, AUX_REG, DATA, INSTRUCT
from ..data.text import load_data
from ..evaluation.metrics import pooled_recall
from ..models.towers import CADTower, TextEncoder
from .distributed import gather_cat, gather_variable
from .objectives import eq_classes, multipos


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cadret train", description="Train the LoRA retrieval model")
    ap.add_argument("--data_dir", default=DATA, help="Text2CAD dataset directory")
    ap.add_argument("--output_dir", default="runs", help="Directory for generated files")
    ap.add_argument("--cache", help="CAD descriptor cache file")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--levels", default="abstract,beginner,intermediate,expert")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=24, help="per GPU")
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--head_lr", type=float, default=0.0005)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--aux_w", type=float, default=0.5)
    ap.add_argument("--eq_thresh", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--n_pools", type=int, default=1000)
    ap.add_argument("--eval_bs", type=int, default=64)
    args = ap.parse_args(argv)
    levels = args.levels.split(",")
    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache) if args.cache else output_dir / "spec_desc_cache.pt"
    if not cache_path.exists():
        ap.error(
            "Descriptor cache is missing; run cadret prepare with the same data and output directories first"
        )
    if not torch.cuda.is_available():
        ap.error("CUDA is required for LoRA training")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        rank, world = (dist.get_rank(), dist.get_world_size())
        output_dir.mkdir(parents=True, exist_ok=True)

        def log(*a, **k):
            if rank == 0:
                print(*a, **{**k, "flush": True})

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tok.padding_side = "left"
        d = load_data(levels, args.data_dir, cache_path)
        trV, trA, trRows = (d["train"]["V"], d["train"]["A"], d["train"]["rows"])
        evV, evA, evRows = (d["test"]["V"], d["test"]["A"], d["test"]["rows"])
        mu, sd = (trV.mean(0), trV.std(0).clamp(min=0.0001))
        trV, evV = (((trV - mu) / sd).to(dev), ((evV - mu) / sd).to(dev))
        mcls, n_cls = eq_classes(trV.cpu(), args.eq_thresh)
        log(
            f"train models {len(trV)} rows {len(trRows)} | test models {len(evV)} rows {len(evRows)} | eq classes {n_cls}"
        )
        cadt = CADTower().to(dev)
        txt = TextEncoder(args.model, lora_r=args.lora_r).to(dev)
        cadt = DDP(cadt, device_ids=[local_rank])
        txt = DDP(txt, device_ids=[local_rank], find_unused_parameters=False)
        n_tr = sum((p.numel() for p in txt.parameters() if p.requires_grad))
        log(
            f"trainable: text {n_tr / 1000000.0:.1f}M + cad {sum((p.numel() for p in cadt.parameters())) / 1000000.0:.1f}M"
        )
        bb_p = [p for n, p in txt.named_parameters() if p.requires_grad and ".bb." in n]
        hd_p = [p for n, p in txt.named_parameters() if p.requires_grad and ".bb." not in n]
        opt = torch.optim.AdamW(
            [
                {"params": bb_p, "lr": args.lr},
                {"params": hd_p, "lr": args.head_lr},
                {"params": list(cadt.parameters()), "lr": args.head_lr},
            ],
            weight_decay=0.01,
        )
        per_ep = len(trRows) // (args.bs * world)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            [args.lr, args.head_lr, args.head_lr],
            total_steps=args.epochs * per_ep,
            pct_start=0.05,
        )
        mcls_d = torch.from_numpy(mcls).to(dev)
        aux_cls_t = {
            k: torch.tensor([min(a[k], n - 1) for a in trA], dtype=torch.long).to(dev)
            for k, n in AUX_CLS
        }
        aux_reg_t = torch.tensor([[a[k] for k in AUX_REG] for a in trA], dtype=torch.float32).to(
            dev
        )
        for ep in range(args.epochs):
            txt.train()
            cadt.train()
            g = torch.Generator().manual_seed(ep)
            order = torch.randperm(len(trRows), generator=g).tolist()
            tc = ta = nb = 0
            wc = wa = wn = 0
            it = range(per_ep)
            if rank == 0:
                it = tqdm(it, desc=f"epoch {ep}")
            for step in it:
                s = step * args.bs * world + rank * args.bs
                sel = order[s : s + args.bs]
                if len(sel) < args.bs:
                    continue
                mi = torch.tensor([trRows[i][0] for i in sel], device=dev)
                b = tok(
                    [INSTRUCT + trRows[i][2] for i in sel],
                    padding=True,
                    truncation=True,
                    max_length=args.max_len,
                    return_tensors="pt",
                ).to(dev)
                zq, logits, reg = txt(b.input_ids, b.attention_mask)
                zc = cadt(trV[mi])
                ZQ = gather_cat(F.normalize(zq, dim=-1))
                ZC = gather_cat(F.normalize(zc, dim=-1))
                K = gather_cat(mcls_d[mi].float()).long()
                P = (K[:, None] == K[None, :]).float()
                P = P / P.sum(1, keepdim=True)
                l_con = multipos(ZQ @ ZC.t() / args.temp, P)
                l_aux = sum(
                    (F.cross_entropy(logits[k], aux_cls_t[k][mi]) for k, _ in AUX_CLS)
                ) + F.smooth_l1_loss(reg, aux_reg_t[mi])
                (l_con + args.aux_w * l_aux).backward()
                nn.utils.clip_grad_norm_([p for p in txt.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                tc += l_con.item()
                ta += l_aux.item()
                nb += 1
                wc += l_con.item()
                wa += l_aux.item()
                wn += 1
                if rank == 0 and step % 25 == 0 and wn:
                    it.set_postfix(
                        con=f"{wc / wn:.3f}",
                        aux=f"{wa / wn:.3f}",
                        lr=f"{sched.get_last_lr()[0]:.1e}",
                    )
                    log(f"    step {step:5d}  con {wc / wn:.4f}  aux {wa / wn:.4f}")
                    wc = wa = wn = 0
            log(f"  ep {ep} contrastive {tc / max(nb, 1):.4f}  aux {ta / max(nb, 1):.4f}")
        txt.eval()
        cadt.eval()
        lo = rank * len(evRows) // world
        hi = (rank + 1) * len(evRows) // world
        mine = evRows[lo:hi]
        perm = np.argsort([len(r[2]) for r in mine], kind="stable")
        Z = torch.zeros(len(mine), 512, device=dev)
        AC = {k: torch.zeros(len(mine), dtype=torch.long, device=dev) for k, _ in AUX_CLS}
        with torch.no_grad():
            it = range(0, len(perm), args.eval_bs)
            if rank == 0:
                it = tqdm(it, desc="eval encode")
            for i in it:
                g2 = perm[i : i + args.eval_bs]
                b = tok(
                    [INSTRUCT + mine[j][2] for j in g2],
                    padding=True,
                    truncation=True,
                    max_length=args.max_len,
                    return_tensors="pt",
                ).to(dev)
                zq, lg, _ = txt.module(b.input_ids, b.attention_mask)
                Z[g2] = F.normalize(zq, dim=-1)
                for k, _ in AUX_CLS:
                    AC[k][g2] = lg[k].argmax(-1)
            ZCe = F.normalize(cadt.module(evV), dim=-1)
        all_queries = gather_variable(Z).cpu()
        all_predictions = {k: gather_variable(AC[k]).cpu() for k, _ in AUX_CLS}
        if rank == 0:
            ZQe = all_queries
            pred = all_predictions
            ZCe = ZCe.cpu()
            print("\nauxiliary head accuracy (geometry read from text alone):")
            for k, n in AUX_CLS:
                gt = torch.tensor([min(evA[r[0]][k], n - 1) for r in evRows])
                maj = np.bincount(gt.numpy()).max() / len(gt)
                print(
                    f"  {k:11s} {(pred[k] == gt).float().mean().item():.3f}  (majority {maj:.3f})"
                )
            print(f"\nRESULTS (pool={args.pool}, {args.n_pools} trials/level, no coordinates):")
            results = pooled_recall(ZQe, ZCe, evRows, levels, args.pool, args.n_pools)
            for level, metrics in results.items():
                print(level, " ".join((f"{name}={value:.4f}" for name, value in metrics.items())))
            config = {
                name: getattr(args, name) for name in ("model", "lora_r", "max_len", "levels")
            }
            save_checkpoint(cadt.module, txt.module, tok, mu, sd, config, output_dir)
        dist.barrier()
    finally:
        dist.destroy_process_group()
