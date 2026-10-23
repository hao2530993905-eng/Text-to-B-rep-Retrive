from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AUX_CLS, AUX_REG, DATA
from ..data.descriptors import prepare_cache
from ..evaluation.metrics import pooled_recall
from ..models.towers import CADTower, TextTower
from .objectives import eq_classes, multipos


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cadret train-frozen", description="Train with frozen text features"
    )
    ap.add_argument("--data_dir", default=DATA, help="Text2CAD dataset directory")
    ap.add_argument("--output_dir", default="runs", help="Directory for generated files")
    ap.add_argument("--features_dir", help="Directory containing encoded text features")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--levels", default="abstract,beginner,intermediate,expert")
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--n_pools", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--aux_w", type=float, default=0.5)
    ap.add_argument("--eq_thresh", type=float, default=1.0)
    ap.add_argument("--cache", help="CAD descriptor cache file")
    args = ap.parse_args(argv)
    levels = args.levels.split(",")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or output_dir / "spec_desc_cache.pt"
    c = prepare_cache(args.data_dir, cache_path, workers=args.workers)
    device = torch.device("cuda")
    features_dir = Path(args.features_dir) if args.features_dir else output_dir / "textfeat"
    meta = torch.load(features_dir / "meta.pt", map_location="cpu", weights_only=False)
    feats = torch.load(features_dir / "feats.pt", map_location="cpu", weights_only=False)
    rows = meta["rows"]
    print(f"text features {tuple(feats.shape)}", flush=True)

    def assemble(sp):
        U, V, A = (c[sp]["U"], c[sp]["V"], c[sp]["A"])
        pos = {u: i for i, u in enumerate(U)}
        owner, lev, fidx = ([], [], [])
        for (uid, L), fi in rows.items():
            if L in levels and uid in pos:
                owner.append(pos[uid])
                lev.append(L)
                fidx.append(fi)
        return (torch.from_numpy(V).float(), A, np.array(owner), lev, np.array(fidx))

    trV, trA, trO, trL, trI = assemble("train")
    evV, evA, evO, evL, evI = assemble("test")
    mu, sd = (trV.mean(0), trV.std(0).clamp(min=0.0001))
    trV, evV = ((trV - mu) / sd, (evV - mu) / sd)
    trF, evF = (feats[trI], feats[evI])
    print(
        f"models: train {len(trV)} test {len(evV)} | text rows: {len(trF)} / {len(evF)}", flush=True
    )

    def targets(A, O):
        cls = {
            k: torch.tensor([min(A[i][k], n - 1) for i in O], dtype=torch.long) for k, n in AUX_CLS
        }
        reg = torch.tensor([[A[i][k] for k in AUX_REG] for i in O], dtype=torch.float32)
        return (cls, reg)

    trC, trR = targets(trA, trO)
    evC, evR = targets(evA, evO)
    model_cls, n_classes = eq_classes(trV, args.eq_thresh)
    print(f"spectral equivalence classes: {n_classes} / {len(trV)} models", flush=True)
    cadt, txtt = (CADTower().to(device), TextTower(trF.size(1)).to(device))
    params = list(cadt.parameters()) + list(txtt.parameters())
    print(f"trainable {sum((p.numel() for p in params)) / 1000000.0:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(trF) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    trV_d, trF_d = (trV.to(device), trF.to(device))
    trC_d = {k: v.to(device) for k, v in trC.items()}
    trR_d, mcls = (trR.to(device), torch.from_numpy(model_cls).to(device))
    owner_d = torch.from_numpy(trO).to(device)
    for ep in range(args.epochs):
        cadt.train()
        txtt.train()
        idx = torch.randperm(len(trF))
        tc = ta = nb = 0
        for s in range(0, len(idx), args.bs):
            r = idx[s : s + args.bs].to(device)
            if len(r) < 16:
                continue
            m = owner_d[r]
            zc = cadt(trV_d[m])
            zq, logits, reg = txtt(trF_d[r])
            S = F.normalize(zq, dim=-1) @ F.normalize(zc, dim=-1).t() / args.temp
            k = mcls[m]
            P = (k[:, None] == k[None, :]).float()
            P = P / P.sum(1, keepdim=True)
            l_con = multipos(S, P)
            l_aux = sum(
                (F.cross_entropy(logits[n], trC_d[n][r]) for n, _ in AUX_CLS)
            ) + F.smooth_l1_loss(reg, trR_d[r])
            (l_con + args.aux_w * l_aux).backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tc += l_con.item()
            ta += float(l_aux)
            nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(
                f"  ep {ep:3d} contrastive {tc / max(nb, 1):.4f}  aux {ta / max(nb, 1):.4f}",
                flush=True,
            )
    cadt.eval()
    txtt.eval()
    with torch.no_grad():
        ZC = F.normalize(cadt(evV.to(device)), dim=-1).cpu()
        ZQ, elog, _ = txtt(evF.to(device))
        ZQ = F.normalize(ZQ, dim=-1).cpu()
        acc = {k: (elog[k].argmax(-1).cpu() == evC[k]).float().mean().item() for k, _ in AUX_CLS}
    print("\nauxiliary head accuracy (how much geometry the text alone reveals):")
    for k, _ in AUX_CLS:
        maj = np.bincount(evC[k].numpy()).max() / len(evC[k])
        print(f"  {k:11s} {acc[k]:.3f}  (majority {maj:.3f})")
    print(f"\nRESULTS (pool={args.pool}, {args.n_pools} trials/level, no coordinates):")
    metric_rows = [(int(owner), level, "") for owner, level in zip(evO, evL)]
    results = pooled_recall(ZQ, ZC, metric_rows, levels, args.pool, args.n_pools)
    for level, metrics in results.items():
        print(level, " ".join((f"{name}={value:.4f}" for name, value in metrics.items())))
    torch.save(
        {"cadt": cadt.state_dict(), "txtt": txtt.state_dict(), "mu": mu, "sd": sd},
        output_dir / "spec_model.pt",
    )
