from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..config import DATA
from ..data.sequences import CADSeqStore
from ..geometry.descriptor import descriptor_vector
from ..geometry.spectral import describe

_STORE = None


def _init_store(data_dir):
    global _STORE
    data_dir = Path(data_dir)
    _STORE = CADSeqStore(data_dir / "extracted" / "cad_seq", data_dir / "cad_seq.zip")


def _work(uid):
    global _STORE
    if _STORE is None:
        _STORE = CADSeqStore(f"{DATA}/extracted/cad_seq", f"{DATA}/cad_seq.zip")
    cv, fv, iv = _STORE.load_full(uid)
    if cv is None:
        return None
    try:
        d = describe(cv, fv, iv)
    except Exception:
        return None
    if not d:
        return None
    return (
        uid,
        descriptor_vector(d),
        np.asarray(cv).tobytes(),
        {
            "n_parts": d["n_parts"],
            "b1": d["b1"],
            "sym": d["max_sym"],
            "n_circles": d["n_circles"],
            "n_curves": d["n_curves"],
        },
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cadret validate-descriptors", description="Evaluate geometric descriptors"
    )
    ap.add_argument("--data_dir", default=DATA, help="Text2CAD dataset directory")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--n_pools", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default="runs/spec_val.pt")
    args = ap.parse_args(argv)
    with (Path(args.data_dir) / "text2cad_v1.0" / "train_test_val.json").open(
        encoding="utf-8"
    ) as f:
        splits = json.load(f)
    rng = random.Random(0)
    uids = splits[args.split]
    uids = rng.sample(uids, min(args.n, len(uids)))
    V, U, RAW, META = ([], [], [], [])
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_store, initargs=(args.data_dir,)
    ) as ex:
        for r in tqdm(ex.map(_work, uids, chunksize=8), total=len(uids), desc="descriptors"):
            if r is None:
                continue
            U.append(r[0])
            V.append(r[1])
            RAW.append(r[2])
            META.append(r[3])
    V = np.stack(V)
    n = len(V)
    print(f"\n{n} descriptors, dim {V.shape[1]}")
    keys = [tuple(np.round(v, 4)) for v in V]
    print(f"distinct descriptors      : {len(set(keys))}/{n}  ({len(set(keys)) / n:.1%})")
    raw_groups = Counter(RAW)
    print(f"distinct raw cad_vec      : {len(raw_groups)}/{n}  ({len(raw_groups) / n:.1%})")
    dead = np.abs(V).max(0) < 1e-06
    print(f"always-zero dimensions    : {int(dead.sum())}/{V.shape[1]}")
    std = V.std(0)
    print(f"dims with std < 1e-3      : {int((std < 0.001).sum())}")
    Z = (V - V.mean(0)) / np.maximum(V.std(0), 1e-06)
    Z[:, dead] = 0.0
    by_raw: dict[bytes, list[int]] = {}
    for i, r in enumerate(RAW):
        by_raw.setdefault(r, []).append(i)
    within = [
        float(np.linalg.norm(Z[a] - Z[b]))
        for g in by_raw.values()
        if len(g) > 1
        for a, b in [(g[0], g[1])]
    ]
    idx = np.random.default_rng(0).integers(0, n, (5000, 2))
    between = [float(np.linalg.norm(Z[a] - Z[b])) for a, b in idx if RAW[a] != RAW[b]]
    if within:
        print(
            f"\nwithin-duplicate distance : mean {np.mean(within):.3f}  p95 {np.percentile(within, 95):.3f}  (n={len(within)})"
        )
    print(
        f"between-model distance    : mean {np.mean(between):.3f}  p5  {np.percentile(between, 5):.3f}"
    )
    if within:
        print(
            f"separation (p95 vs p5)    : {np.percentile(within, 95):.3f} < {np.percentile(between, 5):.3f}  -> {('SEPARATED' if np.percentile(within, 95) < np.percentile(between, 5) else 'OVERLAPPING')}"
        )
    T = torch.from_numpy(Z).float()
    T = torch.nn.functional.normalize(T, dim=-1)
    prng = random.Random(1)
    hits = {1: 0, 5: 0, 10: 0}
    ties = 0
    for _ in tqdm(range(args.n_pools), desc="ceiling"):
        g = prng.randrange(n)
        cand = [g] + prng.sample([i for i in range(n) if i != g], args.pool - 1)
        s = (T[cand] @ T[g]).numpy()
        gs = s[0]
        rank = int((s[1:] > gs + 1e-06).sum() + 0.5 * (np.abs(s[1:] - gs) <= 1e-06).sum())
        ties += int((np.abs(s[1:] - gs) <= 1e-06).sum() > 0)
        for k in hits:
            hits[k] += rank < k
    print(f"\nself-retrieval ceiling (pool={args.pool}, {args.n_pools} trials):")
    for k in (1, 5, 10):
        print(f"  R@{k:<3d} {hits[k] / args.n_pools:.4f}")
    print(f"  pools containing an exact tie: {ties / args.n_pools:.1%}")
    print("\ntext-predictable invariants (marginal distributions):")
    for f in ("n_parts", "b1", "sym", "n_circles"):
        c = Counter((m[f] for m in META))
        top = ", ".join((f"{k}:{v / n:.1%}" for k, v in sorted(c.items())[:8]))
        print(f"  {f:10s} {len(c):3d} values | {top}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"uids": U, "V": V, "meta": META, "raw": RAW}, args.out)
    print(f"\nsaved {args.out}")
