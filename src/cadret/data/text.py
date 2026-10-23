from __future__ import annotations

import pandas as pd
import torch

from ..config import DATA, DatasetPaths


def load_data(levels, data_dir=DATA, cache_path="runs/spec_desc_cache.pt"):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    df = pd.read_csv(DatasetPaths(data_dir).descriptions)
    text = {}
    for _, r in df.iterrows():
        for L in levels:
            t = r.get(L)
            if isinstance(t, str) and t.strip():
                text[r["uid"], L] = t.strip()
    out = {}
    for sp in ("train", "test"):
        U, V, A = (cache[sp]["U"], cache[sp]["V"], cache[sp]["A"])
        pos = {u: i for i, u in enumerate(U)}
        rows = [(pos[u], L, text[u, L]) for u, L in text if u in pos]
        out[sp] = {"V": torch.from_numpy(V).float(), "A": A, "rows": rows}
    return out


def gather_texts(levels, data_dir=DATA):
    df = pd.read_csv(DatasetPaths(data_dir).descriptions)
    uniq: dict[str, int] = {}
    rows: dict[tuple[str, str], int] = {}
    for _, r in df.iterrows():
        for L in levels:
            t = r.get(L)
            if isinstance(t, str) and t.strip():
                t = t.strip()
                rows[r["uid"], L] = uniq.setdefault(t, len(uniq))
    order = sorted(uniq, key=uniq.get)
    return (rows, order)
