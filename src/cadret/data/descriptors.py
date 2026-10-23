from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..config import DATA, DatasetPaths
from ..geometry.descriptor import descriptor_vector
from ..geometry.spectral import describe
from .sequences import CADSeqStore

_STORE = None


def _init_store(data_dir):
    global _STORE
    paths = DatasetPaths(data_dir)
    _STORE = CADSeqStore(paths.sequences, paths.sequence_zip)


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
    parts = d["parts"]
    big = max(parts, key=lambda p: p["big"]["outer"]["area"])
    dep = np.array([p["depth"] for p in parts], dtype=np.float32)
    aux = {
        "n_parts": d["n_parts"],
        "b1": d["b1"],
        "n_circles": d["n_circles"],
        "sym": 0 if d["max_sym"] == 0 else min(d["max_sym"], 7),
        "n_faces": sum((p["n_faces"] for p in parts)),
        "slender": float(np.tanh(big["slender"])),
        "iso": float(big["big"]["outer"]["iso"]),
        "fill": float(big["big"]["fill"]),
        "aspect": float(big["big"]["outer"]["aspect"]),
        "depth_spread": float(dep.std() / max(dep.mean(), 1e-09)),
    }
    return (uid, descriptor_vector(d), aux)


def build_desc(uids, workers=48, data_dir=DATA):
    U, V, A = ([], [], [])
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_store, initargs=(str(data_dir),)
    ) as ex:
        for r in tqdm(ex.map(_work, uids, chunksize=16), total=len(uids), desc="describe"):
            if r is None:
                continue
            U.append(r[0])
            V.append(r[1])
            A.append(r[2])
    if not U:
        raise ValueError("No valid CAD sequences were found in the requested split")
    return (U, np.stack(V), A)


def prepare_cache(data_dir, cache_path, workers=48):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    with DatasetPaths(data_dir).splits.open(encoding="utf-8") as f:
        splits = json.load(f)
    cache = {}
    for split in ("train", "test"):
        U, V, A = build_desc(splits[split], workers=workers, data_dir=data_dir)
        cache[split] = {"U": U, "V": V, "A": A}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".part")
    torch.save(cache, tmp)
    os.replace(tmp, cache_path)
    return cache
