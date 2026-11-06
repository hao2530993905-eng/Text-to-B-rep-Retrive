from __future__ import annotations

from pathlib import Path

import torch

from .config import DATASET_LEVELS, DESCRIPTOR_DIM


def load_split(cache_path: Path, split: str, levels: tuple[str, ...]):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    dataset = cache.get("dataset")
    if dataset not in DATASET_LEVELS:
        raise ValueError("unknown dataset in cache")
    if split not in cache["splits"]:
        raise ValueError(f"{dataset} cache has no {split} split")
    data = cache["splits"][split]
    vectors = torch.as_tensor(data["vectors"], dtype=torch.float32)
    uids = list(data["uids"])
    texts = list(data["texts"])
    if vectors.ndim != 2 or vectors.shape[1] != DESCRIPTOR_DIM:
        raise ValueError(f"expected {DESCRIPTOR_DIM}-D descriptors")
    if len(uids) != len(vectors) or len(texts) != len(uids):
        raise ValueError("cache split fields have inconsistent lengths")
    indexed_rows = []
    for index, values in enumerate(texts):
        for level in levels:
            text = values.get(level)
            if isinstance(text, str) and text.strip():
                indexed_rows.append((index, level, text.strip()))
    if not indexed_rows:
        raise ValueError(f"no text rows in {split}")
    return {
        "dataset": dataset,
        "uids": uids,
        "vectors": vectors,
        "rows": indexed_rows,
    }
