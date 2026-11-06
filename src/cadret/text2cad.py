from __future__ import annotations

import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import DATASET_LEVELS, Text2CADPaths
from .geometry import describe_sequence, descriptor_vector

_STORE = None


def _to_array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return None if value is None else np.asarray(value)


def unpack_sequence(value):
    for _ in range(3):
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if isinstance(value, dict) and "vec" in value and "cad_vec" not in value:
            value = value["vec"]
            continue
        break
    if not isinstance(value, dict):
        cad_vec = _to_array(value)
        return (cad_vec, None) if cad_vec is not None and cad_vec.ndim == 2 else (None, None)
    cad_vec = _to_array(value.get("cad_vec"))
    if cad_vec is None and "vec" in value:
        nested = value["vec"]
        cad_vec = _to_array(nested.get("cad_vec") if isinstance(nested, dict) else nested)
    if cad_vec is None or cad_vec.ndim != 2:
        return None, None
    return cad_vec, _to_array(value.get("index_vec"))


class SequenceStore:
    def __init__(self, extracted_root: Path, archive_path: Path):
        self.extracted_root = extracted_root if extracted_root.is_dir() else None
        self.archive_path = archive_path if archive_path.is_file() else None
        if self.extracted_root is None and self.archive_path is None:
            raise FileNotFoundError("Text2CAD cad_seq.zip or extracted cad_seq is required")
        self._archive = None

    def _zip(self):
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path)
        return self._archive

    def load(self, uid: str):
        name = uid.rsplit("/", 1)[-1]
        member = f"cad_seq/{uid}/seq/{name}.pth"
        if self.extracted_root is not None:
            for path in (
                self.extracted_root / uid / "seq" / f"{name}.pth",
                self.extracted_root.parent / member,
            ):
                if path.is_file():
                    try:
                        return unpack_sequence(
                            torch.load(path, map_location="cpu", weights_only=False)
                        )
                    except (EOFError, OSError, RuntimeError):
                        break
        if self.archive_path is None:
            return None, None
        try:
            with self._zip().open(member) as handle:
                return unpack_sequence(torch.load(handle, map_location="cpu", weights_only=False))
        except (EOFError, KeyError, OSError, RuntimeError):
            return None, None

    def close(self):
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None
        return state


def _initialize_store(root: str):
    global _STORE
    paths = Text2CADPaths(Path(root))
    _STORE = SequenceStore(paths.sequences, paths.archive)


def _describe_uid(uid: str):
    cad_vec, index_vec = _STORE.load(uid)
    if cad_vec is None:
        return None
    try:
        vector = descriptor_vector(describe_sequence(cad_vec, index_vec))
    except Exception:
        return None
    if not np.isfinite(vector).all():
        return None
    return uid, vector


def _load_text(data_root: Path):
    frame = pd.read_csv(Text2CADPaths(data_root).descriptions)
    result = {}
    for row in frame.itertuples(index=False):
        values = row._asdict()
        uid = str(values["uid"])
        result[uid] = {
            level: values[level].strip()
            for level in DATASET_LEVELS["text2cad"]
            if isinstance(values.get(level), str) and values[level].strip()
        }
    return result


def prepare_text2cad(data_root: Path, output: Path, workers: int) -> dict:
    paths = Text2CADPaths(data_root)
    required = (paths.descriptions, paths.splits)
    missing = [str(path) for path in required if not path.is_file()]
    if not paths.archive.is_file() and not paths.sequences.is_dir():
        missing.append(str(paths.archive))
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if workers < 1:
        raise ValueError("workers must be positive")
    splits = json.loads(paths.splits.read_text(encoding="utf-8"))
    text = _load_text(data_root)
    cache_splits = {}
    manifest_splits = {}
    for split in ("train", "test"):
        uids = []
        vectors = []
        texts = []
        expected = splits[split]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_store,
            initargs=(str(data_root),),
        ) as executor:
            results = executor.map(_describe_uid, expected, chunksize=16)
            for result in tqdm(results, total=len(expected), desc=split):
                if result is None:
                    continue
                uid, vector = result
                uids.append(uid)
                vectors.append(vector)
                texts.append(text.get(uid, {}))
        if not uids:
            raise ValueError(f"No readable sequences in Text2CAD {split}")
        cache_splits[split] = {
            "uids": uids,
            "vectors": np.stack(vectors).astype(np.float32),
            "texts": texts,
        }
        manifest_splits[split] = {
            "models": len(uids),
            "skipped": len(expected) - len(uids),
        }
    cache = {
        "dataset": "text2cad",
        "levels": list(DATASET_LEVELS["text2cad"]),
        "splits": cache_splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    torch.save(cache, temporary)
    os.replace(temporary, output)
    manifest = {
        "dataset": "text2cad",
        "source": "Text2CAD v1.0 official train/test split",
        "descriptor_dim": int(next(iter(cache_splits.values()))["vectors"].shape[1]),
        "splits": manifest_splits,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
