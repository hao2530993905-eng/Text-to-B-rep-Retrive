from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import torch


def to_full(obj):
    for _ in range(3):
        if isinstance(obj, np.ndarray) and obj.shape == ():
            obj = obj.item()
        if isinstance(obj, dict) and "vec" in obj and ("cad_vec" not in obj):
            obj = obj["vec"]
            continue
        break
    if not isinstance(obj, dict):
        return (_to_vec(obj), None, None)

    def arr(k):
        v = obj.get(k)
        if isinstance(v, torch.Tensor):
            v = v.numpy()
        return np.asarray(v) if v is not None else None

    return (_to_vec(obj), arr("flag_vec"), arr("index_vec"))


def _to_vec(obj):
    for _ in range(3):
        if isinstance(obj, np.ndarray) and obj.shape == ():
            obj = obj.item()
        if isinstance(obj, dict):
            if "cad_vec" in obj:
                obj = obj["cad_vec"]
                break
            if "vec" in obj:
                obj = obj["vec"]
                continue
        break
    if isinstance(obj, torch.Tensor):
        obj = obj.numpy()
    if not isinstance(obj, np.ndarray) or obj.ndim != 2:
        return None
    return obj


class CADSeqStore:
    def __init__(self, root: str | None, zip_path: str | None = None):
        env = os.environ.get("CADSEQ_ZIP")
        if env and os.path.exists(env):
            root, zip_path = (None, env)
        self.root = Path(root) if root and os.path.isdir(root) else None
        self.zip_path = zip_path if zip_path and os.path.exists(zip_path) else None
        self._zf = None
        if self.root is None and self.zip_path is None:
            raise FileNotFoundError("need either an extracted cad_seq dir or cad_seq.zip")

    def _zip(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    def close(self):
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_zf"] = None
        return st

    def load_full(self, uid: str):
        raw = self._raw(uid)
        return (None, None, None) if raw is None else to_full(raw)

    def load(self, uid: str):
        raw = self._raw(uid)
        return None if raw is None else _to_vec(raw)

    def _raw(self, uid: str):
        name = uid.split("/")[-1]
        rel = f"cad_seq/{uid}/seq/{name}.pth"
        if self.root is not None:
            for p in (self.root / rel, self.root / uid / "seq" / f"{name}.pth"):
                if p.exists():
                    try:
                        return torch.load(p, map_location="cpu", weights_only=False)
                    except (RuntimeError, EOFError, OSError):
                        pass
        if self.zip_path is None:
            return None
        try:
            with self._zip().open(rel) as fh:
                return torch.load(fh, map_location="cpu", weights_only=False)
        except (KeyError, FileNotFoundError, RuntimeError, EOFError, OSError):
            return None
