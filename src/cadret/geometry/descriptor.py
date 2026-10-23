from __future__ import annotations

from collections import Counter

import numpy as np

from .spectral import MAX_SYM, N_SPEC, N_TURN

MAX_PARTS = 4
LOOP_D = N_TURN + 8
PART_D = 2 * N_SPEC + LOOP_D + 16
GLOBAL_D = 24
DESC_DIM = GLOBAL_D + MAX_PARTS * PART_D


def _loop_block(l: dict) -> np.ndarray:
    b = np.zeros(LOOP_D, dtype=np.float32)
    b[:N_TURN] = l["turn"]
    s = l["sym"]
    b[N_TURN + 0] = l["iso"]
    b[N_TURN + 1] = l["aspect"]
    b[N_TURN + 2] = 1.0 if l["is_circle"] else 0.0
    b[N_TURN + 3] = min(l["n_curves"], 32) / 32.0
    b[N_TURN + 4] = l["frac_arc"]
    b[N_TURN + 5] = l["frac_line"]
    b[N_TURN + 6] = min(s, MAX_SYM) / MAX_SYM
    b[N_TURN + 7] = 1.0 / s if s >= 1 else 1.0
    return b


def _spec_norm(sp: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(sp, 0.0) / 10.0).astype(np.float32)


def _part_block(p: dict) -> np.ndarray:
    b = np.zeros(PART_D, dtype=np.float32)
    o = 0
    b[o : o + N_SPEC] = _spec_norm(p["big"]["spec"])
    o += N_SPEC
    b[o : o + N_SPEC] = _spec_norm(p["solid_spec"])
    o += N_SPEC
    b[o : o + LOOP_D] = _loop_block(p["big"]["outer"])
    o += LOOP_D
    f = p["big"]
    holes = f["holes"]
    extra = np.zeros(16, dtype=np.float32)
    extra[0] = min(p["b1"], 8) / 8.0
    extra[1] = min(p["n_faces"], 8) / 8.0
    extra[2] = np.tanh(p["slender"])
    extra[3] = p["asym"]
    extra[4] = f["fill"]
    extra[5] = min(f["hole_area_frac"], 1.0)
    extra[6 + min(max(p["op"], 0), 3)] = 1.0
    if holes:
        hi = np.array([h["iso"] for h in holes], dtype=np.float32)
        ha = np.array([h["area"] for h in holes], dtype=np.float32)
        extra[10] = hi.mean()
        extra[11] = hi.std()
        extra[12] = float(np.mean([h["is_circle"] for h in holes]))
        extra[13] = ha.std() / max(ha.mean(), 1e-09)
        extra[14] = min(len(holes), 8) / 8.0
        extra[15] = float(np.mean([h["aspect"] for h in holes]))
    b[o : o + 16] = extra
    return b


def descriptor_vector(d: dict) -> np.ndarray:
    v = np.zeros(DESC_DIM, dtype=np.float32)
    if not d:
        return v
    parts = sorted(d["parts"], key=lambda p: -p["big"]["outer"]["area"])
    g = np.zeros(GLOBAL_D, dtype=np.float32)
    g[0] = min(d["n_parts"], 8) / 8.0
    g[1] = min(d["b1"], 12) / 12.0
    g[2] = min(d["n_curves"], 64) / 64.0
    g[3] = min(d["n_circles"], 12) / 12.0
    s = d["max_sym"]
    g[4] = min(s, MAX_SYM) / MAX_SYM
    g[5] = 1.0 / s if s >= 1 else 1.0
    g[6] = 1.0 if s == 0 else 0.0
    ops = Counter(d["ops"])
    for i in range(4):
        g[7 + i] = min(ops.get(i, 0), 4) / 4.0
    sl = np.array([p["slender"] for p in parts], dtype=np.float32)
    ar = np.array([p["big"]["outer"]["area"] for p in parts], dtype=np.float32)
    iso = np.array([p["big"]["outer"]["iso"] for p in parts], dtype=np.float32)
    dep = np.array([p["depth"] for p in parts], dtype=np.float32)
    g[11] = np.tanh(sl.mean())
    g[12] = np.tanh(sl.std())
    g[13] = iso.mean()
    g[14] = iso.std()
    g[15] = ar.min() / max(ar.max(), 1e-09)
    g[16] = dep.std() / max(dep.mean(), 1e-09)
    g[17] = float(np.mean([p["big"]["outer"]["aspect"] for p in parts]))
    g[18] = float(np.mean([p["big"]["fill"] for p in parts]))
    g[19] = min(sum((p["n_faces"] for p in parts)), 12) / 12.0
    acc = np.zeros(4, dtype=np.float32)
    for p in parts:
        acc += _spec_norm(p["solid_spec"])[:4]
    g[20:24] = acc / max(len(parts), 1)
    v[:GLOBAL_D] = g
    for i, p in enumerate(parts[:MAX_PARTS]):
        o = GLOBAL_D + i * PART_D
        v[o : o + PART_D] = _part_block(p)
    return v
