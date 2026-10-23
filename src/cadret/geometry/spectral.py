from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from .primitives import perimeter, signed_area

N_SPEC = 12
N_TURN = 16
RASTER = 48
MAX_SYM = 12


def _inside(poly: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    x0, y0 = (poly[:, 0][:, None], poly[:, 1][:, None])
    x1, y1 = (np.roll(poly[:, 0], -1)[:, None], np.roll(poly[:, 1], -1)[:, None])
    Y, X = (py[None, :], px[None, :])
    straddles = (y0 <= Y) & (y1 > Y) | (y1 <= Y) & (y0 > Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (Y - y0) / (y1 - y0)
        xint = x0 + t * (x1 - x0)
    return (straddles & (xint > X)).sum(axis=0) % 2 == 1


def rasterize_face(face: dict, res: int = RASTER) -> np.ndarray:
    loops = [lp for lp in face["loops"] if len(lp["poly"]) >= 3]
    if not loops:
        return np.zeros((res, res), bool)
    areas = [abs(signed_area(lp["poly"])) for lp in loops]
    outer = loops[int(np.argmax(areas))]
    holes = [lp for i, lp in enumerate(loops) if lp is not outer]
    p = outer["poly"]
    mn, mx = (p.min(0), p.max(0))
    span = float(max(mx - mn))
    if span < 1e-09:
        return np.zeros((res, res), bool)
    pad = 1.5
    scale = (res - 2 * pad) / span
    ctr = 0.5 * (mn + mx)

    def to_grid(q):
        return (np.asarray(q, float) - ctr) * scale + res / 2.0

    gy, gx = np.mgrid[0:res, 0:res]
    sx, sy = (gx.ravel() + 0.5, gy.ravel() + 0.5)
    mask = _inside(to_grid(p), sx, sy).reshape(res, res)
    for h in holes:
        if len(h["poly"]) >= 3:
            mask &= ~_inside(to_grid(h["poly"]), sx, sy).reshape(res, res)
    return mask


def dirichlet_spectrum(mask: np.ndarray, k: int = N_SPEC) -> np.ndarray:
    out = np.zeros(k, dtype=np.float32)
    idx = -np.ones(mask.shape, dtype=np.int64)
    n = int(mask.sum())
    if n < k + 4:
        return out
    idx[mask] = np.arange(n)
    rows, cols, vals = ([np.arange(n)], [np.arange(n)], [np.full(n, 4.0)])
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        nb = np.roll(idx, shift, axis=axis)
        if axis == 0:
            nb[0 if shift == 1 else -1, :] = -1
        else:
            nb[:, 0 if shift == 1 else -1] = -1
        src, dst = (idx[mask], nb[mask])
        ok = dst >= 0
        rows.append(src[ok])
        cols.append(dst[ok])
        vals.append(np.full(int(ok.sum()), -1.0))
    A = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsc()
    try:
        w = eigsh(A, k=k, sigma=-1e-06, which="LM", return_eigenvectors=False, maxiter=5000)
    except Exception:
        return out
    w = np.sort(np.real(w))
    out[:] = np.clip(w * n, 0, 1000000.0)
    return out.astype(np.float32)


def solid_spectrum(mu: np.ndarray, depth: float, k: int = N_SPEC) -> np.ndarray:
    if depth <= 1e-06 or not np.any(mu):
        return np.zeros(k, dtype=np.float32)
    ax = (np.arange(1, k + 1) * np.pi / depth) ** 2
    lam = np.sort((mu[:, None] + ax[None, :]).ravel())[:k]
    return lam.astype(np.float32)


def _turning_measure(poly: np.ndarray):
    d = np.roll(poly, -1, axis=0) - poly
    seg = np.linalg.norm(d, axis=1)
    keep = seg > 1e-09
    d, seg = (d[keep], seg[keep])
    if len(seg) < 3:
        return None
    ang = np.arctan2(d[:, 1], d[:, 0])
    dth = (ang - np.roll(ang, 1) + np.pi) % (2 * np.pi) - np.pi
    s = np.cumsum(np.roll(seg, 1))
    L = float(seg.sum())
    if L < 1e-09:
        return None
    return (dth, (s - s[0]) / L, L)


def _turning_fourier(poly: np.ndarray, n_harm: int) -> np.ndarray:
    tm = _turning_measure(poly)
    if tm is None:
        return np.zeros(n_harm, dtype=np.float64)
    dth, u, _ = tm
    k = np.arange(1, n_harm + 1)[:, None]
    return np.abs((dth[None, :] * np.exp(-2j * np.pi * k * u[None, :])).sum(axis=1))


def turning_spectrum(poly: np.ndarray, n_harm: int = N_TURN) -> np.ndarray:
    out = np.zeros(n_harm, dtype=np.float32)
    mag = _turning_fourier(poly, n_harm)
    nrm = np.linalg.norm(mag)
    if nrm > 1e-09:
        out[:] = mag / nrm
    return out


def symmetry_order(poly: np.ndarray, max_n: int = MAX_SYM, tol: float = 0.98) -> int:
    K = max_n * 2
    mag = _turning_fourier(poly, K)
    p = mag**2
    tot = float(p.sum())
    tm = _turning_measure(poly)
    if tm is None:
        return 1
    winding = float(np.abs(tm[0].sum()))
    if tot < (0.02 * winding) ** 2:
        return 0
    k = np.arange(1, K + 1)
    best = 1
    for n in range(2, max_n + 1):
        if p[k % n == 0].sum() / tot >= tol:
            best = n
    return best


def loop_descriptor(lp: dict) -> dict:
    poly = lp["poly"]
    A = signed_area(poly)
    P = perimeter(poly)
    iso = 4 * np.pi * abs(A) / P**2 if P > 1e-09 else 0.0
    n_arc = sum((1 for t in lp["types"] if t == 1))
    n_line = sum((1 for t in lp["types"] if t == 0))
    return {
        "area": abs(A),
        "orient": 1.0 if A > 0 else -1.0,
        "iso": float(np.clip(iso, 0, 1.05)),
        "turn": turning_spectrum(poly),
        "sym": symmetry_order(poly),
        "is_circle": bool(lp["is_circle"]),
        "n_curves": lp["n_curves"],
        "frac_arc": n_arc / max(lp["n_curves"], 1),
        "frac_line": n_line / max(lp["n_curves"], 1),
        "aspect": _aspect(poly),
    }


def _aspect(poly: np.ndarray) -> float:
    c = poly - poly.mean(0)
    cov = c.T @ c / max(len(c), 1)
    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 1e-12, None)
    return float(np.sqrt(w[0] / w[1]))


def face_descriptor(face: dict) -> dict:
    loops = [loop_descriptor(lp) for lp in face["loops"] if len(lp["poly"]) >= 3]
    if not loops:
        return {}
    areas = [l["area"] for l in loops]
    o = int(np.argmax(areas))
    outer = loops[o]
    holes = [l for i, l in enumerate(loops) if i != o]
    mask = rasterize_face(face)
    return {
        "spec": dirichlet_spectrum(mask),
        "outer": outer,
        "holes": holes,
        "b1": len(holes),
        "hole_area_frac": sum((h["area"] for h in holes)) / max(outer["area"], 1e-09),
        "fill": float(mask.sum()) / max(mask.size, 1),
        "n_loops": len(loops),
    }


def part_descriptor(part: dict) -> dict:
    faces = [f for f in (face_descriptor(f) for f in part["faces"]) if f]
    if not faces:
        return {}
    e = part["extrude"]
    depth = float(e["e1"]) + float(e["e2"])
    big = max(faces, key=lambda f: f["outer"]["area"])
    return {
        "faces": faces,
        "big": big,
        "solid_spec": solid_spectrum(big["spec"], depth),
        "depth": depth,
        "asym": abs(float(e["e1"]) - float(e["e2"])) / max(depth, 1e-09),
        "op": int(e["op"]),
        "scale": float(e["scale"]),
        "b1": sum((f["b1"] for f in faces)),
        "n_faces": len(faces),
        "sym": big["outer"]["sym"],
        "slender": depth / max(np.sqrt(big["outer"]["area"]) / 255.0, 1e-06),
    }


def describe(cad_vec, flag_vec=None, index_vec=None) -> dict:
    from .primitives import parse_geometry

    parts = [
        p for p in (part_descriptor(p) for p in parse_geometry(cad_vec, flag_vec, index_vec)) if p
    ]
    if not parts:
        return {}
    return {
        "parts": parts,
        "n_parts": len(parts),
        "b1": sum((p["b1"] for p in parts)),
        "ops": [p["op"] for p in parts],
        "n_curves": sum(
            (l["n_curves"] for p in parts for f in p["faces"] for l in [f["outer"]] + f["holes"])
        ),
        "n_circles": sum(
            (
                1
                for p in parts
                for f in p["faces"]
                for l in [f["outer"]] + f["holes"]
                if l["is_circle"]
            )
        ),
        "max_sym": max((p["sym"] for p in parts)),
    }
