from __future__ import annotations

import numpy as np

COORD_BASE = 11
END_PAD = 7
GRID = 255.0
ARC_SAMPLES = 12
CIRCLE_SAMPLES = 48


def _arc_through(s, m, e, n=ARC_SAMPLES):
    s, m, e = map(np.asarray, (s, m, e))
    ax, ay = s.astype(float)
    bx, by = m.astype(float)
    cx, cy = e.astype(float)
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-09:
        t = np.linspace(0, 1, n)[:, None]
        return s + t * (e - s)
    ux = (
        (ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)
    ) / d
    uy = (
        (ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)
    ) / d
    ctr = np.array([ux, uy])
    r = np.linalg.norm(s - ctr)
    a0, a1, a2 = (float(np.arctan2(p[1] - uy, p[0] - ux)) for p in (s, m, e))
    TAU = 2 * np.pi
    t1, t2 = ((a1 - a0) % TAU, (a2 - a0) % TAU)
    span = t2 if t1 <= t2 else -((a0 - a2) % TAU)
    ang = a0 + np.linspace(0, span, n)
    return ctr + r * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _circle_poly(ctr, pt, n=CIRCLE_SAMPLES):
    ctr, pt = (np.asarray(ctr, float), np.asarray(pt, float))
    r = float(np.linalg.norm(pt - ctr))
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (ctr + r * np.stack([np.cos(ang), np.sin(ang)], axis=1), r)


def parse_geometry(cad_vec, flag_vec=None, index_vec=None) -> list[dict]:
    v = np.asarray(cad_vec)
    keep = v[:, 0] != 0
    rows = v[keep]
    idx = np.asarray(index_vec)[keep] if index_vec is not None else np.zeros(len(rows), dtype=int)
    parts: dict[int, dict] = {}
    cur_pts: list[tuple[int, int]] = []
    loop_curves: list[list[tuple[int, int]]] = []
    face_loops: list[dict] = []
    sketch_faces: list[dict] = []
    ext_params: list[int] = []
    ext_op = -1
    in_extrude = False
    part_of_sketch = 0

    def finish_loop():
        if not loop_curves:
            return None
        if len(loop_curves) == 1 and len(loop_curves[0]) == 2:
            poly, r = _circle_poly(loop_curves[0][0], loop_curves[0][1])
            return {"poly": poly, "n_curves": 1, "is_circle": True, "types": [2], "radius": r}
        starts = [c[0] for c in loop_curves]
        segs, types = ([], [])
        for i, c in enumerate(loop_curves):
            s = np.asarray(c[0], float)
            e = np.asarray(starts[(i + 1) % len(starts)], float)
            if len(c) == 1:
                segs.append(np.stack([s, e]))
                types.append(0)
            else:
                segs.append(_arc_through(c[0], c[1], e))
                types.append(1)
        poly = np.concatenate([sg[:-1] for sg in segs], axis=0)
        return {
            "poly": poly,
            "n_curves": len(loop_curves),
            "is_circle": False,
            "types": types,
            "radius": 0.0,
        }

    for r, (a, b) in enumerate(rows[:, :2].tolist()):
        a, b = (int(a), int(b))
        if a == 1:
            continue
        if a >= COORD_BASE:
            if in_extrude:
                ext_params.append(a - COORD_BASE)
            else:
                cur_pts.append((a - COORD_BASE, b - COORD_BASE))
            continue
        if END_PAD <= a < COORD_BASE:
            ext_op = a - END_PAD
            continue
        if a == 5:
            if cur_pts:
                loop_curves.append(cur_pts)
                cur_pts = []
        elif a == 4:
            lp = finish_loop()
            if lp is not None:
                face_loops.append(lp)
            loop_curves = []
        elif a == 3:
            if face_loops:
                sketch_faces.append({"loops": face_loops})
            face_loops = []
        elif a == 2:
            in_extrude = True
            part_of_sketch = int(idx[r])
        elif a == 6:
            p = [x / GRID for x in ext_params]
            parts.setdefault(part_of_sketch, {"faces": [], "extrude": {}})
            parts[part_of_sketch] = {
                "faces": sketch_faces,
                "extrude": {
                    "e1": p[0] if len(p) > 0 else 0.0,
                    "e2": p[1] if len(p) > 1 else 0.0,
                    "offset": p[2:5] if len(p) >= 5 else [0.0, 0.0, 0.0],
                    "euler": p[5:8] if len(p) >= 8 else [0.0, 0.0, 0.0],
                    "scale": p[-1] if len(p) >= 9 else 1.0,
                    "op": ext_op,
                },
            }
            sketch_faces = []
            ext_params, ext_op, in_extrude = ([], -1, False)
    return [parts[k] for k in sorted(parts) if parts[k]["faces"]]


def signed_area(poly: np.ndarray) -> float:
    x, y = (poly[:, 0], poly[:, 1])
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def perimeter(poly: np.ndarray) -> float:
    d = np.roll(poly, -1, axis=0) - poly
    return float(np.linalg.norm(d, axis=1).sum())
