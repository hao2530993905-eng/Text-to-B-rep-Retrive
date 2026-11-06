from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from .config import DESCRIPTOR_DIM

N_SPEC = 12
N_TURN = 16
RASTER = 48
MAX_SYM = 12
MAX_PARTS = 4
LOOP_DIM = N_TURN + 8
PART_DIM = 2 * N_SPEC + LOOP_DIM + 16
GLOBAL_DIM = 24
COORD_BASE = 11
END_PAD = 7
GRID = 255.0
ARC_SAMPLES = 12
CIRCLE_SAMPLES = 48


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def perimeter(poly: np.ndarray) -> float:
    distance = np.roll(poly, -1, axis=0) - poly
    return float(np.linalg.norm(distance, axis=1).sum())


def _arc_through(
    start: np.ndarray,
    middle: np.ndarray,
    end: np.ndarray,
    samples: int = ARC_SAMPLES,
) -> np.ndarray:
    start, middle, end = (np.asarray(value, dtype=float) for value in (start, middle, end))
    ax, ay = start
    bx, by = middle
    cx, cy = end
    determinant = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(determinant) < 1e-9:
        t = np.linspace(0, 1, samples)[:, None]
        return start + t * (end - start)
    center_x = (
        (ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)
    ) / determinant
    center_y = (
        (ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)
    ) / determinant
    center = np.array([center_x, center_y], dtype=float)
    radius = np.linalg.norm(start - center)
    angles = [
        float(np.arctan2(point[1] - center_y, point[0] - center_x))
        for point in (start, middle, end)
    ]
    first = (angles[1] - angles[0]) % (2 * np.pi)
    second = (angles[2] - angles[0]) % (2 * np.pi)
    span = second if first <= second else -((angles[0] - angles[2]) % (2 * np.pi))
    samples_at = angles[0] + np.linspace(0, span, samples)
    return center + radius * np.stack([np.cos(samples_at), np.sin(samples_at)], axis=1)


def _circle(center: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.asarray(center, dtype=float)
    radius = float(np.linalg.norm(np.asarray(point, dtype=float) - center))
    angles = np.linspace(0, 2 * np.pi, CIRCLE_SAMPLES, endpoint=False)
    points = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return points, radius


def parse_sequence(
    cad_vec: np.ndarray,
    index_vec: np.ndarray | None = None,
) -> list[dict]:
    values = np.asarray(cad_vec)
    rows = values[values[:, 0] != 0]
    indices = (
        np.asarray(index_vec)[values[:, 0] != 0]
        if index_vec is not None
        else np.zeros(len(rows), dtype=int)
    )
    parts: dict[int, dict] = {}
    points_buffer: list[tuple[int, int]] = []
    loop_curves: list[list[tuple[int, int]]] = []
    face_loops: list[dict] = []
    sketch_faces: list[dict] = []
    extrude_values: list[int] = []
    operation = -1
    in_extrude = False
    sketch_id = 0

    def finish_loop() -> dict | None:
        if not loop_curves:
            return None
        if len(loop_curves) == 1 and len(loop_curves[0]) == 2:
            points, radius = _circle(loop_curves[0][0], loop_curves[0][1])
            return {
                "poly": points,
                "n_curves": 1,
                "is_circle": True,
                "types": [2],
                "radius": radius,
            }
        starts = [curve[0] for curve in loop_curves]
        segments = []
        curve_types = []
        for index, curve in enumerate(loop_curves):
            start = np.asarray(curve[0], dtype=float)
            end = np.asarray(starts[(index + 1) % len(starts)], dtype=float)
            if len(curve) == 1:
                segments.append(np.stack([start, end]))
                curve_types.append(0)
            else:
                segments.append(_arc_through(curve[0], curve[1], end))
                curve_types.append(1)
        return {
            "poly": np.concatenate([segment[:-1] for segment in segments], axis=0),
            "n_curves": len(loop_curves),
            "is_circle": False,
            "types": curve_types,
            "radius": 0.0,
        }

    for row, (first, second) in enumerate(rows[:, :2].tolist()):
        first, second = int(first), int(second)
        if first == 1:
            continue
        if first >= COORD_BASE:
            point = (first - COORD_BASE, second - COORD_BASE)
            if in_extrude:
                extrude_values.append(first - COORD_BASE)
            else:
                points_buffer.append(point)
            continue
        if END_PAD <= first < COORD_BASE:
            operation = first - END_PAD
            continue
        if first == 5:
            if points_buffer:
                loop_curves.append(points_buffer)
                points_buffer = []
        elif first == 4:
            loop = finish_loop()
            if loop is not None:
                face_loops.append(loop)
            loop_curves = []
        elif first == 3:
            if face_loops:
                sketch_faces.append({"loops": face_loops})
            face_loops = []
        elif first == 2:
            in_extrude = True
            sketch_id = int(indices[row])
        elif first == 6:
            converted = [value / GRID for value in extrude_values]
            parts[sketch_id] = {
                "faces": sketch_faces,
                "extrude": {
                    "e1": converted[0] if converted else 0.0,
                    "e2": converted[1] if len(converted) > 1 else 0.0,
                    "offset": converted[2:5] if len(converted) >= 5 else [0.0, 0.0, 0.0],
                    "euler": converted[5:8] if len(converted) >= 8 else [0.0, 0.0, 0.0],
                    "scale": converted[-1] if len(converted) >= 9 else 1.0,
                    "op": operation,
                },
            }
            sketch_faces = []
            extrude_values = []
            operation = -1
            in_extrude = False
    return [parts[key] for key in sorted(parts) if parts[key]["faces"]]


def _inside(poly: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, y0 = poly[:, 0][:, None], poly[:, 1][:, None]
    x1 = np.roll(poly[:, 0], -1)[:, None]
    y1 = np.roll(poly[:, 1], -1)[:, None]
    grid_y, grid_x = y[None, :], x[None, :]
    straddles = ((y0 <= grid_y) & (y1 > grid_y)) | ((y1 <= grid_y) & (y0 > grid_y))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (grid_y - y0) / (y1 - y0)
        crossings = x0 + t * (x1 - x0)
    return (straddles & (crossings > grid_x)).sum(axis=0) % 2 == 1


def rasterize_face(face: dict, resolution: int = RASTER) -> np.ndarray:
    loops = [loop for loop in face["loops"] if len(loop["poly"]) >= 3]
    if not loops:
        return np.zeros((resolution, resolution), dtype=bool)
    outer = max(loops, key=lambda loop: abs(signed_area(loop["poly"])))
    holes = [loop for loop in loops if loop is not outer]
    points = outer["poly"]
    minimum, maximum = points.min(0), points.max(0)
    span = float(max(maximum - minimum))
    if span < 1e-9:
        return np.zeros((resolution, resolution), dtype=bool)
    padding = 1.5
    scale = (resolution - 2 * padding) / span
    center = 0.5 * (minimum + maximum)

    def transform(values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - center) * scale + resolution / 2.0

    grid_y, grid_x = np.mgrid[0:resolution, 0:resolution]
    sample_x, sample_y = grid_x.ravel() + 0.5, grid_y.ravel() + 0.5
    mask = _inside(transform(points), sample_x, sample_y).reshape(resolution, resolution)
    for hole in holes:
        if len(hole["poly"]) >= 3:
            mask &= ~_inside(transform(hole["poly"]), sample_x, sample_y).reshape(
                resolution, resolution
            )
    return mask


def dirichlet_spectrum(mask: np.ndarray, count: int = N_SPEC) -> np.ndarray:
    result = np.zeros(count, dtype=np.float32)
    indices = -np.ones(mask.shape, dtype=np.int64)
    pixels = int(mask.sum())
    if pixels < count + 4:
        return result
    indices[mask] = np.arange(pixels)
    rows, columns, values = (
        [np.arange(pixels)],
        [np.arange(pixels)],
        [np.full(pixels, 4.0)],
    )
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        neighbor = np.roll(indices, shift, axis=axis)
        if axis == 0:
            neighbor[0 if shift == 1 else -1, :] = -1
        else:
            neighbor[:, 0 if shift == 1 else -1] = -1
        source, destination = indices[mask], neighbor[mask]
        valid = destination >= 0
        rows.append(source[valid])
        columns.append(destination[valid])
        values.append(np.full(int(valid.sum()), -1.0))
    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(pixels, pixels),
    ).tocsc()
    try:
        spectrum = eigsh(
            matrix,
            k=count,
            sigma=-1e-6,
            which="LM",
            return_eigenvectors=False,
            maxiter=5000,
        )
    except Exception:
        return result
    result[:] = np.clip(np.sort(np.real(spectrum)) * pixels, 0, 1_000_000.0)
    return result.astype(np.float32)


def solid_spectrum(surface: np.ndarray, depth: float, count: int = N_SPEC) -> np.ndarray:
    if depth <= 1e-6 or not np.any(surface):
        return np.zeros(count, dtype=np.float32)
    axial = (np.arange(1, count + 1) * np.pi / depth) ** 2
    return np.sort((surface[:, None] + axial[None, :]).ravel())[:count].astype(np.float32)


def _turning_measure(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    difference = np.roll(poly, -1, axis=0) - poly
    length = np.linalg.norm(difference, axis=1)
    valid = length > 1e-9
    difference, length = difference[valid], length[valid]
    if len(length) < 3:
        return None
    angle = np.arctan2(difference[:, 1], difference[:, 0])
    delta = (angle - np.roll(angle, 1) + np.pi) % (2 * np.pi) - np.pi
    position = np.cumsum(np.roll(length, 1))
    total = float(length.sum())
    if total < 1e-9:
        return None
    return delta, (position - position[0]) / total, total


def _turning_fourier(poly: np.ndarray, harmonics: int) -> np.ndarray:
    measure = _turning_measure(poly)
    if measure is None:
        return np.zeros(harmonics, dtype=np.float64)
    delta, position, _ = measure
    indices = np.arange(1, harmonics + 1)[:, None]
    return np.abs((delta[None, :] * np.exp(-2j * np.pi * indices * position[None, :])).sum(axis=1))


def turning_spectrum(poly: np.ndarray, harmonics: int = N_TURN) -> np.ndarray:
    result = np.zeros(harmonics, dtype=np.float32)
    magnitude = _turning_fourier(poly, harmonics)
    norm = np.linalg.norm(magnitude)
    if norm > 1e-9:
        result[:] = magnitude / norm
    return result


def symmetry_order(poly: np.ndarray, maximum: int = MAX_SYM) -> int:
    harmonics = maximum * 2
    power = _turning_fourier(poly, harmonics) ** 2
    total = float(power.sum())
    measure = _turning_measure(poly)
    if measure is None:
        return 1
    winding = float(np.abs(measure[0].sum()))
    if total < (0.02 * winding) ** 2:
        return 0
    values = np.arange(1, harmonics + 1)
    order = 1
    for candidate in range(2, maximum + 1):
        if power[values % candidate == 0].sum() / total >= 0.98:
            order = candidate
    return order


def _aspect(poly: np.ndarray) -> float:
    centered = poly - poly.mean(0)
    covariance = centered.T @ centered / max(len(poly), 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 1e-12, None)
    return float(np.sqrt(eigenvalues[0] / eigenvalues[1]))


def loop_descriptor(loop: dict) -> dict:
    poly = loop["poly"]
    area = signed_area(poly)
    length = perimeter(poly)
    isoperimetric = 4 * np.pi * abs(area) / length**2 if length > 1e-9 else 0.0
    arc_count = sum(value == 1 for value in loop["types"])
    line_count = sum(value == 0 for value in loop["types"])
    return {
        "area": abs(area),
        "orient": 1.0 if area > 0 else -1.0,
        "iso": float(np.clip(isoperimetric, 0, 1.05)),
        "turn": turning_spectrum(poly),
        "sym": symmetry_order(poly),
        "is_circle": bool(loop["is_circle"]),
        "n_curves": loop["n_curves"],
        "frac_arc": arc_count / max(loop["n_curves"], 1),
        "frac_line": line_count / max(loop["n_curves"], 1),
        "aspect": _aspect(poly),
    }


def face_descriptor(face: dict) -> dict:
    loops = [loop_descriptor(loop) for loop in face["loops"] if len(loop["poly"]) >= 3]
    if not loops:
        return {}
    outer = max(loops, key=lambda loop: loop["area"])
    holes = [loop for loop in loops if loop is not outer]
    mask = rasterize_face(face)
    return {
        "spec": dirichlet_spectrum(mask),
        "outer": outer,
        "holes": holes,
        "b1": len(holes),
        "hole_area_frac": sum(hole["area"] for hole in holes) / max(outer["area"], 1e-9),
        "fill": float(mask.sum()) / max(mask.size, 1),
        "n_loops": len(loops),
    }


def part_descriptor(part: dict) -> dict:
    faces = [face for face in (face_descriptor(item) for item in part["faces"]) if face]
    if not faces:
        return {}
    extrude = part["extrude"]
    depth = float(extrude["e1"]) + float(extrude["e2"])
    largest = max(faces, key=lambda face: face["outer"]["area"])
    return {
        "faces": faces,
        "big": largest,
        "solid_spec": solid_spectrum(largest["spec"], depth),
        "depth": depth,
        "asym": abs(float(extrude["e1"]) - float(extrude["e2"])) / max(depth, 1e-9),
        "op": int(extrude["op"]),
        "scale": float(extrude["scale"]),
        "b1": sum(face["b1"] for face in faces),
        "n_faces": len(faces),
        "sym": largest["outer"]["sym"],
        "slender": depth / max(np.sqrt(largest["outer"]["area"]) / 255.0, 1e-6),
    }


def describe_sequence(
    cad_vec: np.ndarray,
    index_vec: np.ndarray | None = None,
) -> dict:
    parts = [
        part
        for part in (part_descriptor(item) for item in parse_sequence(cad_vec, index_vec))
        if part
    ]
    if not parts:
        return {}
    curves = [
        loop for part in parts for face in part["faces"] for loop in [face["outer"], *face["holes"]]
    ]
    return {
        "parts": parts,
        "n_parts": len(parts),
        "b1": sum(part["b1"] for part in parts),
        "ops": [part["op"] for part in parts],
        "n_curves": sum(loop["n_curves"] for loop in curves),
        "n_circles": sum(loop["is_circle"] for loop in curves),
        "max_sym": max(part["sym"] for part in parts),
    }


def _normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(spectrum, 0.0) / 10.0).astype(np.float32)


def _loop_block(loop: dict) -> np.ndarray:
    block = np.zeros(LOOP_DIM, dtype=np.float32)
    block[:N_TURN] = loop["turn"]
    order = loop["sym"]
    block[N_TURN] = loop["iso"]
    block[N_TURN + 1] = loop["aspect"]
    block[N_TURN + 2] = float(loop["is_circle"])
    block[N_TURN + 3] = min(loop["n_curves"], 32) / 32.0
    block[N_TURN + 4] = loop["frac_arc"]
    block[N_TURN + 5] = loop["frac_line"]
    block[N_TURN + 6] = min(order, MAX_SYM) / MAX_SYM
    block[N_TURN + 7] = 1.0 / order if order >= 1 else 1.0
    return block


def _part_block(part: dict) -> np.ndarray:
    block = np.zeros(PART_DIM, dtype=np.float32)
    block[:N_SPEC] = _normalize_spectrum(part["big"]["spec"])
    block[N_SPEC : 2 * N_SPEC] = _normalize_spectrum(part["solid_spec"])
    offset = 2 * N_SPEC
    block[offset : offset + LOOP_DIM] = _loop_block(part["big"]["outer"])
    offset += LOOP_DIM
    values = np.zeros(16, dtype=np.float32)
    values[0] = min(part["b1"], 8) / 8.0
    values[1] = min(part["n_faces"], 8) / 8.0
    values[2] = np.tanh(part["slender"])
    values[3] = part["asym"]
    values[4] = part["big"]["fill"]
    values[5] = min(part["big"]["hole_area_frac"], 1.0)
    values[6 + min(max(part["op"], 0), 3)] = 1.0
    holes = part["big"]["holes"]
    if holes:
        hole_iso = np.array([hole["iso"] for hole in holes], dtype=np.float32)
        hole_area = np.array([hole["area"] for hole in holes], dtype=np.float32)
        values[10] = hole_iso.mean()
        values[11] = hole_iso.std()
        values[12] = np.mean([hole["is_circle"] for hole in holes])
        values[13] = hole_area.std() / max(hole_area.mean(), 1e-9)
        values[14] = min(len(holes), 8) / 8.0
        values[15] = np.mean([hole["aspect"] for hole in holes])
    block[offset : offset + 16] = values
    return block


def descriptor_vector(description: dict) -> np.ndarray:
    vector = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    if not description:
        return vector
    parts = sorted(description["parts"], key=lambda part: -part["big"]["outer"]["area"])
    global_values = np.zeros(GLOBAL_DIM, dtype=np.float32)
    global_values[0] = min(description["n_parts"], 8) / 8.0
    global_values[1] = min(description["b1"], 12) / 12.0
    global_values[2] = min(description["n_curves"], 64) / 64.0
    global_values[3] = min(description["n_circles"], 12) / 12.0
    order = description["max_sym"]
    global_values[4] = min(order, MAX_SYM) / MAX_SYM
    global_values[5] = 1.0 / order if order >= 1 else 1.0
    global_values[6] = float(order == 0)
    operations = Counter(description["ops"])
    for operation in range(4):
        global_values[7 + operation] = min(operations.get(operation, 0), 4) / 4.0
    slenderness = np.array([part["slender"] for part in parts], dtype=np.float32)
    areas = np.array([part["big"]["outer"]["area"] for part in parts], dtype=np.float32)
    isoperimetric = np.array([part["big"]["outer"]["iso"] for part in parts], dtype=np.float32)
    depths = np.array([part["depth"] for part in parts], dtype=np.float32)
    global_values[11] = np.tanh(slenderness.mean())
    global_values[12] = np.tanh(slenderness.std())
    global_values[13] = isoperimetric.mean()
    global_values[14] = isoperimetric.std()
    global_values[15] = areas.min() / max(areas.max(), 1e-9)
    global_values[16] = depths.std() / max(depths.mean(), 1e-9)
    global_values[17] = np.mean([part["big"]["outer"]["aspect"] for part in parts])
    global_values[18] = np.mean([part["big"]["fill"] for part in parts])
    global_values[19] = min(sum(part["n_faces"] for part in parts), 12) / 12.0
    accumulated = np.zeros(4, dtype=np.float32)
    for part in parts:
        accumulated += _normalize_spectrum(part["solid_spec"])[:4]
    global_values[20:24] = accumulated / max(len(parts), 1)
    vector[:GLOBAL_DIM] = global_values
    for index, part in enumerate(parts[:MAX_PARTS]):
        offset = GLOBAL_DIM + index * PART_DIM
        vector[offset : offset + PART_DIM] = _part_block(part)
    return vector


def prism_descriptor(faces: list[dict], depth: float) -> np.ndarray:
    part = part_descriptor(
        {
            "faces": faces,
            "extrude": {
                "e1": depth,
                "e2": depth,
                "op": 0,
                "scale": 1.0,
            },
        }
    )
    description = {
        "parts": [part],
        "n_parts": 1,
        "b1": part["b1"],
        "ops": [0],
        "n_curves": sum(
            loop["n_curves"] for face in part["faces"] for loop in [face["outer"], *face["holes"]]
        ),
        "n_circles": sum(
            loop["is_circle"] for face in part["faces"] for loop in [face["outer"], *face["holes"]]
        ),
        "max_sym": part["sym"],
    }
    return descriptor_vector(description)
