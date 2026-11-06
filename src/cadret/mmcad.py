from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import DATASET_LEVELS
from .geometry import prism_descriptor

_ROOT = None
_STAGING = None


def _volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return abs(properties.Mass())


def _step_path(row):
    source_member = Path(str(row.step_member))
    archive_name = Path(str(row.step_archive)).name
    archive_stem = Path(archive_name).stem
    for candidate in (
        _ROOT / source_member,
        _ROOT / source_member.name,
        _ROOT / "steps" / archive_stem / source_member.name,
        _ROOT / archive_stem / source_member.name,
    ):
        if candidate.is_file():
            return candidate
    archive = next(
        (
            candidate
            for candidate in (
                _ROOT / archive_name,
                _ROOT / "archives" / archive_name,
                _ROOT / f"{archive_stem}.tar",
                _ROOT / "archives" / f"{archive_stem}.tar",
            )
            if candidate.is_file()
        ),
        None,
    )
    if archive is not None:
        target = _STAGING / source_member.name
        if not target.is_file():
            with tarfile.open(archive, mode="r:*") as handle:
                archive_member = next(
                    (
                        item
                        for item in handle
                        if item.isfile() and Path(item.name).name == source_member.name
                    ),
                    None,
                )
                if archive_member is None:
                    return None
                source = handle.extractfile(archive_member)
                if source is None:
                    return None
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".part")
                temporary.write_bytes(source.read())
                os.replace(temporary, target)
        return target
    return None


def _describe_step(row):
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line, GeomAbs_Plane
    from OCP.gp import gp_Vec
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    path = _step_path(row)
    if path is None:
        return None
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        return None
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    solids = []
    while explorer.More():
        solids.append(TopoDS.Solid_s(explorer.Current()))
        explorer.Next()
    if len(solids) != 1:
        return None
    shape = solids[0]
    if not BRepCheck_Analyzer(shape).IsValid():
        return None
    volume = _volume(shape)
    if volume < 1e-12:
        return None
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    caps = []
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Plane:
            properties = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, properties)
            caps.append((properties.Mass(), face, surface.Plane()))
        explorer.Next()
    selected = None
    for area, face, plane in sorted(caps, key=lambda item: -item[0])[:6]:
        if area < 1e-12:
            continue
        depth = volume / area
        axis = plane.Axis().Direction()
        for sign in (1, -1):
            prism = BRepPrimAPI_MakePrism(face, gp_Vec(axis).Multiplied(sign * depth)).Shape()
            prism_volume = _volume(prism)
            if abs(prism_volume - volume) / volume > 1e-5:
                continue
            common = BRepAlgoAPI_Common(shape, prism)
            if not common.IsDone():
                continue
            error = (volume + prism_volume - 2 * _volume(common.Shape())) / volume
            if -1e-5 <= error < 1e-5:
                selected = (area, face, plane, depth)
                break
        if selected is not None:
            break
    if selected is None:
        return None
    area, face, plane, depth = selected
    placement = plane.Position()
    origin = np.array(placement.Location().Coord())
    x_axis = np.array(placement.XDirection().Coord())
    y_axis = np.array(placement.YDirection().Coord())
    loops = []
    explorer = TopExp_Explorer(face, TopAbs_WIRE)
    while explorer.More():
        wire = TopoDS.Wire_s(explorer.Current())
        wire_explorer = BRepTools_WireExplorer(wire, face)
        points = []
        kinds = []
        while wire_explorer.More():
            edge = TopoDS.Edge_s(wire_explorer.Current())
            curve = BRepAdaptor_Curve(edge)
            first, last = curve.FirstParameter(), curve.LastParameter()
            if not math.isfinite(first) or not math.isfinite(last):
                return None
            curve_type = curve.GetType()
            samples = (
                2 if curve_type == GeomAbs_Line else 49 if curve_type == GeomAbs_Circle else 97
            )
            parameters = np.linspace(first, last, samples)
            if edge.Orientation() == TopAbs_REVERSED:
                parameters = parameters[::-1]
            points.extend(curve.Value(float(value)).Coord() for value in parameters[:-1])
            kinds.append(
                0 if curve_type == GeomAbs_Line else 1 if curve_type == GeomAbs_Circle else 2
            )
            wire_explorer.Next()
        if len(points) < 3:
            return None
        coordinates = np.asarray(points)
        delta = coordinates - origin
        poly = np.stack([delta @ x_axis, delta @ y_axis], axis=1) * 255 / math.sqrt(area)
        loops.append(
            {
                "poly": poly,
                "types": kinds,
                "n_curves": len(kinds),
                "is_circle": len(kinds) == 1 and kinds[0] == 1,
            }
        )
        explorer.Next()
    normalized_depth = depth / math.sqrt(area)
    vector = prism_descriptor([{"loops": loops}], normalized_depth)
    if not np.isfinite(vector).all():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return int(row.uid), vector, digest


def _initialize(root: str, staging: str):
    global _ROOT, _STAGING
    _ROOT = Path(root)
    _STAGING = Path(staging)
    _STAGING.mkdir(parents=True, exist_ok=True)


def _prepare_record(row):
    result = _describe_step(row)
    if result is None:
        return None
    uid, vector, digest = result
    return {
        "uid": uid,
        "abc_uid": int(row.abc_uid),
        "vector": vector,
        "sha256": digest,
        "texts": {
            level: str(getattr(row, level)).strip()
            for level in DATASET_LEVELS["mmcad"]
            if hasattr(row, level)
            and str(getattr(row, level)).strip()
            and str(getattr(row, level)).strip() != "nan"
        },
    }


def prepare_mmcad(
    metadata_path: Path,
    step_root: Path,
    output: Path,
    workers: int,
    test_size: int,
    split_seed: int,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    if test_size < 1:
        raise ValueError("test-size must be positive")
    frame = pd.read_parquet(metadata_path)
    required = {
        "uid",
        "abc_uid",
        "split",
        "title",
        "description",
        "applications",
        "has_step",
        "record_available",
        "step_archive",
        "step_member",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"metadata columns missing: {sorted(missing)}")
    frame = frame[frame.has_step & frame.record_available].copy()
    staging = output.parent / "staging"
    records = {}
    for split in ("train", "validation"):
        rows = list(frame[frame.split == split].itertuples(index=False))
        described = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize,
            initargs=(str(step_root), str(staging)),
        ) as executor:
            results = executor.map(_prepare_record, rows, chunksize=4)
            for result in tqdm(results, total=len(rows), desc=f"mmcad-{split}"):
                if result is not None:
                    described.append(result)
        records[split] = described
    train_hashes = {record["sha256"] for record in records["train"]}
    train_vectors = {tuple(np.round(record["vector"], 5)) for record in records["train"]}
    before = len(records["validation"])
    records["validation"] = [
        record
        for record in records["validation"]
        if record["sha256"] not in train_hashes
        and tuple(np.round(record["vector"], 5)) not in train_vectors
    ]
    after_dedup = len(records["validation"])
    validation_uids = sorted(record["uid"] for record in records["validation"])
    if len(validation_uids) < test_size:
        raise ValueError("validation split is smaller than requested test size")
    test_uids = set(random.Random(split_seed).sample(validation_uids, test_size))
    test_by_uid = {record["uid"]: record for record in records["validation"]}
    records["test"] = [test_by_uid[uid] for uid in sorted(test_uids)]
    records["validation"] = [
        record for record in records["validation"] if record["uid"] not in test_uids
    ]
    cache_splits = {}
    for split, split_records in records.items():
        cache_splits[split] = {
            "uids": [record["uid"] for record in split_records],
            "vectors": np.stack([record["vector"] for record in split_records]).astype(np.float32),
            "texts": [record["texts"] for record in split_records],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    torch.save(
        {
            "dataset": "mmcad",
            "levels": list(DATASET_LEVELS["mmcad"]),
            "splits": cache_splits,
        },
        temporary,
    )
    os.replace(temporary, output)
    manifest = {
        "dataset": "mmcad",
        "descriptor_dim": 280,
        "official_training_split_preserved": True,
        "validation_duplicates_removed": before - after_dedup,
        "test_sampling_seed": split_seed,
        "counts": {
            "train": len(cache_splits["train"]["uids"]),
            "validation": len(cache_splits["validation"]["uids"]),
            "test": len(cache_splits["test"]["uids"]),
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
