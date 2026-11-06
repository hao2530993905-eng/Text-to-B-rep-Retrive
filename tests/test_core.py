from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cadret.config import DATASET_LEVELS
from cadret.data import load_split
from cadret.geometry import describe_sequence, descriptor_vector, prism_descriptor
from cadret.runtime import pooled_recall, symmetric_info_nce
from cadret.text2cad import prepare_text2cad

ROOT = Path(__file__).resolve().parents[1]


def payload(width):
    rows = [[1, 0]]
    for x, y in ((10, 10), (10 + width, 10), (10 + width, 210), (10, 210)):
        rows.extend(((11 + x, 11 + y), (5, 0)))
    rows.extend(((4, 0), (3, 0), (2, 0), (7, 0)))
    rows.extend((11 + value, 0) for value in (64, 32, 0, 0, 0, 0, 0, 0, 0, 255))
    rows.append((6, 0))
    return torch.tensor(rows, dtype=torch.int64)


class CoreTests(unittest.TestCase):
    def test_text2cad_prepare(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            annotations = root / "text2cad_v1.0"
            annotations.mkdir(parents=True)
            uids = ["0001/00010001", "0001/00010002"]
            (annotations / "train_test_val.json").write_text(
                json.dumps({"train": uids[:1], "test": uids[1:]}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "uid": uid,
                        "abstract": "box",
                        "beginner": "a box",
                        "intermediate": "a rectangular box",
                        "expert": "a rectangular prism",
                    }
                    for uid in uids
                ]
            ).to_csv(annotations / "text2cad_v1.0.csv", index=False)
            with zipfile.ZipFile(root / "cad_seq.zip", "w") as archive:
                for uid in uids:
                    buffer = io.BytesIO()
                    torch.save(
                        {
                            "cad_vec": payload(180),
                            "index_vec": torch.zeros(len(payload(180)), dtype=torch.int64),
                        },
                        buffer,
                    )
                    archive.writestr(
                        f"cad_seq/{uid}/seq/{uid.rsplit('/', 1)[-1]}.pth",
                        buffer.getvalue(),
                    )
            cache = Path(temporary) / "cache.pt"
            prepare_text2cad(root, cache, workers=1)
            data = load_split(cache, "train", DATASET_LEVELS["text2cad"])
            self.assertEqual(len(data["uids"]), 1)
            self.assertEqual(len(data["rows"]), 4)

    def test_sequence_descriptor(self):
        vector = descriptor_vector(describe_sequence(payload(180)))
        self.assertEqual(vector.shape, (280,))
        self.assertTrue(np.isfinite(vector).all())

    def test_step_descriptor(self):
        poly = np.array([[0.0, 0.0], [120.0, 0.0], [120.0, 80.0], [0.0, 80.0]])
        vector = prism_descriptor(
            [
                {
                    "loops": [
                        {
                            "poly": poly,
                            "types": [0, 0, 0, 0],
                            "n_curves": 4,
                            "is_circle": False,
                        }
                    ]
                }
            ],
            1.5,
        )
        self.assertEqual(vector.shape, (280,))
        self.assertTrue(np.isfinite(vector).all())

    def test_single_pair_loss(self):
        text = torch.nn.Parameter(torch.randn(8, 16))
        cad = torch.nn.Parameter(torch.randn(8, 16))
        text = torch.nn.functional.normalize(text, dim=-1)
        cad = torch.nn.functional.normalize(cad, dim=-1)
        loss = symmetric_info_nce(text, cad, 0.05)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_metrics(self):
        embeddings = torch.eye(12)
        rows = [(index, "expert", "") for index in range(12)]
        result, digest = pooled_recall(
            embeddings,
            embeddings,
            rows,
            ("expert",),
            10,
            20,
        )
        self.assertEqual(result["expert"]["R@1"], 1.0)
        self.assertEqual(len(digest), 64)

    def test_cache_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.pt"
            torch.save(
                {
                    "dataset": "text2cad",
                    "levels": list(DATASET_LEVELS["text2cad"]),
                    "splits": {
                        "train": {
                            "uids": ["a"],
                            "vectors": np.zeros((1, 280), dtype=np.float32),
                            "texts": [{"expert": "box"}],
                        }
                    },
                },
                path,
            )
            data = load_split(path, "train", ("expert",))
            self.assertEqual(data["rows"], [(0, "expert", "box")])

    def test_cli(self):
        environment = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "cadret"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("prepare-text2cad", completed.stdout)
        self.assertIn("prepare-mmcad", completed.stdout)


if __name__ == "__main__":
    unittest.main()
