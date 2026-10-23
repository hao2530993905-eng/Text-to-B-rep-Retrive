import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cadret.checkpoints import load_checkpoint, restore_text_heads, save_checkpoint
from cadret.cli import COMMANDS, main
from cadret.config import DatasetPaths
from cadret.data.sequences import CADSeqStore
from cadret.data.text import gather_texts, load_data
from cadret.evaluation.metrics import pooled_recall
from cadret.geometry.descriptor import DESC_DIM, descriptor_vector
from cadret.geometry.spectral import describe
from cadret.models.towers import CADTower, TextTower
from cadret.training.objectives import multipos

ROOT = Path(__file__).resolve().parents[1]


def make_payload(width):
    rows = [[1, 0]]
    for x, y in ((10, 10), (10 + width, 10), (10 + width, 210), (10, 210)):
        rows.extend(((11 + x, 11 + y), (5, 0)))
    rows.extend(((4, 0), (3, 0), (2, 0), (7, 0)))
    rows.extend((11 + value, 0) for value in (64, 32, 0, 0, 0, 0, 0, 0, 0, 255))
    rows.append((6, 0))
    values = torch.tensor(rows, dtype=torch.int64)
    return {"vec": {"cad_vec": values, "index_vec": torch.zeros(len(values), dtype=torch.int64)}}


def make_dataset(root):
    import zipfile

    paths = DatasetPaths(root)
    paths.descriptions.parent.mkdir(parents=True)
    uids = ["0001/00010001", "0001/00010002", "0002/00020001"]
    paths.splits.write_text(json.dumps({"train": uids[:2], "test": uids[2:]}), encoding="utf-8")
    pd.DataFrame([{"uid": uid, "expert": f"part {i}"} for i, uid in enumerate(uids)]).to_csv(
        paths.descriptions, index=False
    )
    with zipfile.ZipFile(paths.sequence_zip, "w") as archive:
        for i, uid in enumerate(uids):
            buffer = io.BytesIO()
            torch.save(make_payload(200 - i * 35), buffer)
            archive.writestr(f"cad_seq/{uid}/seq/{uid.split('/')[-1]}.pth", buffer.getvalue())
    return paths, uids


class PipelineTests(unittest.TestCase):
    def test_prepare_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths, uids = make_dataset(base / "dataset")
            with CADSeqStore(None, paths.sequence_zip) as store:
                values, flags, indices = store.load_full(uids[0])
                vector = descriptor_vector(describe(values, flags, indices))
                self.assertEqual(vector.shape, (DESC_DIM,))
                self.assertTrue(np.isfinite(vector).all())
            output = base / "generated"
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "cadret",
                    "prepare",
                    "--data_dir",
                    str(paths.root),
                    "--output_dir",
                    str(output),
                    "--workers",
                    "1",
                ],
                cwd=base,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            cache_path = output / "spec_desc_cache.pt"
            data = load_data(["expert"], paths.root, cache_path)
            self.assertEqual(tuple(data["train"]["V"].shape), (2, DESC_DIM))
            self.assertEqual(len(data["test"]["rows"]), 1)
            mapping, texts = gather_texts(["expert"], paths.root)
            self.assertEqual((len(mapping), len(texts)), (3, 3))

    def test_loss_backpropagates_and_model_state_roundtrips(self):
        cad = CADTower(hidden=16, out=8, n_blocks=1, p=0)
        text = TextTower(dim=12, hidden=16, out=8, p=0)
        x, y = torch.randn(4, DESC_DIM), torch.randn(4, 12)
        zc, (zq, _, _) = cad(x), text(y)
        classes = torch.tensor([0, 0, 1, 2])
        positives = (classes[:, None] == classes[None, :]).float()
        positives /= positives.sum(1, keepdim=True)
        loss = multipos(zq @ zc.T, positives)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(cad.inp[0].weight.grad).all())
        cad.eval()
        restored = CADTower(hidden=16, out=8, n_blocks=1, p=0).eval()
        restored.load_state_dict(copy.deepcopy(cad.state_dict()))
        torch.testing.assert_close(cad(x), restored(x), rtol=0, atol=0)

    def test_metrics_and_invalid_pool(self):
        embeddings = torch.eye(12)
        rows = [(i, "expert", "") for i in range(12)]
        result = pooled_recall(embeddings, embeddings, rows, ["expert"], 10, 20)
        self.assertEqual(result["expert"], {"R@1": 1, "R@5": 1, "R@10": 1, "MedR": 1})
        with self.assertRaises(ValueError):
            pooled_recall(embeddings, embeddings, rows, ["expert"], 13, 20)

    def test_command_help(self):
        with contextlib.redirect_stdout(io.StringIO()):
            main([])
            for command in COMMANDS:
                with self.assertRaises(SystemExit) as raised:
                    main([command, "--help"])
                self.assertEqual(raised.exception.code, 0)

    def test_retained_weights(self):
        checkpoint = load_checkpoint(ROOT / "model" / "lora_cad.pt")
        model = CADTower().eval()
        model.load_state_dict(checkpoint["cadt"])
        with torch.no_grad():
            output = model(torch.zeros(2, checkpoint["mu"].numel()))
        self.assertEqual(tuple(output.shape), (2, 512))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaises(ValueError):
            load_checkpoint(ROOT / "model" / "lora_cad.pt", require_text=True)


class AdapterStub(torch.nn.Module):
    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


class TextStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AdapterStub()
        self.trunk = torch.nn.Linear(3, 4)
        self.proj = torch.nn.Linear(4, 2)
        self.cls = torch.nn.ModuleDict({"attribute": torch.nn.Linear(4, 2)})
        self.reg = torch.nn.Linear(4, 1)

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


class CheckpointTests(unittest.TestCase):
    def test_complete_checkpoint_preserves_all_heads(self):
        cad = CADTower(hidden=16, out=2, n_blocks=1)
        text = TextStub()
        with tempfile.TemporaryDirectory() as temporary:
            save_checkpoint(
                cad, text, text, torch.zeros(DESC_DIM), torch.ones(DESC_DIM), {}, temporary
            )
            checkpoint = load_checkpoint(Path(temporary) / "lora_cad.pt", require_text=True)
            restored = TextStub()
            restore_text_heads(restored, checkpoint)
            for name in ("trunk", "proj", "cls", "reg"):
                for key, value in getattr(text, name).state_dict().items():
                    torch.testing.assert_close(
                        value, getattr(restored, name).state_dict()[key], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
