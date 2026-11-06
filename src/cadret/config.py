from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DESCRIPTOR_DIM = 280
INSTRUCTION = (
    "Instruct: Given a text description of a CAD part, retrieve the matching CAD model.\nQuery: "
)
DATASET_LEVELS = {
    "text2cad": ("abstract", "beginner", "intermediate", "expert"),
    "mmcad": ("title", "description", "applications"),
}


@dataclass(frozen=True)
class Text2CADPaths:
    root: Path

    @property
    def descriptions(self) -> Path:
        return self.root / "text2cad_v1.0" / "text2cad_v1.0.csv"

    @property
    def splits(self) -> Path:
        return self.root / "text2cad_v1.0" / "train_test_val.json"

    @property
    def sequences(self) -> Path:
        return self.root / "extracted" / "cad_seq"

    @property
    def archive(self) -> Path:
        return self.root / "cad_seq.zip"
