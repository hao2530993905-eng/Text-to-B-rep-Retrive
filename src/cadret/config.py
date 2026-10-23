import os
from dataclasses import dataclass
from pathlib import Path

DATA = os.environ.get("CADRET_DATA_DIR", "data/Text2CAD")
LEVELS = ("abstract", "beginner", "intermediate", "expert")
INSTRUCT = (
    "Instruct: Given a text description of a CAD part, retrieve the matching CAD model.\nQuery: "
)
AUX_CLS = (("n_parts", 5), ("b1", 6), ("sym", 8), ("n_circles", 5), ("n_faces", 5))
AUX_REG = ("slender", "iso", "fill", "aspect", "depth_spread")


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    @property
    def descriptions(self):
        return self.root / "text2cad_v1.0" / "text2cad_v1.0.csv"

    @property
    def splits(self):
        return self.root / "text2cad_v1.0" / "train_test_val.json"

    @property
    def sequences(self):
        return self.root / "extracted" / "cad_seq"

    @property
    def sequence_zip(self):
        return self.root / "cad_seq.zip"
