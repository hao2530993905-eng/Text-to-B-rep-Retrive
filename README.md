# Text-to-B-rep Retrieval
Text-to-CAD retrieval on Text2CAD with spectral CAD descriptors and Qwen3-Embedding.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

### What it is
This repository contains two Text2CAD experiments:

1. Qwen3-Embedding-4B with a frozen text encoder, FP32 projection training, 30 epochs, and global batch 512.
2. Qwen3-Embedding-0.6B with LoRA, 30 epochs, global batch 512, and FP32 training.

Both experiments use the same 280-D geometric descriptor, residual CAD tower, symmetric one-to-one contrastive loss, and pooled retrieval evaluation. No additional prediction heads or grouping logic are present.

Models, datasets, descriptor caches, text-feature caches, adapters, checkpoints, and metrics remain outside Git.

---

### Quick start
Install a CUDA-enabled PyTorch build for the host, then install the package.

```bash
python -m pip install -e .
```

Download [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) to the relative `models/Qwen3-Embedding-0.6B` directory.

Download [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) to the relative `models/Qwen3-Embedding-4B` directory.

Download [Text2CAD](https://huggingface.co/datasets/SadilKhan/Text2CAD) and the project resources from [Text2CAD](https://github.com/SadilKhan/Text2CAD). Use this layout:

```text
data/Text2CAD/
  text2cad_v1.0/
    text2cad_v1.0.csv
    train_test_val.json
  cad_seq.zip
```

Build the descriptor and text-row cache:

```bash
bash scripts/prepare_text2cad.sh
```

---

### 4B frozen experiment
The text features are encoded once across eight GPUs with the 4B model in bf16, stored as FP32, and reused during training. The text projection and CAD tower are trained in FP32 without autocast.

Create the frozen text-feature cache:

```bash
bash scripts/encode_text_4b.sh
```

Train:

```bash
bash scripts/train_text2cad_4b_frozen.sh
```

Evaluate:

```bash
bash scripts/evaluate_text2cad_4b_frozen.sh
```

The frozen preset uses:

- eight GPUs
- 64 samples per device
- 512 samples per global step
- 1,215 steps per epoch
- 30 epochs
- 36,450 optimizer steps
- head and CAD learning rate `3e-4`
- AdamW with weight decay `0.01`
- OneCycleLR warmup fraction `0.1`
- temperature `0.05`
- gradient clipping `1.0`
- maximum text length `384`
- seed `0`

The expected output is stored under `runs/text2cad/4b_frozen`.

---

### 0.6B LoRA experiment
The LoRA preset follows the 0.6B frozen run for epochs, batch size, schedule, precision, seed, and text length.

Train:

```bash
bash scripts/train_text2cad_06b_lora.sh
```

Evaluate:

```bash
bash scripts/evaluate_text2cad_06b_lora.sh
```

The LoRA preset uses:

- Qwen3-Embedding-0.6B
- eight GPUs
- 64 samples per device
- 512 samples per global step
- 1,215 steps per epoch
- 30 epochs
- 36,450 optimizer steps
- backbone learning rate `1e-4`
- head and CAD learning rate `3e-4`
- LoRA rank `16`
- LoRA alpha `32`
- LoRA dropout `0.05`
- AdamW with weight decay `0.01`
- OneCycleLR warmup fraction `0.1`
- temperature `0.05`
- gradient clipping `1.0`
- maximum text length `384`
- FP32 without autocast
- seed `0`

The expected output is stored under `runs/text2cad/06b_lora`.

---

### Validation
Both evaluation scripts use the official Text2CAD test split, 1,000 candidate pools, 1,000 trials per text level, and report R@1, R@5, R@10, and MedR.

Results are written to `test_metrics.json` beside the checkpoint.

The 4B evaluation reads the saved text-feature cache. The 0.6B evaluation reads the saved LoRA adapter and base model.

---

### Background runs and logs
Create the log directory:

```bash
mkdir -p logs
```

Start 4B feature encoding:

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/encode_text_4b.sh > logs/encode_4b.log 2>&1 &
```

Start 4B frozen training after encoding finishes:

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_text2cad_4b_frozen.sh > logs/train_4b_frozen.log 2>&1 &
```

Start 0.6B LoRA training:

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train_text2cad_06b_lora.sh > logs/train_06b_lora.log 2>&1 &
```

Follow a log:

```bash
tail -f logs/train_4b_frozen.log
```

```bash
tail -f logs/train_06b_lora.log
```

Run evaluation after training completes:

```bash
bash scripts/evaluate_text2cad_4b_frozen.sh
```

```bash
bash scripts/evaluate_text2cad_06b_lora.sh
```

---

### Output files

| File | Purpose |
|---|---|
| `config.json` | Resolved training arguments and data counts |
| `training_curve.csv` | Per-epoch loss and learning rate |
| `adapter` | LoRA weights and tokenizer for the 0.6B run |
| `retrieval.pt` | CAD tower, text projection, normalization statistics, and configuration |
| `test_metrics.json` | Retrieval metrics |

---

### Optional MMCAD-B data preparation
The data pipeline also supports MMCAD-B metadata and STEP files. Download the [MMCAD dataset](https://huggingface.co/datasets/exanos/MMCAD) and use this layout:

```text
data/MMCAD-B/
  metadata.parquet
  steps/
    step-00000/
    step-00001/
```

Build the MMCAD-B cache:

```bash
bash scripts/prepare_mmcad.sh
```

---

### Tests
The tests use synthetic CAD data and do not need downloaded datasets or model weights.

```bash
python -m unittest discover -s tests -v
```

---

### Documentation map
| Document | What it covers |
|---|---|
| [README.md](README.md) | Data sources, setup, the two training experiments, evaluation, and outputs |
| [pyproject.toml](pyproject.toml) | Package metadata and dependencies |
| [scripts/prepare_text2cad.sh](scripts/prepare_text2cad.sh) | Text2CAD descriptor and text cache |
| [scripts/encode_text_4b.sh](scripts/encode_text_4b.sh) | Frozen 4B text-feature cache |
| [scripts/train_text2cad_4b_frozen.sh](scripts/train_text2cad_4b_frozen.sh) | 4B frozen training |
| [scripts/train_text2cad_06b_lora.sh](scripts/train_text2cad_06b_lora.sh) | 0.6B LoRA training |
| [scripts/evaluate_text2cad_4b_frozen.sh](scripts/evaluate_text2cad_4b_frozen.sh) | 4B frozen evaluation |
| [scripts/evaluate_text2cad_06b_lora.sh](scripts/evaluate_text2cad_06b_lora.sh) | 0.6B LoRA evaluation |
| [scripts/prepare_mmcad.sh](scripts/prepare_mmcad.sh) | Optional MMCAD-B STEP cache |
