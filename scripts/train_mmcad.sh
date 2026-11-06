PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-runs/mmcad/cache.pt}"
MODEL="${MODEL:-models/Qwen3-Embedding-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/mmcad/lora}"
NPROC="${NPROC:-7}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-96}"
EPOCHS="${EPOCHS:-3}"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC" \
  -m cadret train \
  --dataset mmcad \
  --cache "$CACHE" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --levels title,description,applications \
  --epochs "$EPOCHS" \
  --per-device-batch "$BATCH_PER_DEVICE" \
  --backbone-lr 1.5e-4 \
  --head-lr 5e-4 \
  --temperature 0.05 \
  --max-length 320 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --seed 20261025
