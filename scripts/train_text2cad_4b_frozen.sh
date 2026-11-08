PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-runs/text2cad/cache.pt}"
MODEL="${MODEL:-models/Qwen3-Embedding-4B}"
TEXT_FEATURES="${TEXT_FEATURES:-runs/text2cad/text_features_4b.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/text2cad/4b_frozen}"
EPOCHS="${EPOCHS:-30}"
NPROC="${NPROC:-8}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-64}"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC" \
  -m cadret train \
  --dataset text2cad \
  --tuning frozen \
  --precision fp32 \
  --cache "$CACHE" \
  --model "$MODEL" \
  --text-features "$TEXT_FEATURES" \
  --output-dir "$OUTPUT_DIR" \
  --levels abstract,beginner,intermediate,expert \
  --epochs "$EPOCHS" \
  --per-device-batch "$BATCH_PER_DEVICE" \
  --head-lr 3e-4 \
  --temperature 0.05 \
  --warmup-fraction 0.1 \
  --max-length 384 \
  --seed 0
