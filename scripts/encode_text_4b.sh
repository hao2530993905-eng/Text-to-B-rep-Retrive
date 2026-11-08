PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-runs/text2cad/cache.pt}"
MODEL="${MODEL:-models/Qwen3-Embedding-4B}"
TEXT_FEATURES="${TEXT_FEATURES:-runs/text2cad/text_features_4b.pt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-384}"
NPROC="${NPROC:-8}"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC" \
  -m cadret encode-text \
  --dataset text2cad \
  --cache "$CACHE" \
  --model "$MODEL" \
  --output "$TEXT_FEATURES" \
  --levels abstract,beginner,intermediate,expert \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH"
