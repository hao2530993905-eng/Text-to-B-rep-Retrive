PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-runs/text2cad/cache.pt}"
CHECKPOINT="${CHECKPOINT:-runs/text2cad/4b_frozen/retrieval.pt}"
TEXT_FEATURES="${TEXT_FEATURES:-runs/text2cad/text_features_4b.pt}"
NPROC="${NPROC:-8}"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC" \
  -m cadret evaluate \
  --dataset text2cad \
  --cache "$CACHE" \
  --checkpoint "$CHECKPOINT" \
  --text-features "$TEXT_FEATURES" \
  --split test \
  --levels abstract,beginner,intermediate,expert \
  --batch-size 64 \
  --pool-size 1000 \
  --trials 1000
