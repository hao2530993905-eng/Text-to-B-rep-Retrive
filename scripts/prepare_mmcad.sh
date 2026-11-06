PYTHON_BIN="${PYTHON_BIN:-python}"
METADATA="${METADATA:-data/MMCAD-B/metadata.parquet}"
STEP_ROOT="${STEP_ROOT:-data/MMCAD-B/steps}"
CACHE="${CACHE:-runs/mmcad/cache.pt}"
WORKERS="${WORKERS:-8}"
TEST_SIZE="${TEST_SIZE:-1000}"
SPLIT_SEED="${SPLIT_SEED:-20261025}"

"$PYTHON_BIN" -m cadret prepare-mmcad \
  --metadata "$METADATA" \
  --step-root "$STEP_ROOT" \
  --output "$CACHE" \
  --workers "$WORKERS" \
  --test-size "$TEST_SIZE" \
  --split-seed "$SPLIT_SEED"
