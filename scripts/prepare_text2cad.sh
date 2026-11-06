PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/Text2CAD}"
CACHE="${CACHE:-runs/text2cad/cache.pt}"
WORKERS="${WORKERS:-8}"

"$PYTHON_BIN" -m cadret prepare-text2cad \
  --data-root "$DATA_ROOT" \
  --output "$CACHE" \
  --workers "$WORKERS"
