PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-runs/text2cad/cache.pt}"
MODEL="${MODEL:-models/Qwen3-Embedding-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/text2cad/lora}"
NPROC="${NPROC:-2}"
BATCH_PER_DEVICE="${BATCH_PER_DEVICE:-512}"
EPOCHS="${EPOCHS:-3}"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC" \
  -m cadret train \
  --dataset text2cad \
  --cache "$CACHE" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --levels abstract,beginner,intermediate,expert \
  --epochs "$EPOCHS" \
  --per-device-batch "$BATCH_PER_DEVICE" \
  --backbone-lr 1e-4 \
  --head-lr 5e-4 \
  --temperature 0.05 \
  --max-length 320 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --seed 0
