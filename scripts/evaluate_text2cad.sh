PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-runs/text2cad/cache.pt}"
MODEL="${MODEL:-models/Qwen3-Embedding-0.6B}"
CHECKPOINT="${CHECKPOINT:-runs/text2cad/lora/retrieval.pt}"
NPROC="${NPROC:-1}"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC" \
  -m cadret evaluate \
  --dataset text2cad \
  --cache "$CACHE" \
  --model "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --levels abstract,beginner,intermediate,expert \
  --batch-size 64 \
  --pool-size 1000 \
  --trials 1000
