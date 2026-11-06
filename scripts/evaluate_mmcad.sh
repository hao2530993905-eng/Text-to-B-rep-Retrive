PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="${CACHE:-runs/mmcad/cache.pt}"
MODEL="${MODEL:-models/Qwen3-Embedding-0.6B}"
CHECKPOINT="${CHECKPOINT:-runs/mmcad/lora/retrieval.pt}"
NPROC="${NPROC:-1}"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC" \
  -m cadret evaluate \
  --dataset mmcad \
  --cache "$CACHE" \
  --model "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --levels title,description,applications \
  --batch-size 128 \
  --pool-size 1000 \
  --trials 1000
