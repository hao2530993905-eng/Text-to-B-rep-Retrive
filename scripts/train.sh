#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${TORCHRUN:-torchrun}" --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" -m cadret train "$@"
