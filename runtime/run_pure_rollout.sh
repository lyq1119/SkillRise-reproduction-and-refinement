#!/usr/bin/env bash
set -euo pipefail

SKILLRISE_ROOT=/data/lanyuqi/skillrise
source "$SKILLRISE_ROOT/runtime/run_env.sh"
export PYTHONHASHSEED=0
export PYTHONPATH="$SKILLRISE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR=/tmp/skillrise_ray

RUN_ID="qwen3.5-9b_sciworld_group0_seed0_$(date +%Y%m%dT%H%M%S%z)"
OUTPUT_DIR="${SKILLRISE_OUTPUT_ROOT}/pure_rollout/${RUN_ID}"
mkdir -p "$OUTPUT_DIR"
cd "$SKILLRISE_ROOT"

python runtime/pure_rollout.py \
  --model "$SKILLRISE_MODEL_PATH" \
  --group-file data/groups/skillrise_sciworld_K3.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --gpu-memory-utilization 0.94 \
  "$@" \
  2>&1 | tee "$OUTPUT_DIR/run.log"

echo "$OUTPUT_DIR"
