#!/bin/bash
# Environment configuration for SkillRise ALFWorld training.

# ── WandB (export your own key before running, or set WANDB_MODE=offline) ──
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ── ALFWorld data (TextWorld game files, json_2.1.1/{train,valid_seen,...}) ──
# Download via `alfworld-download` (from the alfworld package) and point this at it.
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/data/alfworld}"

# ── Runtime tuning ──
export RAY_worker_register_timeout_seconds=600
export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export RAY_IGNORE_UNHANDLED_ERRORS=1

# ── Policy model (path to a local HF checkpoint, e.g. Qwen3-4B) ──
export SKILLRISE_MODEL_PATH="${SKILLRISE_MODEL_PATH:-/path/to/Qwen3-4B}"
