#!/usr/bin/env bash
set -euo pipefail

export SKILLRISE_ROOT=/data/lanyuqi/skillrise
export HOME="$SKILLRISE_ROOT/runtime/home"
export VIRTUAL_ENV="$SKILLRISE_ROOT/runtime/venv"
export JAVA_HOME="$SKILLRISE_ROOT/runtime/java/jdk-11.0.32.1+1-jre"
export SKILLRISE_MODEL_PATH="$SKILLRISE_ROOT/runtime/models/Qwen3.5-9B"
export SCIWORLD_DATA="$SKILLRISE_ROOT/runtime/data/sciworld"
export SKILLRISE_OUTPUT_ROOT="$SKILLRISE_ROOT/runtime/outputs"
export WANDB_DIR="$SKILLRISE_ROOT/runtime/wandb"
export WANDB_CACHE_DIR="$SKILLRISE_ROOT/runtime/cache/wandb"
export WANDB_CONFIG_DIR="$SKILLRISE_ROOT/runtime/wandb/config"
export HF_HOME="$SKILLRISE_ROOT/runtime/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export UV_CACHE_DIR="$SKILLRISE_ROOT/runtime/cache/uv"
export TORCH_HOME="$SKILLRISE_ROOT/runtime/cache/torch"
export TRITON_CACHE_DIR="$SKILLRISE_ROOT/runtime/cache/triton"
export TMPDIR="$SKILLRISE_ROOT/runtime/cache/tmp"
export PATH="$VIRTUAL_ENV/bin:$JAVA_HOME/bin:$PATH"

mkdir -p \
  "$HOME" "$SKILLRISE_OUTPUT_ROOT" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$UV_CACHE_DIR" \
  "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"
