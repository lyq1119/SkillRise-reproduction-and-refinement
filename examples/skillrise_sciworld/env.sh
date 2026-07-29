#!/bin/bash
# Environment configuration for SkillRise ScienceWorld training.

# ── WandB (export your own key before running, or set WANDB_MODE=offline) ──
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ── ScienceWorld data (variations_idx lives here) ──
export SCIWORLD_DATA="${SCIWORLD_DATA:-$HOME/data/sciworld}"

# ── Java 11 (ScienceWorld JVM backend; system default Java 8 will not work) ──
# Point JAVA_HOME at a local JDK 11 install.
export JAVA_HOME="${JAVA_HOME:-/path/to/jdk-11}"
export PATH="${JAVA_HOME}/bin:${PATH}"
# Limit per-SciWorld-worker JVM threads (serial GC, single compiler thread, 1
# processor, small stacks) to avoid pthread_create exhaustion when many JVM
# workers run alongside vLLM. -XX:CICompilerCount=1 is illegal under the default
# tiered compilation, so -XX:-TieredCompilation makes it legal.
export JAVA_TOOL_OPTIONS="-XX:+UseSerialGC -XX:-TieredCompilation -XX:CICompilerCount=1 -XX:ActiveProcessorCount=1 -Xss256k"

# ── Process/thread limits ──
# Raise process/fd caps so JVM workers + vLLM torch.compile don't hit
# "pthread_create failed: Resource temporarily unavailable" -> SIGABRT.
ulimit -u unlimited 2>/dev/null || ulimit -u 65535 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

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
export TOKENIZERS_PARALLELISM=false
# Cap Triton/Inductor parallel compile workers so vLLM's torch.compile doesn't
# spawn a thread storm on top of the JVM workers (the SIGABRT trigger).
export TORCHINDUCTOR_COMPILE_THREADS=1

export RAY_IGNORE_UNHANDLED_ERRORS=1

# ── Policy model (path to a local HF checkpoint, e.g. Qwen3-4B) ──
export SKILLRISE_MODEL_PATH="${SKILLRISE_MODEL_PATH:-/path/to/Qwen3-4B}"
