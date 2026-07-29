#!/bin/bash
# Environment configuration for SkillRise WebShop training.
# WebShop needs a JVM (Lucene/pyserini BM25) + gym 0.26 + a spaCy model. The
# recommended setup is a dedicated conda env with Java 11 and the WebShop deps.

# ── Clean up stale Ray/worker processes (prevents thread accumulation across reruns) ──
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
pkill -f "WebshopWorker" 2>/dev/null || true
pkill -f "verl.trainer.main_ppo" 2>/dev/null || true
sleep 2

# ── Java 11 (WebShop Lucene backend) ──
# Point JAVA_HOME at a local JDK 11 install (a conda env that bundles openjdk 11
# also works: set JAVA_HOME to that env prefix).
export JAVA_HOME="${JAVA_HOME:-/path/to/jdk-11}"
export PATH="${JAVA_HOME}/bin:${PATH}"

# ── JVM footprint per worker (each WebshopWorker starts a JVM for lucene) ──
export _JAVA_OPTIONS="-XX:+UseSerialGC -Xss512k -Xmx1g"

# ── Thread limits: keep numeric libs single-threaded; datasets non-concurrent ──
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export DATASETS_MAX_WORKERS=1
ulimit -u 65536 2>/dev/null || true

# ── WandB (export your own key before running, or set WANDB_MODE=offline) ──
export WANDB_API_KEY="${WANDB_API_KEY:-}"

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# ── WebShop runtime deps (installs only if missing) ──
if ! python3 -c "import gym, spacy; assert tuple(map(int, gym.__version__.split('.')[:2])) >= (0, 26); spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
    echo "[env.sh] Installing WebShop deps (gym 0.26.2 + spaCy model + helpers) ..."
    python3 -m pip install \
        gym==0.26.2 gymnasium termcolor thefuzz "clean-text[gpl]" rank-bm25 rich beautifulsoup4 flask spacy >/dev/null 2>&1 || true
    python3 -m spacy download en_core_web_sm 2>&1 | tail -1 || true
fi

# Some containers put an unreadable /root/bin on PATH; flashinfer/tvm_ffi tries to
# stat every PATH entry and dies with PermissionError. Drop it defensively.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/root/bin$' | tr '\n' ':' | sed 's/:$//')

# ── Policy model (path to a local HF checkpoint, e.g. Qwen3-4B) ──
export SKILLRISE_MODEL_PATH="${SKILLRISE_MODEL_PATH:-/path/to/Qwen3-4B}"
