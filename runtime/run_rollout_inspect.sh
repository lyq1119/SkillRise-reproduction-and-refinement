#!/usr/bin/env bash
# Two-GPU trajectory-inspection run.
#
# Goal: produce the FIRST real Qwen3.5-9B ScienceWorld trajectories through the
# OFFICIAL verl pipeline (solve -> curate -> cross-task advance) without burning
# the full 8-GPU budget. Everything algorithmic/data/prompt stays official; the
# overrides below only scale batch/memory to fit two idle RTX 4090s.
#
# Engine = HF transformers generation (the vLLM nightly is incompatible with the
# vendored veRL interface; HF path is the repo's own fallback rollout worker).
set -euo pipefail

ROOT=/data/lanyuqi/skillrise
source "$ROOT/runtime/run_env.sh"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_TMPDIR=/tmp/skillrise-ray
# scienceworld + gym are already installed in the venv; keep the official
# script's `pip3 install` local-only so it can't hang on flaky network.
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
# Dedicated NCCL port range + explicit socket ifname so our collectives never
# collide with the other NCCL jobs (sglang/SHANNON) running on this host.
export NCCL_SOCKET_IFNAME=ens11f1
export NCCL_PORT_RANGE=29200-29300
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
# Fragmentation-tolerant CUDA allocator: FSDP's sync_module_states broadcast
# needs ~full-model + chunk transiently, right at the 24 GiB limit.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec bash examples/skillrise_sciworld/skillrise_sciworld_qwen3_4b.sh hf \
  trainer.n_gpus_per_node=4 \
  data.train_batch_size=4 \
  data.val_batch_size=1 \
  env.rollout.n=1 \
  data.max_response_length=512 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  trainer.logger=['console'] \
  trainer.save_freq=-1 \
  trainer.save_start_step=1000 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.log_val_generations=2 \
  env.max_env_per_rollout=1
