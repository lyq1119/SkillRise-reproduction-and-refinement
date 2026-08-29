#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/lanyuqi/skillrise
source "$ROOT/runtime/run_env.sh"
cd "$ROOT"

# Keep the official entry point and 8-GPU topology. These are smoke-only Hydra
# tail overrides; the production launcher below supplies none of them.
exec bash examples/skillrise_sciworld/skillrise_sciworld_qwen3_4b.sh vllm \
  trainer.total_epochs=1 \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  trainer.save_freq=1 \
  trainer.save_start_step=0 \
  trainer.test_freq=1 \
  trainer.log_val_generations=1
