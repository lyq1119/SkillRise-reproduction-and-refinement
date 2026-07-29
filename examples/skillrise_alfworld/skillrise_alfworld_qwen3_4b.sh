#!/usr/bin/env bash
set -x
# SkillRise cross-task meta-RL (K=3 tasks/group, N=8 trials/group). Full training config
# aligned with SkillPilot (8 GPUs/node). For a single-GPU smoke test see debug/.
ENGINE=${1:-vllm}
shift 2>/dev/null || true
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

pip3 install alfworld
pip3 install debugpy

# ── Model & naming ──
project_name="verl_agent_alfworld_meta"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
experiment_name="skillrise_qwen3-4b_K3_${TIMESTAMP}"
export OUTPUT_DIR="${SKILLRISE_OUTPUT_ROOT:-$HOME/experiments/skillrise}/${project_name}/${experiment_name}"
mkdir -p "$OUTPUT_DIR"

train_data_size=16        # number of GROUPS per step
val_data_size=128
group_size=8             # N = trials per group (env.rollout.n)
K=3                      # tasks per group (env.num_attempts)
mode="mean_norm"
num_cpus_per_env_worker=0.25
GROUP_FILE="${SCRIPT_DIR}/../../data/groups/skillrise_alfworld_K3.jsonl"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=skillrise \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$SKILLRISE_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.6 \
    +algorithm.curate_loss_weight=1.0 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    reward_model.reward_manager=episode \
    env.env_name="skillrise_alfworld/AlfredTWEnv" \
    env.seed=0 \
    env.alfworld.eval_dataset=eval_in_distribution \
    env.max_steps=30 \
    env.max_turns=10 \
    env.rollout.n=$group_size \
    env.num_attempts=$K \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.meta_mode=skillrise \
    +env.group_file="$GROUP_FILE" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.rollout_data_dir=$OUTPUT_DIR/rollouts \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.log_val_generations=10 \
    trainer.save_freq=10 \
    trainer.save_start_step=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.ray_wait_register_center_timeout=3600 \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/skillrise_alfworld_qwen3_4b.log"
