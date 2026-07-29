set -x
# DEBUG: WebShop GRPO (single attempt, no reflect), single GPU, 2 steps.
ENGINE=${1:-vllm}
shift 2>/dev/null || true
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

train_data_size=4
val_data_size=4
group_size=8
mode="mean_norm"
num_cpus_per_env_worker=0.25
OUTPUT_DIR="${SCRIPT_DIR}/out_grpo"
mkdir -p "$OUTPUT_DIR"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' --train_data_size $train_data_size --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    +actor_rollout_ref.model.enable_thinking=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.6 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    reward_model.reward_manager=episode \
    env.env_name=Webshop \
    env.seed=0 \
    env.webshop.use_small=True \
    env.webshop.human_goals=False \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.rollout.n=$group_size \
    env.num_attempts=1 \
    env.max_steps=40 \
    env.max_turns=12 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='webshop_debug' \
    trainer.experiment_name='debug_grpo' \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.rollout_data_dir="$OUTPUT_DIR/rollouts" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=2 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/debug_grpo.log"
