# run on 4xH100
# make sure your current working directory is the root of the project

set -x
export HYDRA_FULL_ERROR=1
# export RAY_DEBUG_POST_MORTEM=1
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/config"
# completion_callback=none
env_url=http://$MASTER_ADDR:8000
current_time=$(date "+%Y%m%d_%H%M%S")
log_file="logs/assignment/qwen3-32b_semadv_lossmask_turbo_sparse_n16_ppobs32_${current_time}.log"
experiment_name="qwen3-32b_semadv_lossmask_turbo_sparse_n16_ppobs32"
# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/config/runtime_env.yaml"}
NNODES=${NNODES:-4}

swanlab login --api-key xSxgnzpo2HEXkIzoxD2Ua

ray job submit --address="$RAY_ADDRESS" \
    --runtime-env="${RUNTIME_ENV}" \
    --working-dir "." \
    -- python -m beyondagent.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='beyond_agent_dataflow' \
    env_service.env_url=$env_url \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.advantage.good_scale=1.0 \
    actor_rollout_ref.actor.advantage.bad_scale=0.2 \
    actor_rollout_ref.actor.advantage.neg_bad_scale=-0.2 \
    +semantic_eval_model='qwen-turbo' \
    +env_sparse=True \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=20480 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.use_qwen3=True \
    actor_rollout_ref.rollout.enable_request_id=False \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.max_model_len=20480 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.model.path=/mnt/data/yunpeng.zyp/models/Qwen3-32B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=8 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='beyondagent' \
    trainer.experiment_name=$experiment_name \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="${WORKING_DIR}/experiments/exp_${current_time}/validation_log" \
    trainer.rollout_data_dir="${WORKING_DIR}/experiments/exp_${current_time}/rollout_log" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    critic.ppo_max_token_len_per_gpu=20480 \
    critic.forward_max_token_len_per_gpu=20480 \
    data.train_files=/mnt/data/yunpeng.zyp/data/appworld_verl/train.parquet \
    data.val_files=/mnt/data/yunpeng.zyp/data/appworld_verl/dev.parquet \
    experience_maker.enable_summarizer=False \
    experience_maker.enable_context_generator=False \
    experience_maker.workspace_id="w1_qwen3_${current_time}" \
    2>&1 | tee "$log_file" \
    $@