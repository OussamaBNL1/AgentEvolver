# run on 4xH100 with local vLLM semantic evaluation
# make sure your current working directory is the root of the project
# adv_assignment master
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
suffix="qwen3_14b_sparse_adv_groupnorm_std_trbs16_ppobs16"
log_file="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/logs/qwen3/${suffix}_${current_time}.log"

# Ray - 修改为直接连接Ray集群，而不是通过Job API
RAY_ADDRESS=${RAY_ADDRESS:-"ray://localhost:10001"}  # 改为Ray客户端地址
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/config/runtime_env.yaml"}
NNODES=${NNODES:-4}

swanlab login --api-key xSxgnzpo2HEXkIzoxD2Ua

# 设置运行环境变量（原来在runtime_env.yaml中的变量）
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_DISABLE="1"

# 直接执行训练命令，而不是使用 ray job submit
python -m beyondagent.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='beyond_agent_dataflow' \
    env_service.env_url=$env_url \
    algorithm.adv_estimator=grpo \
    semantic_advantage.enable=true \
    semantic_advantage.evaluation_type='api' \
    semantic_advantage.mask_type='response_mask' \
    semantic_advantage.mode='semantic' \
    semantic_advantage.consistent_scale=1.0 \
    semantic_advantage.pos_unconsistent_scale=0.2 \
    semantic_advantage.neg_unconsistent_scale=-0.2 \
    semantic_advantage.api_max_retries=200 \
    semantic_advantage.concurrent=5 \
    semantic_advantage.model='qwen-turbo' \
    semantic_advantage.adv_norm.enable=true \
    semantic_advantage.adv_norm.level="group" \
    env_sparse=true \
    data.train_batch_size=16 \
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
    actor_rollout_ref.model.path=/mnt/data_aisys_cpfs/xielipeng.xlp/models/Qwen3-14B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=8 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','swanlab']" \
    trainer.project_name='beyondagent' \
    trainer.experiment_name="${suffix}" \
    trainer.nnodes=2 \
    trainer.default_local_dir="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/checkpoints/qwen3/${suffix}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=60 \
    trainer.val_before_train=True \
    trainer.validation_data_dir="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/experiments/exp_${suffix}_${current_time}/validation_log" \
    trainer.rollout_data_dir="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/experiments/exp_${suffix}_${current_time}/rollout_log" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    critic.ppo_max_token_len_per_gpu=20480 \
    critic.forward_max_token_len_per_gpu=20480 \
    data.train_files=/mnt/data_aisys_cpfs/zouanni.zan/data/appworld_parquet/train.parquet \
    data.val_files=/mnt/data_aisys_cpfs/zouanni.zan/data/appworld_parquet/dev.parquet \
    experience_maker.enable_summarizer=False \
    experience_maker.enable_context_generator=False \
    experience_maker.workspace_id="w1_qwen3_api_turbo_${current_time}" \
    2>&1 | tee "$log_file"