#!/bin/bash
# Define color codes for better log visibility
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Define log function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${BLUE}[INFO]${NC} $1"
}
export current_time=$(date "+%Y%m%d_%H%M%S")
# Define environment variables
# export ENV_PATH="/mnt/data/taoshuchang.tsc/beyondagent/EnvService"
export ENV_PATH="/mnt/data/taoshuchang.tsc/beyondagent/EnvService_copy"
export PROJECT_PATH="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent"
export TRAIN_SCRIPT="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/examples/qwen3/run_tsc_qwen3_14b_baseline_trbs16_ppobs16.sh"

export FILE_PATH="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/logs/assignment/$MASTER_ADDR-master-ip.log"
export EXPECTED_WORKERS=$WORLD_SIZE

log "Starting script execution"
log "TRAIN_SCRIPT set to: $TRAIN_SCRIPT"
log "PROJECT_PATH set to: $PROJECT_PATH"

# Activate conda environment
source /mnt/data/taoshuchang.tsc/anaconda3/etc/profile.d/conda.sh
conda activate verl
which python

# 检查worker节点数量
check_workers() {
    worker_count=$(ray status 2>/dev/null | grep "node_" | wc -l)
    if [ -z "$worker_count" ]; then
        echo 0 # worker_count=0
    fi
    echo $worker_count # $((worker_count-1)) 
}

# 颜色输出函数
print_green() {
    echo -e "\033[32m$1\033[0m"
}

print_red() {
    echo -e "\033[31m$1\033[0m"
}

print_green "=== Debug Information ==="
print_green "Hostname: $MASTER_ADDR"
print_green "All environment variables:"
env | sort
print_green "========================="

# rm -f "$FILE_PATH"

# 判断是否是master节点
if [[ $HOSTNAME == *"-master-"* ]]; then
    print_green "This is master node: $HOSTNAME"

    # 停止可能存在的Ray进程
    ray stop || true

    # 启动master节点
    print_green "Starting Ray head node at $MASTER_ADDR"
    ray start --head \
        --node-ip-address $MASTER_ADDR\
        --num-gpus 8 \

    # 将master IP写入共享目录
    echo $MASTER_ADDR > $FILE_PATH

    # 等待所有worker节点加入
    print_green "Waiting for all worker nodes to join..."
    TIMEOUT=600  # 10分钟超时
    INTERVAL=10  # 每10秒检查一次
    ELAPSED=0

    while true; do
        current_workers=$(check_workers)
        print_green "Current worker count: $current_workers/$EXPECTED_WORKERS"

        if [ "$current_workers" -eq "$EXPECTED_WORKERS" ]; then
            print_green "All workers have joined the cluster!"
            break
        fi

        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            print_red "Timeout waiting for workers. Only $current_workers/$EXPECTED_WORKERS workers joined."
            print_red "Please check the worker nodes and try again."
            exit 1
        fi

        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
    done

    # 检查集群状态
    ray status


    # 等待Ray dashboard完全启动
    print_green "Waiting for Ray dashboard to be ready..."
    while ! curl -s http://127.0.0.1:8265 > /dev/null; do
        sleep 5
    done

    # 执行训练脚本
    export RAY_CLUSTER_MODE=multi_node
    conda activate base
    cd $ENV_PATH
    which python
    # python -m env.env_service &
    nohup python -m env.env_service &> logs/14b_baseline_online_env_output.log &

    # 执行训练脚本
    print_green "Submitting training job..."
    cd $PROJECT_PATH  # 确保在正确的目录
    conda activate verl
    bash $TRAIN_SCRIPT

    # 保持脚本运行
    while true; do
        sleep 120
    done

else
    print_green "This is worker node: $HOSTNAME"

    # 等待master IP文件出现
    while [ ! -f $FILE_PATH ]; do
        print_green "Waiting for master node IP..."
        sleep 5
    done

    # 读取master IP
    MASTER_ADDR=$(cat $FILE_PATH)
    print_green "Found master node at $MASTER_ADDR"

    # 停止可能存在的Ray进程
    ray stop || true

    # 启动worker节点
    MAX_RETRIES=2
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        ray start --address $MASTER_ADDR:6379 --num-gpus 8 && break

        RETRY_COUNT=$((RETRY_COUNT + 1))
        print_red "Failed to start worker node, attempt $RETRY_COUNT of $MAX_RETRIES"
        sleep 10
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_red "Failed to start worker node after $MAX_RETRIES attempts"
        exit 1
    fi

    # 保持脚本运行
    while true; do
        sleep 120
    done
fi

