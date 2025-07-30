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

# Define environment variables
# export ENV_PATH="/mnt/data/taoshuchang.tsc/beyondagent/EnvService"
export RAY_CLUSTER_MODE="multi_node"
export ENV_PATH="/mnt/data/taoshuchang.tsc/beyondagent/EnvService_copy"
export PROJECT_PATH="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent"
suffix="qwen3_14b_adv_groupnorm_std_trbs16_ppobs16"
export TRAIN_SCRIPT="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/examples/wulanchabu/run_tsc_${suffix}.sh"

# 在最开始就设置PYTHONPATH，确保所有节点都能找到beyondagent模块
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"
echo "Initial PYTHONPATH set to: $PYTHONPATH"

export FILE_PATH="/mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/logs/assignment/$MASTER_ADDR-master-ip.log"
export EXPECTED_WORKERS=$WORLD_SIZE

log "Starting script execution"
log "TRAIN_SCRIPT set to: $TRAIN_SCRIPT"
log "PROJECT_PATH set to: $PROJECT_PATH"

# Activate conda environment
source /mnt/data/taoshuchang.tsc/anaconda3/etc/profile.d/conda.sh
conda activate verl
echo "Current Python path: $(which python)"

# 验证beyondagent模块是否可以导入
print_green "Verifying beyondagent module import..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_PATH')
try:
    import beyondagent
    print('✅ beyondagent module imported successfully')
    print(f'beyondagent location: {beyondagent.__file__}')
except ImportError as e:
    print(f'❌ Failed to import beyondagent: {e}')
    print(f'Python path: {sys.path}')
    exit(1)
" || {
    print_red "Failed to import beyondagent module!"
    print_red "Please check if the module is properly installed"
    exit 1
}

# 检查worker节点数量
check_workers() {
    worker_count=$(ray status 2>/dev/null | grep "node_" | wc -l)
    if [ -z "$worker_count" ]; then
        echo 0 # worker_count=0
    fi
    echo $worker_count # $((worker_count-1)) 
}

# 检查GPU资源是否完全就绪
check_gpu_resources() {
    gpu_count=$(ray status 2>/dev/null | grep -A 10 "Resources" | grep "GPU" | awk '{print $1}' | cut -d'/' -f2)
    if [ -z "$gpu_count" ]; then
        echo 0
    else
        # 将浮点数转换为整数
        printf "%.0f" "$gpu_count"
    fi
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
print_green "WORLD_SIZE: $WORLD_SIZE"
print_green "Expected total GPUs: $((WORLD_SIZE * 8))"
print_green "All environment variables:"
env | sort
print_green "========================="

# rm -f "$FILE_PATH"

# 判断是否是master节点
if [[ $HOSTNAME == *"-master-"* ]]; then
    print_green "This is master node: $HOSTNAME"

    # 停止可能存在的Ray进程
    ray stop --force || true

    # 启动master节点
    print_green "Starting Ray head node at $MASTER_ADDR"
    ray start --head \
        --node-ip-address $MASTER_ADDR \
        --num-gpus 8

    # 将master IP写入共享目录
    echo $MASTER_ADDR > $FILE_PATH

    # 等待所有worker节点加入
    print_green "Waiting for all worker nodes to join..."
    TIMEOUT=1000  # 10分钟超时
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

    # 额外等待GPU资源完全就绪
    print_green "Waiting for GPU resources to be fully available..."
    EXPECTED_GPUS=$((WORLD_SIZE * 8))
    GPU_TIMEOUT=300  # 5分钟超时
    GPU_ELAPSED=0

    while true; do
        current_gpus=$(check_gpu_resources)
        print_green "Current GPU count: $current_gpus/$EXPECTED_GPUS"

        if [ "$current_gpus" -eq "$EXPECTED_GPUS" ]; then
            print_green "All GPUs are available!"
            break
        fi

        if [ "$GPU_ELAPSED" -ge "$GPU_TIMEOUT" ]; then
            print_red "Timeout waiting for GPUs. Only $current_gpus/$EXPECTED_GPUS GPUs available."
            # 显示详细的集群状态用于调试
            ray status
            exit 1
        fi

        sleep 5
        GPU_ELAPSED=$((GPU_ELAPSED + 5))
    done

    # 检查集群状态
    print_green "Final cluster status before training:"
    ray status

    # 等待Ray dashboard完全启动
    print_green "Waiting for Ray dashboard to be ready..."
    while ! curl -s http://127.0.0.1:8265 > /dev/null; do
        sleep 5
    done

    # 执行训练脚本
    conda activate appworld
    cd $ENV_PATH
    echo "Environment service Python path: $(which python)"
    # python -m env.env_service &
    nohup python -m env.env_service &> "/mnt/data/taoshuchang.tsc/beyondagent/EnvService_copy/logs/qwen3/${suffix}.log" &

    # 等待环境服务启动
    sleep 15

    # 执行训练脚本
    print_green "Starting training job..."
    cd $PROJECT_PATH  # 确保在正确的目录
    conda activate verl
    echo "Training Python path: $(which python)"

    # 关键修改：设置RAY_ADDRESS环境变量，让训练脚本直接连接到Ray集群
    export RAY_ADDRESS="ray://localhost:10001"
    
    # 确保PYTHONPATH在训练环境中正确设置
    export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"
    echo "Training PYTHONPATH set to: $PYTHONPATH"
    
    # 验证beyondagent模块在训练环境中的可用性
    print_green "Verifying beyondagent in training environment..."
    python -c "
import sys
sys.path.insert(0, '$PROJECT_PATH')
try:
    import beyondagent
    print('✅ beyondagent available in training environment')
    print(f'Module path: {beyondagent.__file__}')
except ImportError as e:
    print(f'❌ beyondagent import failed in training environment: {e}')
    exit(1)
" || {
    print_red "beyondagent not available in training environment!"
    exit 1
}
    
    # 验证Ray连接
    print_green "Verifying Ray cluster connection..."
    python -c "
import ray
import sys
sys.path.insert(0, '$PROJECT_PATH')

try:
    ray.init(address='ray://localhost:10001', ignore_reinit_error=True)
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    print('✅ Ray cluster connection successful')
    print(f'Cluster resources: {cluster_resources}')
    print(f'Available resources: {available_resources}')
    
    # 测试远程模块导入
    @ray.remote
    def test_remote_import():
        import sys
        sys.path.insert(0, '$PROJECT_PATH')
        import beyondagent
        return f'Remote import successful: {beyondagent.__file__}'
    
    try:
        result = ray.get(test_remote_import.remote())
        print(f'✅ {result}')
    except Exception as e:
        print(f'❌ Remote import failed: {e}')
        raise e
    
    ray.shutdown()
except Exception as e:
    print(f'❌ Ray cluster connection or remote import failed: {e}')
    exit(1)
"
    
    if [ $? -ne 0 ]; then
        print_red "Failed to connect to Ray cluster"
        exit 1
    fi

    # 调用训练脚本
    bash $TRAIN_SCRIPT

else
    print_green "This is worker node: $HOSTNAME"

    # 在worker节点上也设置PYTHONPATH
    export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"
    echo "Worker PYTHONPATH set to: $PYTHONPATH"

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
    MAX_RETRIES=3  # 增加重试次数
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if ray start --address $MASTER_ADDR:6379 --num-gpus 8; then
            print_green "Worker node started successfully"
            break
        fi

        RETRY_COUNT=$((RETRY_COUNT + 1))
        print_red "Failed to start worker node, attempt $RETRY_COUNT of $MAX_RETRIES"
        sleep 10
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_red "Failed to start worker node after $MAX_RETRIES attempts"
        exit 1
    fi

    # Worker节点保持运行状态
    print_green "Worker node is running, keeping alive..."
    while true; do
        sleep 60
        # 检查与Ray集群的连接状态
        if ! ray status > /dev/null 2>&1; then
            print_red "Lost connection to Ray cluster, exiting..."
            break
        fi
    done
fi