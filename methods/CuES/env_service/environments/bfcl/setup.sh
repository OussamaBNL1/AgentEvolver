#!/bin/bash

set -e
set -o pipefail

# Path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
BFCL_ROOT="$SCRIPT_DIR"
WORKSPACE_DIR="$BEYONDAGENT_DIR"


# 2. Environment variable configuration
echo "📁 Setting environment variables..."
export NODE_ENV=production
export WORKSPACE_DIR="$WORKSPACE_DIR"
export BFCL_ROOT="$BFCL_ROOT"
export PYTHONPATH="$BEYONDAGENT_DIR:$PYTHONPATH"


# 3. Create Conda environment
if ! conda info --envs | grep -w "bfcl" &>/dev/null; then
    echo "🐍 Creating Conda environment bfcl (Python 3.11.13)..."
    conda create -n bfcl python=3.11.13 -y
else
    echo "⚠️ Conda environment 'bfcl' already exists; delete or modify if needed (skipping creation)."
fi

# 4. Install dependencies
if [ -d "$SCRIPT_DIR/gorilla" ]; then
    echo "🔄 Updating gorilla repository..."
    cd "$SCRIPT_DIR/gorilla"
    git pull
else
    echo "📦 Cloning gorilla repository..."
    git clone https://github.com/ShishirPatil/gorilla.git
fi

echo "📋 Installing Python dependencies..."

conda run -n bfcl pip install -e "$SCRIPT_DIR/gorilla/berkeley-function-call-leaderboard/."
conda run -n bfcl pip install -r "$SCRIPT_DIR/requirements.txt"

# 5. Prepare data
echo "📁 Preparing BFCL data..."
cp -r "$SCRIPT_DIR/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data" "$SCRIPT_DIR/bfcl_eval"

cd "$SCRIPT_DIR/"
echo "Current working directory: $(pwd)"
echo "Script directory: $SCRIPT_DIR"

python "$SCRIPT_DIR/bfcl_dataprocess.py"

# 6. Export runtime environment variables
echo "🌎 Exporting runtime environment variables..."
export ENV_PATH="$SCRIPT_DIR"
export BFCL_DATA_PATH="$ENV_PATH/bfcl_data/multi_turn_base_processed.jsonl"
export BFCL_SPLID_ID_PATH="$ENV_PATH/bfcl_data/multi_turn_base_split_ids.json"
export BFCL_ANSWER_PATH="$ENV_PATH/bfcl_eval/possible_answer"
export OPENAI_API_KEY="$OPENAI_API_KEY"



echo "✅ Setup complete!"

echo ""


echo "To switch datasets, use the following commands:"
echo ""
echo "Available dataset names:"
echo "- all"
echo "- all_scoring"
echo "- multi_turn"
echo "- single_turn"
echo "- live"
echo "- non_live"
echo "- non_python"
echo "- python"
echo "- multi_turn_base"
echo ""
echo "Example commands:"
echo "export DATASET_NAME=multi_turn_base"
echo "export BFCL_DATA_PATH=\"$ENV_PATH/bfcl_data/\${DATASET_NAME}_processed.jsonl\""
echo "export BFCL_SPLID_ID_PATH=\"$ENV_PATH/bfcl_data/\${DATASET_NAME}_split_ids.json\""
echo ""
echo "Replace $DATASET_NAME with the desired dataset name."

echo "👉 Launch instructions:"
echo "----------------------------------------"
echo "source \$(conda info --base)/etc/profile.d/conda.sh"
echo "conda activate bfcl"
echo "cd $BEYONDAGENT_DIR/env_service/launch_script"
echo "bash bfcl.sh"
echo "----------------------------------------"

exec bash
