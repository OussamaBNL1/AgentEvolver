#!/bin/bash

set -e
set -o pipefail

# Path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"

WORKSPACE_DIR="$BEYONDAGENT_DIR"


# 2. Environment variable configuration
echo "📁 Setting environment variables..."
export NODE_ENV=production
export WORKSPACE_DIR="$WORKSPACE_DIR"

export PYTHONPATH="$BEYONDAGENT_DIR:$PYTHONPATH"



# 3. Create Conda environment
if ! conda info --envs | grep -w "openworld" &>/dev/null; then
    echo "🐍 Creating Conda environment 'openworld' (Python 3.11)..."
    conda create -n openworld python=3.11.0 -y
else
    echo "⚠️ Conda environment 'openworld' already exists; delete or rename if needed (skipping creation)."
fi
# 4. Install dependencies

echo "📋 Installing Python dependencies..."
conda run -n openworld pip install -r "$SCRIPT_DIR/requirements.txt"

# 5. Initialize openworld
echo "📁 Initializing openworld..."


echo ""
echo "👉 Launch instructions:"
echo "----------------------------------------"
echo "source \$(conda info --base)/etc/profile.d/conda.sh"
echo "conda activate openworld"
echo "cd $BEYONDAGENT_DIR/env_service/launch_script"
echo "bash openworld.sh"
echo "----------------------------------------"
