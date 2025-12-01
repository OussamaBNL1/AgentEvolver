#!/bin/bash

set -e
set -o pipefail

# Path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
APPWORLD_ROOT="$SCRIPT_DIR"
WORKSPACE_DIR="$BEYONDAGENT_DIR"


# 2. Environment variable configuration
echo "📁 Setting environment variables..."
export NODE_ENV=production
export WORKSPACE_DIR="$WORKSPACE_DIR"
export APPWORLD_ROOT="$APPWORLD_ROOT"
export PYTHONPATH="$BEYONDAGENT_DIR:$PYTHONPATH"



# 3. Conda environment creation
if ! conda info --envs | grep -w "appworld" &>/dev/null; then
    echo "🐍 Creating Conda environment appworld (Python 3.11)..."
    conda create -n appworld python=3.11.0 -y
else
    echo "⚠️ Conda environment 'appworld' already exists. Skipping creation (remove or modify if needed)."
fi

# 4. Install dependencies
echo "📦 Installing libcst..."
conda install -n appworld -y libcst

echo "📋 Installing Python dependencies..."
conda run -n appworld pip install -r "$SCRIPT_DIR/requirements.txt"

# 5. Initialize appworld
echo "📁 Initializing appworld..."
conda run -n appworld appworld install

# 6. Download data
echo "📦 Downloading data (fallback used if automatic download fails)..."
if ! conda run -n appworld appworld download data; then
    echo "⚠️ Automatic download failed. Attempting sanitized public fallback..."
    # Privacy sanitization: removed author-specific path; replace with a public placeholder.
    FALLBACK_URL="https://dail-wlcb.oss-accelerate.aliyuncs.com/public/appworld_data.zip"
    wget -O "$APPWORLD_ROOT/appworld_data.zip" "$FALLBACK_URL" || echo "⚠️ Fallback URL unavailable, please supply APPWORLD_DATA manually."
    if [ -f "$APPWORLD_ROOT/appworld_data.zip" ]; then
        mkdir -p /tmp/unziptemp
        unzip "$APPWORLD_ROOT/appworld_data.zip" -d /tmp/unziptemp
        mv /tmp/unziptemp/*/* "$APPWORLD_ROOT" 2>/dev/null || mv /tmp/unziptemp/* "$APPWORLD_ROOT" 2>/dev/null || true
        rm -rf /tmp/unziptemp "$APPWORLD_ROOT/appworld_data.zip"
    fi
fi

echo "✅ Setup complete!"

echo ""
echo "👉 Startup instructions:"
echo "----------------------------------------"
echo "source \$(conda info --base)/etc/profile.d/conda.sh"
echo "conda activate appworld"
echo "cd $BEYONDAGENT_DIR/env_service/launch_script"
echo "bash appworld.sh"
echo "----------------------------------------"
