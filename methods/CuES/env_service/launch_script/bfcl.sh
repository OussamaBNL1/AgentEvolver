#!/bin/bash
set -eu

# Note:
# Run EnvService/env_sandbox/environments/bfcl/bfcl_dataprocess.py first.
# Retrieve BFCL_DATA_PATH and BFCL_SPLID_ID_PATH and set the variables accordingly.

# Environment variables (update to actual paths as needed)

#
# Get launch_script directory
LAUNCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get env_service directory
ENV_SERVICE_DIR="$(dirname "$LAUNCH_SCRIPT_DIR")"

# Get bfcl environment directory
DEFAULT_BFCL_ENV_DIR="$ENV_SERVICE_DIR/environments/bfcl"
BFCL_ENV_DIR=${BFCL_ENV_DIR:-$DEFAULT_BFCL_ENV_DIR}

export ENV_PATH="$BFCL_ENV_DIR"
export BFCL_DATA_PATH="$BFCL_ENV_DIR/bfcl_data/multi_turn_base_processed.jsonl"
export BFCL_SPLID_ID_PATH="$BFCL_ENV_DIR/bfcl_data/multi_turn_base_split_ids.json"
export BFCL_ANSWER_PATH="$BFCL_ENV_DIR/bfcl_eval/possible_answer"

echo "🌍 Environment variables set:"
echo "ENV_PATH: $ENV_PATH"
echo "BFCL_DATA_PATH: $BFCL_DATA_PATH"
echo "BFCL_SPLID_ID_PATH: $BFCL_SPLID_ID_PATH"
echo "BFCL_ANSWER_PATH: $BFCL_ANSWER_PATH"

# Check required files
if [ -f "$BFCL_DATA_PATH" ]; then
    echo "✅ Data file exists: $BFCL_DATA_PATH"
else
    echo "❌ Data file missing: $BFCL_DATA_PATH"
fi

if [ -f "$BFCL_SPLID_ID_PATH" ]; then
    echo "✅ Split-ID file exists: $BFCL_SPLID_ID_PATH"
else
    echo "❌ Split-ID file missing: $BFCL_SPLID_ID_PATH"
fi

if [ -d "$BFCL_ANSWER_PATH" ]; then
    echo "✅ Answer folder exists: $BFCL_ANSWER_PATH"
else
    echo "❌ Answer folder missing: $BFCL_ANSWER_PATH"
fi

export OPENAI_API_KEY=xx

# only for multinode running
export RAY_ENV_NAME=bfcl 

# Get absolute path of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to project root (env_service)
PROJECT_ROOT="$SCRIPT_DIR/../../"
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Debug: show current working directory and PYTHONPATH
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Run Python entrypoint
exec python -m env_service.env_service --env bfcl --portal 127.0.0.1 --port 8080