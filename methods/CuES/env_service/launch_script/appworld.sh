#!/bin/bash


# You can override APPWORLD_ROOT to your own dataset path
# Use installed default path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
APPWORLD_ROOT="${APPWORLD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../environments/appworld" && pwd)}"
export APPWORLD_ROOT
echo "APPWORLD_ROOT: $APPWORLD_ROOT"

#
export RAY_ENV_NAME=appworld


# Get absolute path of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to project root (env_service)
PROJECT_ROOT="$SCRIPT_DIR/../../"
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Debug: show current working directory and PYTHONPATH
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Run Python entrypoint
exec python -m env_service.env_service --env appworld --portal 127.0.0.1 --port 8080