#!/bin/bash


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
exec python -m env_service.env_service --env openworld --portal 127.0.0.1 --port 8080