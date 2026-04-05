#!/usr/bin/env bash
# start.sh
# Run this inside WSL2 to start the LLM serving FastAPI application.
# Usage: bash start.sh

set -euo pipefail

# Determine the project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$HOME/llm_serve_env/bin/activate"

# Set the model path
export MODEL_PATH="$HOME/models/Qwen3.5-4B"

# Verify model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Run setup/setup_environment.sh first to download the model."
    exit 1
fi

echo "Starting LLM server..."
echo "  Model:  $MODEL_PATH"
echo "  Server: http://0.0.0.0:8765"
echo ""

# Change to project directory and start the server
cd "$SCRIPT_DIR"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8765 --reload
