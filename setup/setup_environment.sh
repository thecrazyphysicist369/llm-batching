#!/usr/bin/env bash
# setup_environment.sh
# Run this INSIDE WSL2 Ubuntu-22.04 to set up the LLM serving environment.
# Usage: bash setup/setup_environment.sh

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  WSL2 Environment Setup for LLM Serving${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# Determine the project directory (where this script lives is setup/, go one up)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Update apt packages ---
echo -e "${YELLOW}[1/6] Updating apt packages...${NC}"
sudo apt-get update -y
sudo apt-get upgrade -y

# --- Install system dependencies ---
echo -e "${YELLOW}[2/6] Installing Python 3.11, pip, venv, git, build-essential...${NC}"
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    wget \
    curl

# Make python3.11 the default python3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# --- Install CUDA Toolkit 12.8 for WSL-Ubuntu ---
# Drivers come from the Windows host. Only the toolkit is needed inside WSL.
echo -e "${YELLOW}[3/6] Installing CUDA Toolkit 12.8 for WSL-Ubuntu...${NC}"

# Remove any old CUDA GPG keys
sudo apt-key del 7fa2af80 2>/dev/null || true

# Add NVIDIA CUDA repository for WSL-Ubuntu
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb
sudo apt-get update -y

# Install CUDA toolkit only (no drivers)
sudo apt-get install -y cuda-toolkit-12-8

# Set up CUDA environment variables
CUDA_ENV_LINES='
# CUDA Toolkit 12.8
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
'

if ! grep -q "cuda-12.8" ~/.bashrc 2>/dev/null; then
    echo "$CUDA_ENV_LINES" >> ~/.bashrc
fi

# Source for current session
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

echo -e "${GREEN}[OK] CUDA Toolkit 12.8 installed.${NC}"

# --- Create Python virtual environment ---
echo -e "${YELLOW}[4/6] Creating Python virtual environment at ~/llm_serve_env...${NC}"

if [ -d "$HOME/llm_serve_env" ]; then
    echo -e "${GREEN}[OK] Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv "$HOME/llm_serve_env"
    echo -e "${GREEN}[OK] Virtual environment created.${NC}"
fi

# Activate venv
source "$HOME/llm_serve_env/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# --- Install Python requirements ---
echo -e "${YELLOW}[5/6] Installing Python requirements...${NC}"
pip install -r "$PROJECT_DIR/requirements.txt"
echo -e "${GREEN}[OK] Python packages installed.${NC}"

# --- Download the model ---
echo -e "${YELLOW}[6/6] Downloading Qwen3.5-4B model (this may take a while)...${NC}"

if [ -d "$HOME/models/Qwen3.5-4B" ] && [ "$(ls -A "$HOME/models/Qwen3.5-4B" 2>/dev/null)" ]; then
    echo -e "${GREEN}[OK] Model directory already exists and is not empty. Skipping download.${NC}"
    echo "    To re-download, remove ~/models/Qwen3.5-4B and run this script again."
else
    mkdir -p "$HOME/models"
    huggingface-cli download Qwen/Qwen3.5-4B --local-dir "$HOME/models/Qwen3.5-4B"
    echo -e "${GREEN}[OK] Model downloaded to ~/models/Qwen3.5-4B${NC}"
fi

# --- Done ---
echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Setup Complete!${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""
echo -e "${GREEN}Environment is ready. To start the server:${NC}"
echo ""
echo "  cd $PROJECT_DIR"
echo "  bash start.sh"
echo ""
echo -e "${GREEN}The server will be available at http://localhost:8765${NC}"
echo -e "${GREEN}Access from Windows browser at the same address.${NC}"
echo ""
