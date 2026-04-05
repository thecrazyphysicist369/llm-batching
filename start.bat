@echo off
title LLM Serving Dashboard
echo ============================================
echo   LLM Serving Algorithms - GPU Dashboard
echo ============================================
echo.
echo Starting server via WSL2 (Ubuntu-22.04)...
echo Server will be available at: http://localhost:8765
echo Press Ctrl+C to stop.
echo.

wsl.exe -d Ubuntu-22.04 -u soujanyar --exec bash -c "source $HOME/llm_serve_env/bin/activate && export MODEL_PATH=$HOME/models/Qwen3.5-4B && export PATH=/usr/local/cuda-12.8/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH && cd '/mnt/c/Users/soujanyar/OneDrive - NVIDIA Corporation/Documents/fun_projects/llm_user' && echo 'Model: $MODEL_PATH' && echo 'Server: http://0.0.0.0:8765' && echo '' && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8765 --reload"

pause
