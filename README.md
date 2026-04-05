# LLM Batching: Interactive GPU Serving Algorithms Dashboard

An interactive, real-time dashboard for visualizing and benchmarking LLM serving algorithms on GPU. Built for CS engineers who want to **see and understand** how different serving strategies (naive, static batching, continuous batching, PagedAttention, speculative decoding, chunked prefill, quantization) affect throughput, GPU utilization, memory, and cost.

![Dashboard Overview](img%20(1).png)

## What This Project Does

This project runs a **Qwen3.5-4B** language model on your local NVIDIA GPU via **vLLM** and lets you:

- **Visualize** how each serving algorithm works with animated GPU pipeline diagrams
- **Benchmark** all 7 algorithms automatically (5 runs each, 30s cooldown, full telemetry capture)
- **Compare** throughput (tok/s), GPU utilization, VRAM usage, power draw, and temperature across algorithms
- **Evaluate** results using the same LLM to generate a cost/performance analysis for cloud deployment
- **Save & review** benchmark reports as standalone HTML files

![Benchmark Results](img%20(2).png)

## Algorithms Implemented

| Algorithm | What It Does | Key Insight |
|-----------|-------------|-------------|
| **Naive Sequential** | Processes one user at a time | ~75% GPU idle. Baseline for comparison. |
| **Static Batching** | Batches all users with padding to max length | Padding wastes compute. Everyone waits for the slowest. |
| **Continuous Batching** | Users complete independently, slots freed instantly | vLLM's default mode. No padding waste. Best throughput. |
| **PagedAttention** | KV cache allocated in non-contiguous pages | Like OS virtual memory. Eliminates fragmentation. |
| **Quantized (4-bit)** | Model compressed to 4-bit via bitsandbytes | ~4x less VRAM = more room for KV cache and concurrent users. |
| **Speculative Decoding** | Draft K tokens via n-gram matching, verify in 1 pass | When acceptance rate is high, get K tokens for ~2 forward passes. |
| **Chunked Prefill** | Long prompts split into chunks, interleaved with decode | Prevents prefill from starving other users' decode steps. |

## Features

- **Real-time GPU telemetry** - utilization, VRAM, power, temperature, clock speeds via NVML
- **Per-algorithm animated visualizations** - Gantt charts, padded bars, page tables, draft/verify sequences
- **4 user terminals** - type prompts manually or run auto-generated prompts
- **Benchmark All** - automated full benchmark suite across all 7 algorithms
- **LLM-powered evaluation** - the Qwen model itself analyzes the benchmark data and recommends the best algorithm
- **Auto-saved reports** - timestamped HTML + JSON reports viewable anytime
- **Dark / Light mode** - toggle with the moon/sun button
- **Tooltip hints** - hover over action buttons for explanations

## Architecture

```
Browser (HTML/CSS/Canvas)
    |
    |  WebSocket (real-time tokens, telemetry, animation frames)
    |
FastAPI Server (app/main.py)
    |
    |--- vLLM AsyncLLMEngine (GPU inference)
    |--- GPUMonitor (pynvml telemetry)
    |--- 7 Algorithm implementations
    |--- Report generator
```

- **Backend**: Python FastAPI + vLLM 0.19 + pynvml
- **Frontend**: Vanilla JS + Canvas API (no framework dependencies)
- **Model**: Qwen3.5-4B (auto-downloaded from HuggingFace)
- **GPU**: Any NVIDIA GPU with >= 12GB VRAM (tested on RTX 3500 Ada)

## Prerequisites

- **Windows 11** with **WSL2** (Ubuntu 22.04)
- **NVIDIA GPU** with >= 12GB VRAM and up-to-date drivers
- **Git** installed on Windows

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/thecrazyphysicist369/llm-batching.git
cd llm-batching
```

### 2. Install WSL2 + Ubuntu

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart when prompted, then complete the Ubuntu user setup (create username/password).

### 3. Run the Setup Script

Inside WSL2 (open Ubuntu terminal):

```bash
cd /mnt/c/path/to/llm-batching
bash setup/setup_environment.sh
```

This will:
- Install Python 3.11, build tools, git, curl
- Install CUDA Toolkit 12.8 for WSL-Ubuntu
- Create a Python virtual environment at `~/llm_serve_env`
- Install all Python dependencies (PyTorch, vLLM, FastAPI, transformers, etc.)
- Download the **Qwen3.5-4B** model (~8.8 GB) from HuggingFace to `~/models/Qwen3.5-4B`

This takes 15-30 minutes depending on your internet speed.

### 4. Start the Server

**Option A: From WSL terminal**

```bash
cd /mnt/c/path/to/llm-batching
bash start.sh
```

**Option B: From Windows (double-click)**

```
start.bat
```

The server starts on `http://localhost:8765`. Open this URL in your browser.

**Note:** The vLLM engine takes 30-60 seconds to load on first start (model loading + CUDA initialization). The browser will show "Model is loading..." until ready.

## Usage

### Quick Start

1. Open `http://localhost:8765` in your browser
2. Select an algorithm (e.g., **Continuous**) from the bottom bar
3. Type a prompt in any User terminal and press Enter
4. Watch the GPU animation, telemetry, and token streaming in real-time

### Auto Mode

- Click **Auto-Start** to run the selected algorithm continuously with random prompts
- Click **Stop** to halt

### Benchmark All Algorithms

- Click **Benchmark All** to run all 7 algorithms x 5 runs each
- A progress overlay shows current algorithm, run number, and cooldown timer
- After completion, an interactive results dashboard appears with:
  - Bar chart comparing throughput
  - Metrics table (TPS, GPU util, VRAM, power, temperature)
  - LLM-generated cost analysis streamed in real-time
- Reports are auto-saved to the `reports/` directory

### View Saved Reports

- Click **View Report** to open the latest report in a new tab
- Click **Records** to see all previously saved reports in a dropdown

### Theme

- Click the moon/sun icon to toggle between dark and light mode

## Project Structure

```
llm-batching/
  app/
    __init__.py
    main.py              # FastAPI server, WebSocket, benchmark orchestration
    model_manager.py     # vLLM engine lifecycle management
    gpu_monitor.py       # GPU telemetry via pynvml
    algorithms/
      __init__.py        # Algorithm registry (ALGORITHM_MAP)
      base.py            # Base class with vLLM helpers, WebSocket messaging
      naive.py           # Sequential processing
      static_batch.py    # Padded batch, everyone waits for longest
      continuous_batch.py # Independent completion (vLLM default)
      paged_attention.py # + Page table visualization overlay
      quantized.py       # Engine reload with 4-bit quantization
      speculative.py     # Engine reload with n-gram speculation
      chunked_prefill.py # Engine reload with chunked prefill
    static/
      index.html         # Dashboard HTML
      style.css          # Dark/Light theme CSS
      app.js             # Frontend: WebSocket, Canvas animations, controls
  setup/
    setup_environment.sh # Full WSL2 environment setup
    install_wsl.ps1      # PowerShell WSL installer
  reports/               # Auto-saved benchmark reports (HTML + JSON)
  prompts.txt            # Prompt pool for random generation
  requirements.txt       # Python dependencies
  start.sh               # Linux/WSL startup script
  start.bat              # Windows startup script
  README.md
```

## Configuration

### Engine Settings (`app/model_manager.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `gpu_memory_utilization` | 0.90 | Fraction of GPU VRAM to use |
| `max_model_len` | 512 | Maximum sequence length |
| `enforce_eager` | True | Skip CUDA graph compilation (saves VRAM) |
| `max_num_seqs` | 4 | Maximum concurrent sequences |

### Benchmark Settings (`app/main.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `RUNS_PER_ALGO` | 5 | Number of benchmark runs per algorithm |
| `COOLDOWN_SECS` | 30 | Seconds between algorithms during benchmark |
| `MAX_NEW_TOKENS` | 128 | Tokens generated per user per run |

## Troubleshooting

### "CUDA out of memory"
- Reduce `gpu_memory_utilization` to 0.80 in `app/model_manager.py`
- Reduce `max_model_len` to 256
- Close other GPU-using applications

### vLLM engine fails to start
- Ensure NVIDIA drivers are up to date on Windows (not inside WSL)
- Check `nvidia-smi` works in both Windows and WSL
- Verify CUDA toolkit is installed: `nvcc --version` in WSL

### WebSocket disconnects / "Connecting..."
- Refresh the page
- Check the WSL terminal for error messages
- Ensure port 8765 is not blocked by firewall

### Model download fails
- Check internet connectivity in WSL: `curl -I https://huggingface.co`
- Re-run `bash setup/setup_environment.sh` to retry download
- Manually download: `huggingface-cli download Qwen/Qwen3.5-4B --local-dir ~/models/Qwen3.5-4B`

## Tech Stack

- **[vLLM](https://github.com/vllm-project/vllm)** - High-throughput LLM serving engine
- **[FastAPI](https://fastapi.tiangolo.com/)** - Async Python web framework
- **[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)** - 4B parameter language model
- **[pynvml](https://pypi.org/project/pynvml/)** - NVIDIA GPU monitoring
- **HTML5 Canvas** - Real-time GPU pipeline animation
- **WebSocket** - Bidirectional real-time communication

## License

This project is for educational purposes. See individual component licenses (vLLM: Apache 2.0, Qwen: Apache 2.0).
