"""FastAPI application – WebSocket-driven LLM serving demo (vLLM backend)."""

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from vllm import SamplingParams

from .algorithms import ALGORITHM_MAP
from .algorithms.base import BaseAlgorithm
from .gpu_monitor import GPUMonitor
from .model_manager import ModelManager, DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
REPORTS_DIR = BASE_DIR / "reports"
PROMPTS_FILE = BASE_DIR / "prompts.txt"

# ---------------------------------------------------------------------------
# App globals
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM Serving Algorithms Demo")

gpu_monitor = GPUMonitor()
model_manager = ModelManager()


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info("WebSocket connected (%d total)", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        logger.info("WebSocket disconnected (%d remaining)", len(self.active))

    async def broadcast(self, message: dict) -> None:
        # Intercept benchmark results to store in telemetry
        if message.get("type") == "benchmark_result":
            algo = message.get("algorithm", "")
            if algo:
                if algo not in algo_telemetry:
                    algo_telemetry[algo] = {"runs": [], "benchmark": None, "gpu_samples": []}
                algo_telemetry[algo]["benchmark"] = {
                    "total_tps": message.get("total_tokens_per_sec", 0),
                    "per_user_tps": message.get("per_user_tps", []),
                }

        data = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send(self, ws: WebSocket, message: dict) -> None:
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            self.disconnect(ws)


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
current_algorithm: BaseAlgorithm | None = None
algorithm_task: asyncio.Task | None = None
benchmark_history: list[dict[str, Any]] = []

algo_telemetry: dict[str, dict] = {}
telemetry_capture_task: asyncio.Task | None = None
current_capture_algo: str | None = None
benchmark_all_task: asyncio.Task | None = None
benchmark_all_cancel = False


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    logger.info("Starting up – loading vLLM engine ...")
    await manager.broadcast(
        {"type": "model_status", "status": "loading", "message": "Loading Qwen3.5-4B via vLLM..."}
    )
    try:
        await model_manager.load_engine(DEFAULT_CONFIG)
        await manager.broadcast({
            "type": "model_status",
            "status": "ready",
            "message": "vLLM engine loaded and ready!",
        })
    except Exception as exc:
        logger.exception("Engine load failed")
        await manager.broadcast({
            "type": "model_status",
            "status": "error",
            "message": f"Engine load error: {exc}",
        })


@app.on_event("shutdown")
async def shutdown() -> None:
    gpu_monitor.shutdown()
    await model_manager.unload_engine()


# ---------------------------------------------------------------------------
# Static files & index
# ---------------------------------------------------------------------------
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not found."}


@app.get("/report/latest")
async def get_latest_report():
    """Serve the latest saved benchmark report."""
    report_path = REPORTS_DIR / "latest_report.html"
    if report_path.exists():
        return FileResponse(str(report_path), media_type="text/html")
    return {"error": "No report saved yet. Run Benchmark All first."}


@app.get("/report/data")
async def get_latest_report_data():
    """Return the latest report data as JSON."""
    data_path = REPORTS_DIR / "latest_report.json"
    if data_path.exists():
        return FileResponse(str(data_path), media_type="application/json")
    return {"error": "No report data yet."}


@app.get("/report/list")
async def list_reports():
    """List all saved report files (timestamped)."""
    if not REPORTS_DIR.exists():
        return {"reports": []}
    files = sorted(REPORTS_DIR.glob("report_*.html"), reverse=True)
    return {
        "reports": [
            {"name": f.stem, "url": f"/report/file/{f.name}"}
            for f in files
        ]
    }


@app.get("/report/file/{filename}")
async def get_report_file(filename: str):
    """Serve a specific saved report by filename."""
    # Sanitize filename
    safe = Path(filename).name
    report_path = REPORTS_DIR / safe
    if report_path.exists() and report_path.suffix == ".html":
        return FileResponse(str(report_path), media_type="text/html")
    return {"error": "Report not found."}


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------
def save_benchmark_report(telemetry: dict, llm_analysis: str) -> None:
    """Save benchmark results as JSON + standalone HTML report."""
    import datetime

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Build report data
    report_data = {
        "timestamp": timestamp,
        "gpu": "NVIDIA RTX 3500 Ada 12GB",
        "model": "Qwen3.5-4B",
        "backend": "vLLM 0.19.0",
        "algorithms": {},
        "llm_analysis": llm_analysis,
    }

    for algo_name, data in telemetry.items():
        runs = data.get("runs", [])
        if not runs and not data.get("benchmark"):
            continue
        report_data["algorithms"][algo_name] = {
            "avg_tps": data.get("avg_tps", data.get("benchmark", {}).get("total_tps", 0) if data.get("benchmark") else 0),
            "per_run_tps": [round(r["total_tps"], 2) for r in runs] if runs else [],
            "avg_gpu_util": data.get("avg_gpu_util", 0),
            "avg_power_w": data.get("avg_power_w", 0),
            "avg_mem_used_mb": data.get("avg_mem_used_mb", 0),
            "avg_temp_c": data.get("avg_temp_c", 0),
            "mem_total_mb": data.get("mem_total_mb", 0),
            "num_runs": data.get("num_runs", len(runs)),
        }

    # Save JSON
    json_path = REPORTS_DIR / "latest_report.json"
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    # Also save timestamped copy
    (REPORTS_DIR / f"report_{timestamp}.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )

    # Generate standalone HTML report
    algos = report_data["algorithms"]
    sorted_algos = sorted(algos.items(), key=lambda x: x[1].get("avg_tps", 0), reverse=True)
    best = sorted_algos[0][0] if sorted_algos else ""

    algo_names_map = {
        "naive": "Naive Sequential", "static_batch": "Static Batching",
        "continuous_batch": "Continuous Batching", "paged_attention": "PagedAttention",
        "quantized": "Quantized (4-bit)", "speculative": "Speculative Decoding",
        "chunked_prefill": "Chunked Prefill",
    }

    table_rows = ""
    chart_data_js = "["
    for algo_key, d in sorted_algos:
        name = algo_names_map.get(algo_key, algo_key)
        is_best = algo_key == best
        cls = ' class="winner"' if is_best else ""
        star = " ★" if is_best else ""
        table_rows += f"""<tr{cls}>
            <td>{name}{star}</td>
            <td>{d.get('avg_tps', 0):.2f}</td>
            <td>{d.get('avg_gpu_util', 0):.1f}</td>
            <td>{int(d.get('avg_mem_used_mb', 0))}</td>
            <td>{d.get('avg_power_w', 0):.1f}</td>
            <td>{d.get('avg_temp_c', 0):.1f}</td>
            <td>{d.get('num_runs', 0)}</td>
            <td>{d.get('per_run_tps', [])}</td>
        </tr>\n"""
        chart_data_js += f'{{name:"{name}",tps:{d.get("avg_tps", 0):.2f}}},'
    chart_data_js += "]"

    # Escape LLM analysis for HTML
    import html as html_mod
    safe_analysis = html_mod.escape(llm_analysis) if llm_analysis else "No LLM analysis available."

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Report - {timestamp}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0d1a; color: #e0e0f0; font-family: 'Segoe UI', sans-serif; padding: 32px; }}
  h1 {{ color: #ffd740; margin-bottom: 8px; }}
  .meta {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
  .section {{ margin-bottom: 28px; }}
  .section h2 {{ color: #6c63ff; font-size: 16px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }}
  canvas {{ background: #0a0a1e; border-radius: 10px; padding: 12px; width: 100%; max-width: 800px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-family: 'Cascadia Code', monospace; }}
  th {{ background: #1a1a35; color: #6c63ff; padding: 10px 14px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #2a2a4a; }}
  tr.winner td {{ color: #00e676; font-weight: bold; }}
  tr:hover td {{ background: rgba(108,99,255,0.08); }}
  .analysis {{ background: #13132b; border: 1px solid #2a2a4a; border-radius: 10px; padding: 20px; font-family: 'Cascadia Code', monospace; font-size: 12px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; max-height: 400px; overflow-y: auto; }}
  .footer {{ margin-top: 32px; color: #555; font-size: 11px; text-align: center; }}
</style>
</head>
<body>
  <h1>Algorithm Benchmark Report</h1>
  <div class="meta">
    Generated: {timestamp} | GPU: {report_data['gpu']} | Model: {report_data['model']} | Backend: {report_data['backend']}
  </div>

  <div class="section">
    <h2>Throughput Comparison (tok/s)</h2>
    <canvas id="chart" width="800" height="240"></canvas>
  </div>

  <div class="section">
    <h2>Detailed Metrics</h2>
    <table>
      <thead>
        <tr><th>Algorithm</th><th>Avg TPS</th><th>GPU Util %</th><th>VRAM MB</th><th>Power W</th><th>Temp C</th><th>Runs</th><th>Per-Run TPS</th></tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>LLM Cost Analysis</h2>
    <div class="analysis">{safe_analysis}</div>
  </div>

  <div class="footer">LLM Serving Algorithms Dashboard &mdash; Auto-generated benchmark report</div>

  <script>
    const data = {chart_data_js};
    const canvas = document.getElementById('chart');
    const c = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const colors = ['#4ECDC4','#FF6B6B','#45B7D1','#96CEB4','#6c63ff','#ffd740','#ff8a65'];
    const maxTps = Math.max(...data.map(d=>d.tps), 1);
    const barH = Math.min(26, (H-30)/data.length - 4);
    const labelW = 160, chartW = W - labelW - 70;
    c.fillStyle='#888'; c.font='bold 12px Segoe UI'; c.textAlign='left';
    c.fillText('Throughput (tok/s)', labelW, 16);
    data.forEach((d,i) => {{
      const y = 28 + i*(barH+6);
      const bw = Math.max(4, (d.tps/maxTps)*chartW);
      c.fillStyle='#ccc'; c.font='12px Cascadia Code, monospace'; c.textAlign='right';
      c.fillText(d.name, labelW-8, y+barH/2+4);
      c.fillStyle=colors[i%colors.length]; c.globalAlpha=0.8;
      c.beginPath(); c.roundRect(labelW, y, bw, barH, 4); c.fill();
      c.globalAlpha=1; c.fillStyle='#fff'; c.font='bold 11px monospace'; c.textAlign='left';
      c.fillText(d.tps.toFixed(1), labelW+bw+8, y+barH/2+4);
    }});
  </script>
</body>
</html>"""

    html_path = REPORTS_DIR / "latest_report.html"
    html_path.write_text(html_content, encoding="utf-8")

    # Also save timestamped copy
    (REPORTS_DIR / f"report_{timestamp}.html").write_text(
        html_content, encoding="utf-8"
    )

    logger.info("Benchmark report saved to %s", html_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def stop_current_algorithm() -> None:
    global current_algorithm, algorithm_task, telemetry_capture_task, current_capture_algo
    if current_algorithm is not None:
        current_algorithm.stop()
    if algorithm_task is not None and not algorithm_task.done():
        algorithm_task.cancel()
        try:
            await algorithm_task
        except (asyncio.CancelledError, Exception):
            pass
    if telemetry_capture_task and not telemetry_capture_task.done():
        telemetry_capture_task.cancel()
        try:
            await telemetry_capture_task
        except (asyncio.CancelledError, Exception):
            pass
    telemetry_capture_task = None
    current_capture_algo = None
    algorithm_task = None
    current_algorithm = None
    await manager.broadcast({"type": "animation_reset"})


async def start_telemetry_capture(algo_name: str) -> None:
    global telemetry_capture_task, current_capture_algo
    if telemetry_capture_task and not telemetry_capture_task.done():
        telemetry_capture_task.cancel()
        try:
            await telemetry_capture_task
        except (asyncio.CancelledError, Exception):
            pass
    current_capture_algo = algo_name
    if algo_name not in algo_telemetry:
        algo_telemetry[algo_name] = {"runs": [], "benchmark": None, "gpu_samples": []}
    telemetry_capture_task = asyncio.create_task(_capture_loop(algo_name))


async def _capture_loop(algo_name: str) -> None:
    try:
        while True:
            telemetry = gpu_monitor.get_telemetry()
            telemetry.pop("error", None)
            telemetry["timestamp"] = time.time()
            if algo_name in algo_telemetry:
                samples = algo_telemetry[algo_name]["gpu_samples"]
                samples.append(telemetry)
                if len(samples) > 200:
                    algo_telemetry[algo_name]["gpu_samples"] = samples[-200:]
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass


async def run_algorithm_loop(algo: BaseAlgorithm, prompts: dict[int, str]) -> None:
    try:
        while True:
            await algo.run(prompts)
            if not algo.auto_mode:
                break
            await asyncio.sleep(5)
            if not algo.auto_mode:
                break
            prompts = algo.pick_random_prompts()
            for uid, p in prompts.items():
                await algo.send_prompt(uid, p)
    except asyncio.CancelledError:
        logger.info("Algorithm task cancelled")
    except Exception:
        logger.exception("Algorithm error")
        await manager.broadcast({
            "type": "model_status",
            "status": "error",
            "message": f"Algorithm error: {traceback.format_exc()[-200:]}",
        })


# ---------------------------------------------------------------------------
# Evaluate: send telemetry to Qwen LLM, stream response
# ---------------------------------------------------------------------------
async def run_evaluate(ws: WebSocket) -> None:
    if not model_manager.is_loaded():
        await manager.send(ws, {"type": "evaluate_token", "token": "Error: Engine not loaded.", "done": True})
        return

    # Build summary
    summary_lines = ["=== LLM Serving Algorithm Evaluation Data ==="]
    summary_lines.append("GPU: NVIDIA RTX 3500 Ada 12GB | Model: Qwen3.5-4B | Backend: vLLM\n")

    has_data = False
    for algo_name, data in algo_telemetry.items():
        runs = data.get("runs", [])
        bench = data.get("benchmark")
        samples = data.get("gpu_samples", [])
        if not runs and not bench and not samples:
            continue
        has_data = True
        summary_lines.append(f"\n--- Algorithm: {algo_name} ---")

        if runs:
            tps_values = [r["total_tps"] for r in runs]
            avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0
            summary_lines.append(f"  Runs: {len(runs)}")
            summary_lines.append(f"  Avg tok/s: {avg_tps:.2f}")
            summary_lines.append(f"  Per-run tok/s: {[round(t, 2) for t in tps_values]}")
        elif bench:
            summary_lines.append(f"  Total tok/s: {bench.get('total_tps', 0):.2f}")

        if data.get("avg_gpu_util") is not None:
            summary_lines.append(f"  GPU Utilization (avg): {data['avg_gpu_util']}%")
            summary_lines.append(f"  VRAM used (avg): {data.get('avg_mem_used_mb', 0)} MB / {data.get('mem_total_mb', 0)} MB")
            summary_lines.append(f"  Power draw (avg): {data.get('avg_power_w', 0)} W")
            summary_lines.append(f"  Temperature (avg): {data.get('avg_temp_c', 0)} C")
        elif samples:
            avg_gpu = sum(s.get("gpu_util", 0) for s in samples) / len(samples)
            avg_mem = sum(s.get("mem_used_mb", 0) for s in samples) / len(samples)
            avg_pow = sum(s.get("power_w", 0) for s in samples) / len(samples)
            avg_tmp = sum(s.get("temp_c", 0) for s in samples) / len(samples)
            mem_t = samples[0].get("mem_total_mb", 0)
            summary_lines.append(f"  GPU Utilization (avg): {avg_gpu:.1f}%")
            summary_lines.append(f"  VRAM used (avg): {avg_mem:.0f} MB / {mem_t} MB")
            summary_lines.append(f"  Power draw (avg): {avg_pow:.1f} W")
            summary_lines.append(f"  Temperature (avg): {avg_tmp:.1f} C")

    if not has_data:
        await manager.send(ws, {"type": "evaluate_token", "token": "No telemetry data collected yet.", "done": True})
        return

    telemetry_text = "\n".join(summary_lines)
    logger.info("Evaluate: sending telemetry to model")

    await manager.send(ws, {"type": "evaluate_start"})

    # Generate evaluation using vLLM engine
    engine = model_manager.get_engine()
    tokenizer = model_manager.get_tokenizer()

    system_prompt = (
        "You are a cloud computing cost optimization expert. "
        "Given the GPU telemetry and throughput data for different LLM serving algorithms "
        "running the Qwen3.5-4B model on an NVIDIA RTX 3500 Ada (12GB) via vLLM, "
        "analyze which algorithm offers the best cost-to-performance ratio for cloud deployment. "
        "Consider: tokens/second throughput, GPU utilization efficiency, VRAM usage, power consumption, "
        "and how these translate to cloud costs ($/token). "
        "Recommend the best algorithm for production cloud serving and explain why. "
        "Be concise and practical."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": telemetry_text},
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    params = SamplingParams(temperature=0.7, max_tokens=512)
    request_id = f"evaluate-{time.time():.4f}"

    prev_text = ""
    llm_output = ""
    try:
        async for output in engine.generate(formatted, params, request_id):
            comp = output.outputs[0]
            new_text = comp.text[len(prev_text):]
            prev_text = comp.text
            if new_text:
                llm_output += new_text
                await manager.send(ws, {"type": "evaluate_token", "token": new_text, "done": False})
            if output.finished:
                break
    except Exception:
        logger.exception("Evaluate generation error")

    await manager.send(ws, {"type": "evaluate_token", "token": "", "done": True})

    # Auto-save report
    try:
        save_benchmark_report(algo_telemetry, llm_output)
        await manager.broadcast({"type": "report_saved"})
    except Exception:
        logger.exception("Failed to save report")


# ---------------------------------------------------------------------------
# Full Benchmark: run all algorithms sequentially
# ---------------------------------------------------------------------------
ALGO_ORDER = [
    "naive", "static_batch", "continuous_batch", "paged_attention",
    "chunked_prefill", "speculative", "quantized",
]
RUNS_PER_ALGO = 5
COOLDOWN_SECS = 30


async def run_full_benchmark(ws: WebSocket) -> None:
    global current_algorithm, benchmark_all_cancel

    benchmark_all_cancel = False
    total_algos = len(ALGO_ORDER)
    algo_telemetry.clear()

    for algo_idx, algo_name in enumerate(ALGO_ORDER):
        if benchmark_all_cancel:
            break

        await manager.broadcast({
            "type": "benchmark_progress",
            "phase": "starting",
            "algorithm": algo_name,
            "algo_idx": algo_idx,
            "total_algos": total_algos,
            "run": 0,
            "total_runs": RUNS_PER_ALGO,
        })

        # Reload engine if needed for this algorithm
        AlgoClass = ALGORITHM_MAP[algo_name]
        needed_config = AlgoClass.engine_config()
        if model_manager.current_config() != needed_config:
            await manager.broadcast({
                "type": "model_status", "status": "loading",
                "message": f"Reloading engine for {algo_name}...",
            })
            try:
                await model_manager.load_engine(needed_config)
            except Exception as exc:
                logger.exception("Engine reload failed for %s", algo_name)
                await manager.broadcast({
                    "type": "model_status", "status": "error",
                    "message": f"Engine reload failed for {algo_name}: {exc}",
                })
                continue
            await manager.broadcast({
                "type": "model_status", "status": "ready",
                "message": f"Engine ready for {algo_name}.",
            })

        algo_telemetry[algo_name] = {"runs": [], "gpu_samples": [], "benchmark": None}

        for run_idx in range(RUNS_PER_ALGO):
            if benchmark_all_cancel:
                break

            await manager.broadcast({
                "type": "benchmark_progress",
                "phase": "running",
                "algorithm": algo_name,
                "algo_idx": algo_idx,
                "total_algos": total_algos,
                "run": run_idx + 1,
                "total_runs": RUNS_PER_ALGO,
            })

            algo = AlgoClass(
                model_manager=model_manager,
                broadcast_fn=manager.broadcast,
                prompts_file=str(PROMPTS_FILE),
            )
            current_algorithm = algo

            # Capture telemetry for this run
            run_samples: list[dict] = []

            async def _run_capture():
                try:
                    while True:
                        t = gpu_monitor.get_telemetry()
                        t.pop("error", None)
                        t["timestamp"] = time.time()
                        run_samples.append(t)
                        await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    pass

            capture_task = asyncio.create_task(_run_capture())

            prompts = algo.pick_random_prompts()
            try:
                await algo.run(prompts)
            except Exception:
                logger.exception("Error during benchmark run %s/%d", algo_name, run_idx + 1)

            capture_task.cancel()
            try:
                await capture_task
            except (asyncio.CancelledError, Exception):
                pass

            run_bench = algo_telemetry[algo_name].get("benchmark")
            run_data = {
                "total_tps": run_bench["total_tps"] if run_bench else 0,
                "per_user_tps": run_bench.get("per_user_tps", []) if run_bench else [],
                "gpu_samples": run_samples,
            }
            algo_telemetry[algo_name]["runs"].append(run_data)
            algo_telemetry[algo_name]["gpu_samples"].extend(run_samples)

            await manager.broadcast({"type": "animation_reset"})
            current_algorithm = None
            await asyncio.sleep(2)

        if benchmark_all_cancel:
            break

        # Compute averages
        runs = algo_telemetry[algo_name]["runs"]
        if runs:
            avg_tps = sum(r["total_tps"] for r in runs) / len(runs)
            all_samples = algo_telemetry[algo_name]["gpu_samples"]
            s_len = len(all_samples) or 1
            algo_telemetry[algo_name].update({
                "avg_tps": round(avg_tps, 2),
                "avg_gpu_util": round(sum(s.get("gpu_util", 0) for s in all_samples) / s_len, 1),
                "avg_power_w": round(sum(s.get("power_w", 0) for s in all_samples) / s_len, 1),
                "avg_mem_used_mb": round(sum(s.get("mem_used_mb", 0) for s in all_samples) / s_len, 0),
                "avg_temp_c": round(sum(s.get("temp_c", 0) for s in all_samples) / s_len, 1),
                "mem_total_mb": all_samples[0].get("mem_total_mb", 0) if all_samples else 0,
                "num_runs": len(runs),
            })

        # Cooldown
        if algo_idx < total_algos - 1 and not benchmark_all_cancel:
            for sec in range(COOLDOWN_SECS, 0, -1):
                if benchmark_all_cancel:
                    break
                await manager.broadcast({
                    "type": "benchmark_progress",
                    "phase": "cooldown",
                    "algorithm": algo_name,
                    "algo_idx": algo_idx,
                    "total_algos": total_algos,
                    "countdown": sec,
                    "run": RUNS_PER_ALGO,
                    "total_runs": RUNS_PER_ALGO,
                })
                await asyncio.sleep(1)

    if benchmark_all_cancel:
        await manager.broadcast({"type": "benchmark_progress", "phase": "aborted"})
        return

    # Build results
    results = {}
    for algo_name in ALGO_ORDER:
        data = algo_telemetry.get(algo_name, {})
        if not data.get("runs"):
            continue
        results[algo_name] = {
            "avg_tps": data.get("avg_tps", 0),
            "per_run_tps": [round(r["total_tps"], 2) for r in data["runs"]],
            "avg_gpu_util": data.get("avg_gpu_util", 0),
            "avg_power_w": data.get("avg_power_w", 0),
            "avg_mem_used_mb": data.get("avg_mem_used_mb", 0),
            "avg_temp_c": data.get("avg_temp_c", 0),
            "mem_total_mb": data.get("mem_total_mb", 0),
            "num_runs": data.get("num_runs", 0),
        }

    await manager.broadcast({"type": "benchmark_complete", "results": results})

    # Reload default engine for evaluation if needed
    if model_manager.current_config() != DEFAULT_CONFIG:
        await manager.broadcast({
            "type": "model_status", "status": "loading",
            "message": "Reloading engine for evaluation...",
        })
        await model_manager.load_engine(DEFAULT_CONFIG)
        await manager.broadcast({
            "type": "model_status", "status": "ready",
            "message": "Engine ready.",
        })

    logger.info("Full benchmark complete, starting LLM evaluation...")
    await run_evaluate(ws)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    global current_algorithm, algorithm_task, benchmark_all_task, benchmark_all_cancel

    await manager.connect(ws)
    await gpu_monitor.start_broadcast(manager.broadcast, interval=0.5)

    if model_manager.is_loaded():
        await manager.send(ws, {
            "type": "model_status", "status": "ready",
            "message": "vLLM engine loaded and ready!",
        })
    else:
        await manager.send(ws, {
            "type": "model_status", "status": "loading",
            "message": "Engine is loading...",
        })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            # ----- select_algorithm -----
            if msg_type == "select_algorithm":
                algo_name = msg.get("algorithm", "")
                if algo_name not in ALGORITHM_MAP:
                    await manager.send(ws, {
                        "type": "model_status", "status": "error",
                        "message": f"Unknown algorithm: {algo_name}",
                    })
                    continue

                await stop_current_algorithm()

                # Reload engine if this algorithm needs a different config
                AlgoClass = ALGORITHM_MAP[algo_name]
                needed_config = AlgoClass.engine_config()
                if model_manager.current_config() != needed_config:
                    await manager.broadcast({
                        "type": "model_status", "status": "loading",
                        "message": f"Reloading engine for {algo_name}...",
                    })
                    try:
                        await model_manager.load_engine(needed_config)
                    except Exception as exc:
                        await manager.broadcast({
                            "type": "model_status", "status": "error",
                            "message": f"Reload failed: {exc}",
                        })
                        continue
                    await manager.broadcast({
                        "type": "model_status", "status": "ready",
                        "message": f"Engine ready for {algo_name}.",
                    })

                current_algorithm = AlgoClass(
                    model_manager=model_manager,
                    broadcast_fn=manager.broadcast,
                    prompts_file=str(PROMPTS_FILE),
                )
                logger.info("Algorithm selected: %s", algo_name)

            # ----- start_auto -----
            elif msg_type == "start_auto":
                if current_algorithm is None:
                    await manager.send(ws, {
                        "type": "model_status", "status": "error",
                        "message": "Select an algorithm first!",
                    })
                    continue

                if not model_manager.is_loaded():
                    await manager.send(ws, {
                        "type": "model_status", "status": "error",
                        "message": "Engine not loaded yet.",
                    })
                    continue

                if algorithm_task and not algorithm_task.done():
                    algo_name_saved = current_algorithm.name if current_algorithm else "naive"
                    await stop_current_algorithm()
                    if algo_name_saved in ALGORITHM_MAP:
                        AlgoClass = ALGORITHM_MAP[algo_name_saved]
                        current_algorithm = AlgoClass(
                            model_manager=model_manager,
                            broadcast_fn=manager.broadcast,
                            prompts_file=str(PROMPTS_FILE),
                        )

                current_algorithm.auto_mode = True
                prompts = current_algorithm.pick_random_prompts()
                algorithm_task = asyncio.create_task(
                    run_algorithm_loop(current_algorithm, prompts)
                )
                await start_telemetry_capture(current_algorithm.name)
                logger.info("Auto mode started")

            # ----- stop_auto -----
            elif msg_type == "stop_auto":
                if current_algorithm is not None:
                    current_algorithm.auto_mode = False
                    current_algorithm.stop()
                if telemetry_capture_task and not telemetry_capture_task.done():
                    telemetry_capture_task.cancel()
                await manager.broadcast({"type": "animation_reset"})
                logger.info("Auto mode stopped")

            # ----- manual_prompt -----
            elif msg_type == "manual_prompt":
                uid = msg.get("user_id", 0)
                prompt = msg.get("prompt", "Hello!")

                if current_algorithm is None:
                    await manager.send(ws, {
                        "type": "model_status", "status": "error",
                        "message": "Select an algorithm first!",
                    })
                    continue

                if not model_manager.is_loaded():
                    await manager.send(ws, {
                        "type": "model_status", "status": "error",
                        "message": "Engine not loaded yet.",
                    })
                    continue

                algo_name_for_manual = current_algorithm.name if current_algorithm else "naive"
                await stop_current_algorithm()
                AlgoClass = ALGORITHM_MAP.get(algo_name_for_manual, ALGORITHM_MAP["naive"])
                current_algorithm = AlgoClass(
                    model_manager=model_manager,
                    broadcast_fn=manager.broadcast,
                    prompts_file=str(PROMPTS_FILE),
                )

                all_prompts = {uid: prompt}
                algorithm_task = asyncio.create_task(
                    run_algorithm_loop(current_algorithm, all_prompts)
                )
                await start_telemetry_capture(current_algorithm.name)
                logger.info("Manual prompt for user %d: %s", uid, prompt[:50])

            # ----- evaluate -----
            elif msg_type == "evaluate":
                logger.info("Evaluate request received")
                asyncio.create_task(run_evaluate(ws))

            # ----- start_benchmark_all -----
            elif msg_type == "start_benchmark_all":
                if not model_manager.is_loaded():
                    await manager.send(ws, {
                        "type": "model_status", "status": "error",
                        "message": "Engine not loaded yet.",
                    })
                    continue

                await stop_current_algorithm()
                benchmark_all_cancel = False

                if benchmark_all_task and not benchmark_all_task.done():
                    benchmark_all_cancel = True
                    benchmark_all_task.cancel()
                    try:
                        await benchmark_all_task
                    except (asyncio.CancelledError, Exception):
                        pass

                benchmark_all_task = asyncio.create_task(run_full_benchmark(ws))
                logger.info("Benchmark All started")

            # ----- stop_benchmark_all -----
            elif msg_type == "stop_benchmark_all":
                benchmark_all_cancel = True
                if current_algorithm is not None:
                    current_algorithm.stop()
                await manager.broadcast({"type": "animation_reset"})
                logger.info("Benchmark All aborted by user")

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error")
    finally:
        manager.disconnect(ws)
        if not manager.active:
            gpu_monitor.stop_broadcast()
