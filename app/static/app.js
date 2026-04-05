/* ================================================================
   LLM Serving Dashboard - Frontend Application
   ================================================================ */

(function () {
  "use strict";

  // ── Constants ──────────────────────────────────────────────────
  const USER_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"];
  const USER_COLORS_DIM = [
    "rgba(255,107,107,0.25)",
    "rgba(78,205,196,0.25)",
    "rgba(69,183,209,0.25)",
    "rgba(150,206,180,0.25)",
  ];
  const ALGO_NAMES = {
    naive: "Naive Sequential",
    static_batch: "Static Batching",
    continuous_batch: "Continuous Batching",
    paged_attention: "PagedAttention",
    quantized: "Quantized (4-bit)",
    speculative: "Speculative Decoding",
    chunked_prefill: "Chunked Prefill",
  };
  const ALGO_SHORT = {
    naive: "Naive",
    static_batch: "Static",
    continuous_batch: "Contin.",
    paged_attention: "Paged",
    quantized: "Quant.",
    speculative: "Spec.",
    chunked_prefill: "Chunked",
  };
  const WS_URL = `ws://${location.hostname || "localhost"}:8765/ws`;
  const RECONNECT_DELAY = 2000;

  // ── State ──────────────────────────────────────────────────────
  let ws = null;
  let autoRunning = false;
  let currentAlgo = "naive";

  const benchmarks = {}; // algo -> { total_tps, per_user }
  const userTps = [0, 0, 0, 0];

  // GPU animation state (updated from server)
  let gpuAnim = {
    algorithm: "naive",
    active_users: [],
    memory_layout: { model_mb: 0, kv_cache_mb: [0, 0, 0, 0], free_mb: 0 },
    compute_slots: [
      { user_id: null, active: false },
      { user_id: null, active: false },
      { user_id: null, active: false },
      { user_id: null, active: false },
    ],
  };

  // Particle system for token flow
  const particles = []; // {x, y, vx, vy, color, life, maxLife}
  let animFrame = 0;

  // Algorithm-specific animation state (reset on algo switch)
  let benchmarkAllRunning = false;
  let benchmarkResults = null;
  let prevAlgorithm = "";
  let naiveTimeline = [];       // [{uid, startFrame, endFrame}]
  let specTokenHistory = [];    // [{text, status: "draft"|"accepted"|"rejected"|"corrected"}]
  let chunkedTimeline = [];     // [{type, uid, ...}]

  // ── DOM References ─────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const canvas = $("#gpuCanvas");
  const ctx = canvas.getContext("2d");

  const termBodies = [
    $("#termBody0"),
    $("#termBody1"),
    $("#termBody2"),
    $("#termBody3"),
  ];
  const termInputs = [
    $("#termInput0"),
    $("#termInput1"),
    $("#termInput2"),
    $("#termInput3"),
  ];
  const termTpsEls = [
    $("#termTps0"),
    $("#termTps1"),
    $("#termTps2"),
    $("#termTps3"),
  ];
  const tpsUserEls = [
    $("#tpsUser0"),
    $("#tpsUser1"),
    $("#tpsUser2"),
    $("#tpsUser3"),
  ];

  // Track active cursors per terminal
  const activeCursors = [null, null, null, null];

  // ── WebSocket ──────────────────────────────────────────────────
  function connect() {
    updateStatus("Connecting...", "");
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      updateStatus("Connected", "connected");
      // Send current algorithm selection
      wsSend({ type: "select_algorithm", algorithm: currentAlgo });
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        handleMessage(msg);
      } catch (e) {
        console.error("WS parse error:", e);
      }
    };

    ws.onclose = () => {
      updateStatus("Disconnected - reconnecting...", "error");
      setTimeout(connect, RECONNECT_DELAY);
    };

    ws.onerror = () => {
      updateStatus("Connection error", "error");
    };
  }

  function wsSend(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }

  function updateStatus(text, cls) {
    $("#statusText").textContent = text;
    const dot = $("#statusDot");
    dot.className = "status-dot";
    if (cls) dot.classList.add(cls);
  }

  // ── Message Router ─────────────────────────────────────────────
  function handleMessage(msg) {
    switch (msg.type) {
      case "model_status":
        handleModelStatus(msg);
        break;
      case "user_prompt":
        handleUserPrompt(msg);
        break;
      case "user_token":
        handleUserToken(msg);
        break;
      case "user_done":
        handleUserDone(msg);
        break;
      case "gpu_telemetry":
        handleTelemetry(msg);
        break;
      case "gpu_animation":
        handleGpuAnimation(msg);
        break;
      case "benchmark_result":
        handleBenchmark(msg);
        break;
      case "animation_reset":
        handleAnimationReset();
        break;
      case "evaluate_start":
        handleEvaluateStart();
        break;
      case "evaluate_token":
        handleEvaluateToken(msg);
        break;
      case "benchmark_progress":
        handleBenchmarkProgress(msg);
        break;
      case "benchmark_complete":
        handleBenchmarkComplete(msg);
        break;
      case "report_saved":
        handleReportSaved();
        break;
    }
  }

  // ── Model Status ───────────────────────────────────────────────
  function handleModelStatus(msg) {
    const s = msg.status;
    const m = msg.message || "";
    if (s === "loading") {
      updateStatus("Model: Loading... " + m, "");
    } else if (s === "ready") {
      updateStatus("Model: Ready  |  " + ALGO_NAMES[currentAlgo], "connected");
    } else {
      updateStatus("Model: Error - " + m, "error");
    }
  }

  // ── Terminal: Prompt ───────────────────────────────────────────
  function handleUserPrompt(msg) {
    const uid = msg.user_id;
    const body = termBodies[uid];
    // Remove any existing cursor
    removeCursor(uid);

    const promptEl = document.createElement("div");
    promptEl.className = "term-prompt";
    promptEl.textContent = "> " + msg.prompt;
    body.appendChild(promptEl);

    // Create response container with cursor
    const respEl = document.createElement("span");
    respEl.className = "term-response";
    respEl.dataset.active = "1";
    body.appendChild(respEl);

    const cursor = document.createElement("span");
    cursor.className = "term-cursor";
    body.appendChild(cursor);
    activeCursors[uid] = cursor;

    scrollTerminal(uid);
  }

  // ── Terminal: Token ────────────────────────────────────────────
  function handleUserToken(msg) {
    const uid = msg.user_id;
    const body = termBodies[uid];
    // Find active response span
    const activeSpans = body.querySelectorAll('.term-response[data-active="1"]');
    const span = activeSpans[activeSpans.length - 1];
    if (span) {
      span.textContent += msg.token;
    }
    // Update tokens/sec displays
    userTps[uid] = msg.tokens_per_sec || 0;
    termTpsEls[uid].textContent = userTps[uid].toFixed(1) + " tok/s";
    tpsUserEls[uid].textContent = userTps[uid].toFixed(1);

    // Spawn output particle
    spawnOutputParticle(uid);

    scrollTerminal(uid);
  }

  // ── Terminal: Done ─────────────────────────────────────────────
  function handleUserDone(msg) {
    const uid = msg.user_id;
    const body = termBodies[uid];

    // Deactivate response span
    const activeSpans = body.querySelectorAll('.term-response[data-active="1"]');
    const span = activeSpans[activeSpans.length - 1];
    if (span) {
      span.removeAttribute("data-active");
    }

    removeCursor(uid);

    // Done line
    const doneEl = document.createElement("div");
    doneEl.className = "term-done";
    doneEl.textContent = `[Done: ${msg.total_tokens} tokens, ${msg.avg_tokens_per_sec.toFixed(1)} tok/s]`;
    body.appendChild(doneEl);

    // Countdown
    if (msg.wait_seconds && msg.wait_seconds > 0) {
      const cdEl = document.createElement("div");
      cdEl.className = "term-countdown";
      body.appendChild(cdEl);
      startCountdown(cdEl, msg.wait_seconds);
    }

    // Update final tps
    userTps[uid] = msg.avg_tokens_per_sec;
    termTpsEls[uid].textContent = userTps[uid].toFixed(1) + " tok/s";
    tpsUserEls[uid].textContent = userTps[uid].toFixed(1);

    scrollTerminal(uid);
  }

  function removeCursor(uid) {
    if (activeCursors[uid]) {
      activeCursors[uid].remove();
      activeCursors[uid] = null;
    }
  }

  function startCountdown(el, seconds) {
    let remaining = seconds;
    el.textContent = `Next prompt in ${remaining}s...`;
    const iv = setInterval(() => {
      remaining--;
      if (remaining <= 0) {
        clearInterval(iv);
        el.textContent = "";
      } else {
        el.textContent = `Next prompt in ${remaining}s...`;
      }
    }, 1000);
  }

  function scrollTerminal(uid) {
    const body = termBodies[uid];
    requestAnimationFrame(() => {
      body.scrollTop = body.scrollHeight;
    });
  }

  // ── Telemetry ──────────────────────────────────────────────────
  function handleTelemetry(msg) {
    $("#telemetryStatus").textContent = "Live";

    setBar("barGpuUtil", msg.gpu_util, utilColor(msg.gpu_util));
    $("#valGpuUtil").textContent = msg.gpu_util + "%";

    const memPct =
      msg.mem_total_mb > 0
        ? Math.round((msg.mem_used_mb / msg.mem_total_mb) * 100)
        : 0;
    setBar("barMemUsage", memPct);
    $("#valMemUsage").textContent =
      msg.mem_used_mb + " / " + msg.mem_total_mb + " MB";

    setBar("barMemUtil", msg.mem_util);
    $("#valMemUtil").textContent = msg.mem_util + "%";

    const powerPct =
      msg.power_cap_w > 0
        ? Math.round((msg.power_w / msg.power_cap_w) * 100)
        : 0;
    setBar("barPower", powerPct);
    $("#valPower").textContent =
      Math.round(msg.power_w) + " / " + Math.round(msg.power_cap_w) + " W";

    const tempPct = Math.min(100, Math.round((msg.temp_c / 100) * 100));
    setBar("barTemp", tempPct, tempColor(msg.temp_c));
    $("#valTemp").textContent = msg.temp_c + " C";

    setBar("barFan", msg.fan_speed);
    $("#valFan").textContent = msg.fan_speed + "%";

    $("#valSmClock").textContent = msg.clock_sm_mhz + " MHz";
    $("#valMemClock").textContent = msg.clock_mem_mhz + " MHz";

    if (msg.mem_bw_util !== undefined && msg.mem_bw_util !== null) {
      setBar("barMemBw", msg.mem_bw_util);
      $("#valMemBw").textContent = msg.mem_bw_util + "%";
    }
  }

  function setBar(id, pct, colorOverride) {
    const bar = document.getElementById(id);
    if (!bar) return;
    bar.style.width = Math.min(100, Math.max(0, pct)) + "%";
    if (colorOverride) {
      bar.style.background = colorOverride;
    }
  }

  function utilColor(pct) {
    if (pct < 50) return "#00e676";
    if (pct < 80) return "#ffd740";
    return "#ff5252";
  }

  function tempColor(c) {
    if (c < 60) return "#00e676";
    if (c < 80) return "#ffd740";
    return "#ff5252";
  }

  // ── GPU Animation Data ─────────────────────────────────────────
  function handleGpuAnimation(msg) {
    // Reset animation state on algorithm switch
    const algo = msg.algorithm || currentAlgo;
    if (algo !== prevAlgorithm) {
      naiveTimeline = [];
      specTokenHistory = [];
      chunkedTimeline = [];
      prevAlgorithm = algo;
    }
    gpuAnim = msg;
    currentAlgo = algo;
    $("#currentAlgorithmLabel").textContent =
      ALGO_NAMES[currentAlgo] || currentAlgo;
  }

  // ── Benchmarks ─────────────────────────────────────────────────
  function handleBenchmark(msg) {
    benchmarks[msg.algorithm] = {
      total_tps: msg.total_tokens_per_sec,
      per_user: msg.per_user_tps,
    };
    renderBenchmarks();
  }

  function renderBenchmarks() {
    const list = $("#benchmarkList");
    // Sort by total tps descending
    const sorted = Object.entries(benchmarks).sort(
      (a, b) => b[1].total_tps - a[1].total_tps
    );
    const maxTps = sorted.length > 0 ? sorted[0][1].total_tps : 1;

    list.innerHTML = sorted
      .map(([algo, data]) => {
        const pct = maxTps > 0 ? (data.total_tps / maxTps) * 100 : 0;
        return `<div class="bench-item">
        <span class="bench-name">${ALGO_SHORT[algo] || algo}</span>
        <div class="bench-bar-wrap"><div class="bench-bar" style="width:${pct}%"></div></div>
        <span class="bench-value">${data.total_tps.toFixed(1)}</span>
      </div>`;
      })
      .join("");
  }

  // ── Animation Reset ─────────────────────────────────────────────
  function handleAnimationReset() {
    // Reset GPU animation state to blank
    gpuAnim = {
      algorithm: currentAlgo,
      active_users: [],
      memory_layout: { model_mb: 0, kv_cache_mb: [0, 0, 0, 0], free_mb: 0 },
      compute_slots: [
        { user_id: null, active: false },
        { user_id: null, active: false },
        { user_id: null, active: false },
        { user_id: null, active: false },
      ],
    };
    // Reset algorithm-specific state
    naiveTimeline = [];
    specTokenHistory = [];
    chunkedTimeline = [];
    // Reset user TPS displays
    for (let i = 0; i < 4; i++) {
      userTps[i] = 0;
      const el = $("#tpsUser" + i);
      if (el) el.textContent = "0.0";
    }
  }

  // ── Evaluate ──────────────────────────────────────────────────
  function handleEvaluateStart() {
    // Show the standalone evaluate panel
    const panel = $("#evaluatePanel");
    const body = $("#evaluateBody");
    if (panel) panel.style.display = "flex";
    if (body) body.innerHTML = '<span class="evaluate-cursor"></span>';
    const btn = $("#evaluateBtn");
    if (btn) { btn.classList.add("loading"); btn.textContent = "Analyzing..."; }

    // Also reset results dashboard LLM body if visible
    const llmBody = $("#resultsLlmBody");
    if (llmBody) llmBody.innerHTML = '<span class="evaluate-cursor"></span>';
  }

  function handleEvaluateToken(msg) {
    // Write to evaluate panel
    const body = $("#evaluateBody");
    const cursor = body ? body.querySelector(".evaluate-cursor") : null;

    // Also write to results dashboard LLM body if visible
    const llmBody = $("#resultsLlmBody");
    const llmCursor = llmBody ? llmBody.querySelector(".evaluate-cursor") : null;

    if (msg.done) {
      if (cursor) cursor.remove();
      if (llmCursor) llmCursor.remove();
      const btn = $("#evaluateBtn");
      if (btn) { btn.classList.remove("loading"); btn.textContent = "Evaluate"; }
      return;
    }

    // Insert token into evaluate panel
    if (body) {
      const panel = $("#evaluatePanel");
      if (panel) panel.style.display = "flex";
      const t1 = document.createTextNode(msg.token);
      if (cursor) body.insertBefore(t1, cursor);
      else body.appendChild(t1);
      body.scrollTop = body.scrollHeight;
    }

    // Insert token into results dashboard LLM body
    if (llmBody && $("#resultsDashboard").style.display !== "none") {
      const t2 = document.createTextNode(msg.token);
      if (llmCursor) llmBody.insertBefore(t2, llmCursor);
      else llmBody.appendChild(t2);
      llmBody.scrollTop = llmBody.scrollHeight;
    }
  }

  // ── Benchmark Progress ─────────────────────────────────────────
  function handleBenchmarkProgress(msg) {
    const overlay = $("#benchmarkOverlay");
    overlay.style.display = "flex";

    const phase = msg.phase;
    const algoName = ALGO_NAMES[msg.algorithm] || msg.algorithm || "";
    const algoIdx = msg.algo_idx || 0;
    const totalAlgos = msg.total_algos || 7;
    const run = msg.run || 0;
    const totalRuns = msg.total_runs || 5;

    const title = $("#benchOverlayTitle");
    const algoLabel = $("#benchAlgoLabel");
    const runLabel = $("#benchRunLabel");
    const bar = $("#benchProgressBar");
    const detail = $("#benchProgressDetail");

    // Overall progress: (algo_idx * RUNS + run) / (totalAlgos * totalRuns)
    const overallPct = ((algoIdx * totalRuns + run) / (totalAlgos * totalRuns)) * 100;
    bar.style.width = overallPct.toFixed(1) + "%";

    if (phase === "starting") {
      title.textContent = "Benchmarking...";
      algoLabel.textContent = algoName;
      runLabel.textContent = "Starting...";
      detail.textContent = "Algorithm " + (algoIdx + 1) + " of " + totalAlgos;
    } else if (phase === "running") {
      title.textContent = "Benchmarking...";
      algoLabel.textContent = algoName;
      runLabel.textContent = "Run " + run + "/" + totalRuns;
      detail.textContent = "Algorithm " + (algoIdx + 1) + "/" + totalAlgos + " \u2014 Run " + run + "/" + totalRuns;

      // Also highlight the algo button
      $$(".algo-btn").forEach(b => b.classList.remove("active"));
      const activeBtn = document.querySelector('.algo-btn[data-algo="' + msg.algorithm + '"]');
      if (activeBtn) activeBtn.classList.add("active");
      currentAlgo = msg.algorithm;
      $("#currentAlgorithmLabel").textContent = algoName;

    } else if (phase === "cooldown") {
      const countdown = msg.countdown || 0;
      title.textContent = "Cooldown...";
      algoLabel.textContent = "Next algorithm in:";
      runLabel.textContent = countdown + "s";
      detail.textContent = algoName + " complete. Cooling down before next algorithm.";
    } else if (phase === "aborted") {
      overlay.style.display = "none";
      benchmarkAllRunning = false;
      const btn = $("#benchmarkAllBtn");
      if (btn) { btn.textContent = "Benchmark All"; btn.classList.remove("running"); }
    }
  }

  function handleBenchmarkComplete(msg) {
    // Hide progress overlay
    $("#benchmarkOverlay").style.display = "none";
    benchmarkAllRunning = false;
    const btn = $("#benchmarkAllBtn");
    if (btn) { btn.textContent = "Benchmark All"; btn.classList.remove("running"); }

    benchmarkResults = msg.results;
    showResultsDashboard(msg.results);
  }

  function handleReportSaved() {
    // Show the "View Report" button
    const btn = $("#reportBtn");
    if (btn) btn.classList.add("visible");
  }

  function showResultsDashboard(results) {
    const dash = $("#resultsDashboard");
    dash.style.display = "flex";

    // Build table
    const tbody = $("#resultsTableBody");
    const entries = Object.entries(results).sort((a, b) => b[1].avg_tps - a[1].avg_tps);
    const bestAlgo = entries.length > 0 ? entries[0][0] : "";

    tbody.innerHTML = entries.map(([algo, d]) => {
      const isWinner = algo === bestAlgo;
      return '<tr class="' + (isWinner ? "winner" : "") + '">' +
        '<td>' + (ALGO_NAMES[algo] || algo) + (isWinner ? " \u2605" : "") + '</td>' +
        '<td>' + d.avg_tps.toFixed(2) + '</td>' +
        '<td>' + d.avg_gpu_util.toFixed(1) + '</td>' +
        '<td>' + Math.round(d.avg_mem_used_mb) + '</td>' +
        '<td>' + d.avg_power_w.toFixed(1) + '</td>' +
        '<td>' + d.avg_temp_c.toFixed(1) + '</td>' +
        '<td>' + d.num_runs + '</td>' +
        '</tr>';
    }).join("");

    // Draw bar chart on canvas
    drawResultsChart(results);

    // Clear LLM body (will be filled by evaluate_token messages)
    const llmBody = $("#resultsLlmBody");
    llmBody.innerHTML = '<span class="evaluate-cursor"></span>';
  }

  function drawResultsChart(results) {
    const canvas = $("#resultsChart");
    if (!canvas) return;
    const c = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    c.clearRect(0, 0, W, H);

    const entries = Object.entries(results).sort((a, b) => b[1].avg_tps - a[1].avg_tps);
    if (!entries.length) return;

    const maxTps = entries[0][1].avg_tps || 1;
    const barH = Math.min(24, (H - 30) / entries.length - 4);
    const labelW = 130;
    const chartW = W - labelW - 60;

    c.fillStyle = "#888";
    c.font = "bold 11px 'Segoe UI', sans-serif";
    c.textAlign = "left";
    c.fillText("Throughput (tok/s)", labelW, 14);

    const algoColors = {
      naive: "#FF6B6B", static_batch: "#4ECDC4", continuous_batch: "#45B7D1",
      paged_attention: "#96CEB4", quantized: "#6c63ff", speculative: "#ffd740",
      chunked_prefill: "#ff8a65",
    };

    for (let i = 0; i < entries.length; i++) {
      const [algo, d] = entries[i];
      const y = 24 + i * (barH + 6);
      const barW = Math.max(4, (d.avg_tps / maxTps) * chartW);
      const color = algoColors[algo] || "#888";

      // Label
      c.fillStyle = "#ccc";
      c.font = "11px 'Cascadia Code', monospace";
      c.textAlign = "right";
      c.fillText(ALGO_NAMES[algo] || algo, labelW - 8, y + barH / 2 + 4);

      // Bar
      c.fillStyle = color;
      c.globalAlpha = 0.8;
      c.beginPath();
      c.roundRect(labelW, y, barW, barH, 4);
      c.fill();
      c.globalAlpha = 1;

      // Value label
      c.fillStyle = "#fff";
      c.font = "bold 10px monospace";
      c.textAlign = "left";
      c.fillText(d.avg_tps.toFixed(1), labelW + barW + 6, y + barH / 2 + 4);
    }
  }

  // ── Controls ───────────────────────────────────────────────────
  function initControls() {
    // Algorithm buttons
    $$(".algo-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".algo-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentAlgo = btn.dataset.algo;
        $("#currentAlgorithmLabel").textContent =
          ALGO_NAMES[currentAlgo] || currentAlgo;
        wsSend({ type: "select_algorithm", algorithm: currentAlgo });
        updateStatus(
          "Algorithm: " + (ALGO_NAMES[currentAlgo] || currentAlgo),
          "connected"
        );
      });
    });

    // Auto-start/stop
    const autoBtn = $("#autoBtn");
    autoBtn.addEventListener("click", () => {
      autoRunning = !autoRunning;
      if (autoRunning) {
        autoBtn.textContent = "Stop";
        autoBtn.classList.add("running");
        autoBtn.classList.remove("stopped");
        wsSend({ type: "start_auto" });
      } else {
        autoBtn.textContent = "Auto-Start";
        autoBtn.classList.remove("running");
        autoBtn.classList.add("stopped");
        wsSend({ type: "stop_auto" });
      }
    });

    // Terminal manual inputs
    termInputs.forEach((input) => {
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && input.value.trim()) {
          const uid = parseInt(input.dataset.user, 10);
          wsSend({
            type: "manual_prompt",
            user_id: uid,
            prompt: input.value.trim(),
          });
          input.value = "";
        }
      });
    });

    // Evaluate button
    const evalBtn = $("#evaluateBtn");
    if (evalBtn) {
      evalBtn.addEventListener("click", () => {
        wsSend({ type: "evaluate" });
      });
    }

    // Evaluate close button
    const evalClose = $("#evaluateClose");
    if (evalClose) {
      evalClose.addEventListener("click", () => {
        const panel = $("#evaluatePanel");
        if (panel) panel.style.display = "none";
      });
    }

    // Benchmark All button
    const benchBtn = $("#benchmarkAllBtn");
    if (benchBtn) {
      benchBtn.addEventListener("click", () => {
        if (benchmarkAllRunning) {
          // Abort
          wsSend({ type: "stop_benchmark_all" });
          benchmarkAllRunning = false;
          benchBtn.textContent = "Benchmark All";
          benchBtn.classList.remove("running");
        } else {
          // If single-algo auto is running, stop it first
          if (autoRunning) {
            autoRunning = false;
            const ab = $("#autoBtn");
            ab.textContent = "Auto-Start";
            ab.classList.remove("running");
            ab.classList.add("stopped");
            wsSend({ type: "stop_auto" });
          }
          benchmarkAllRunning = true;
          benchBtn.textContent = "Abort";
          benchBtn.classList.add("running");
          wsSend({ type: "start_benchmark_all" });
        }
      });
    }

    // Benchmark abort button (in overlay)
    const abortBtn = $("#benchAbortBtn");
    if (abortBtn) {
      abortBtn.addEventListener("click", () => {
        wsSend({ type: "stop_benchmark_all" });
        benchmarkAllRunning = false;
        const b = $("#benchmarkAllBtn");
        if (b) { b.textContent = "Benchmark All"; b.classList.remove("running"); }
        $("#benchmarkOverlay").style.display = "none";
      });
    }

    // Results close button
    const resultsClose = $("#resultsClose");
    if (resultsClose) {
      resultsClose.addEventListener("click", () => {
        $("#resultsDashboard").style.display = "none";
      });
    }

    // Records button
    const recordsBtn = $("#recordsBtn");
    const recordsDrop = $("#recordsDropdown");
    if (recordsBtn && recordsDrop) {
      recordsBtn.addEventListener("click", async () => {
        const isOpen = recordsDrop.classList.contains("open");
        if (isOpen) {
          recordsDrop.classList.remove("open");
          return;
        }
        // Fetch report list
        try {
          const resp = await fetch("/report/list");
          const data = await resp.json();
          if (data.reports && data.reports.length > 0) {
            recordsDrop.innerHTML = data.reports.map(r => {
              const label = r.name.replace("report_", "").replace(/_/g, " ");
              return '<a href="' + r.url + '" target="_blank">' + label + '</a>';
            }).join("");
          } else {
            recordsDrop.innerHTML = '<div class="no-records">No saved reports yet</div>';
          }
        } catch (e) {
          recordsDrop.innerHTML = '<div class="no-records">Failed to load reports</div>';
        }
        recordsDrop.classList.add("open");
      });
      // Close dropdown on click outside
      document.addEventListener("click", (e) => {
        if (!recordsBtn.contains(e.target) && !recordsDrop.contains(e.target)) {
          recordsDrop.classList.remove("open");
        }
      });
    }

    // Theme toggle
    const themeBtn = $("#themeToggle");
    if (themeBtn) {
      // Restore saved theme
      if (localStorage.getItem("theme") === "light") {
        document.body.classList.add("light");
        themeBtn.innerHTML = "&#9788;"; // sun
      }
      themeBtn.addEventListener("click", () => {
        document.body.classList.toggle("light");
        const isLight = document.body.classList.contains("light");
        themeBtn.innerHTML = isLight ? "&#9788;" : "&#9790;"; // sun : moon
        localStorage.setItem("theme", isLight ? "light" : "dark");
      });
    }
  }

  // ── Particle System ────────────────────────────────────────────
  function spawnOutputParticle(uid) {
    // Spawn from GPU center moving right toward user terminal area
    const cw = canvas.width;
    const ch = canvas.height;
    const gpuRight = cw * 0.52 + 90;
    const yPositions = [ch * 0.18, ch * 0.38, ch * 0.58, ch * 0.78];
    const targetY = yPositions[uid] || ch * 0.5;

    particles.push({
      x: gpuRight,
      y: ch * 0.48,
      vx: 2 + Math.random() * 2,
      vy: (targetY - ch * 0.48) * 0.02 + (Math.random() - 0.5) * 0.5,
      color: USER_COLORS[uid],
      life: 60,
      maxLife: 60,
      radius: 3,
    });
  }

  function updateParticles() {
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.life--;
      if (p.life <= 0 || p.x > canvas.width + 10) {
        particles.splice(i, 1);
      }
    }
  }

  function drawParticles() {
    for (const p of particles) {
      const alpha = Math.max(0, p.life / p.maxLife);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // ── Canvas: GPU Visualization ──────────────────────────────────
  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const headerH =
      canvas.parentElement.querySelector(".panel-header")?.offsetHeight || 0;
    canvas.width = rect.width;
    canvas.height = rect.height - headerH;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height - headerH + "px";
  }

  function drawGPU() {
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);
    animFrame++;

    const gpuX = w * 0.28;
    const gpuY = h * 0.12;
    const gpuW = w * 0.44;
    const gpuH = h * 0.76;

    // Background grid pattern
    drawGrid(w, h);

    // Draw user arrows flowing into GPU (left side)
    drawInputArrows(gpuX, gpuY, gpuH);

    // GPU Box
    drawGPUBox(gpuX, gpuY, gpuW, gpuH);

    // Inside GPU: Memory bar
    drawMemoryBar(gpuX + 14, gpuY + 36, gpuW - 28, 22);

    // Inside GPU: Compute slots
    drawComputeSlots(gpuX + 14, gpuY + 72, gpuW - 28, 26);

    // Inside GPU: Request queue
    drawRequestQueue(gpuX + 14, gpuY + 112, gpuW - 28, 20);

    // Algorithm-specific visuals
    drawAlgorithmSpecific(gpuX, gpuY, gpuW, gpuH);

    // Output particles
    updateParticles();
    drawParticles();

    // User labels on left
    drawUserLabels(gpuX, gpuY, gpuH);

    requestAnimationFrame(drawGPU);
  }

  function drawGrid(w, h) {
    ctx.strokeStyle = "rgba(42, 42, 74, 0.3)";
    ctx.lineWidth = 0.5;
    const step = 30;
    for (let x = 0; x < w; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  }

  function drawUserLabels(gpuX, gpuY, gpuH) {
    const spacing = gpuH / 5;
    for (let i = 0; i < 4; i++) {
      const y = gpuY + spacing * (i + 0.75);
      ctx.fillStyle = USER_COLORS[i];
      ctx.font = "bold 11px 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("User " + i, 10, y + 4);
    }
  }

  function drawInputArrows(gpuX, gpuY, gpuH) {
    const spacing = gpuH / 5;
    const algo = gpuAnim.algorithm || currentAlgo;

    for (let i = 0; i < 4; i++) {
      const au = gpuAnim.active_users.find((u) => u.user_id === i);
      const isActive =
        au && (au.phase === "prefill" || au.phase === "decode");
      const y = gpuY + spacing * (i + 0.75);
      const startX = 58;
      const endX = gpuX - 4;

      // Arrow line
      const alpha = isActive ? 0.9 : 0.15;
      ctx.strokeStyle = USER_COLORS[i];
      ctx.globalAlpha = alpha;
      ctx.lineWidth = isActive ? 3 : 1.5;
      ctx.setLineDash(isActive ? [] : [4, 4]);
      ctx.beginPath();
      ctx.moveTo(startX, y);
      ctx.lineTo(endX, y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Arrowhead
      if (isActive) {
        ctx.fillStyle = USER_COLORS[i];
        ctx.beginPath();
        ctx.moveTo(endX, y);
        ctx.lineTo(endX - 8, y - 4);
        ctx.lineTo(endX - 8, y + 4);
        ctx.closePath();
        ctx.fill();

        // Animated flow dots
        const t = (animFrame * 3 + i * 20) % 80;
        const dotX = startX + ((endX - startX) * t) / 80;
        ctx.beginPath();
        ctx.arc(dotX, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.globalAlpha = 1;

      // For static batch: show padding indicator
      if (algo === "static_batch" && isActive) {
        ctx.fillStyle = "rgba(100,100,100,0.4)";
        ctx.fillRect(endX - 28, y - 3, 22, 6);
        ctx.fillStyle = "#888";
        ctx.font = "8px monospace";
        ctx.textAlign = "center";
        ctx.fillText("pad", endX - 17, y + 3);
      }
    }
  }

  function drawGPUBox(x, y, w, h) {
    // Outer glow
    ctx.shadowColor = "rgba(108, 99, 255, 0.3)";
    ctx.shadowBlur = 15;
    ctx.fillStyle = "#1a1a35";
    roundRect(ctx, x, y, w, h, 10);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Border
    ctx.strokeStyle = "#6c63ff";
    ctx.lineWidth = 2;
    roundRect(ctx, x, y, w, h, 10);
    ctx.stroke();

    // Title
    ctx.fillStyle = "#8888aa";
    ctx.font = "bold 11px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("GPU", x + 14, y + 24);

    // Algorithm indicator (small)
    ctx.fillStyle = "#6c63ff";
    ctx.font = "10px 'Segoe UI', sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(
      ALGO_NAMES[gpuAnim.algorithm || currentAlgo] || "",
      x + w - 14,
      y + 24
    );
  }

  function drawMemoryBar(x, y, w, h) {
    const mem = gpuAnim.memory_layout;
    const totalMb =
      mem.model_mb +
      mem.kv_cache_mb.reduce((a, b) => a + b, 0) +
      mem.free_mb;
    if (totalMb === 0) {
      // Draw empty bar
      ctx.fillStyle = "#111";
      roundRect(ctx, x, y, w, h, 4);
      ctx.fill();
      ctx.fillStyle = "#555";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Memory", x + w / 2, y + h / 2 + 3);
      return;
    }

    const scale = w / totalMb;
    let cx = x;

    // Model weights
    const modelW = mem.model_mb * scale;
    ctx.fillStyle = "#3a3a5a";
    roundRect(ctx, cx, y, modelW, h, cx === x ? 4 : 0);
    ctx.fill();
    if (modelW > 40) {
      ctx.fillStyle = "#aaa";
      ctx.font = "8px monospace";
      ctx.textAlign = "center";
      ctx.fillText(
        "Model " + mem.model_mb + "MB",
        cx + modelW / 2,
        y + h / 2 + 3
      );
    }
    cx += modelW;

    // KV cache per user
    for (let i = 0; i < 4; i++) {
      const kvW = mem.kv_cache_mb[i] * scale;
      if (kvW > 0) {
        ctx.fillStyle = USER_COLORS[i];
        ctx.globalAlpha = 0.7;
        ctx.fillRect(cx, y, kvW, h);
        ctx.globalAlpha = 1;
        if (kvW > 20) {
          ctx.fillStyle = "#fff";
          ctx.font = "7px monospace";
          ctx.textAlign = "center";
          ctx.fillText("U" + i, cx + kvW / 2, y + h / 2 + 3);
        }
        cx += kvW;
      }
    }

    // Free memory
    const freeW = Math.max(0, x + w - cx);
    if (freeW > 0) {
      ctx.fillStyle = "#0a0a1e";
      ctx.fillRect(cx, y, freeW, h);
      if (freeW > 30) {
        ctx.fillStyle = "#555";
        ctx.font = "8px monospace";
        ctx.textAlign = "center";
        ctx.fillText("Free " + mem.free_mb + "MB", cx + freeW / 2, y + h / 2 + 3);
      }
    }

    // Border
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 1;
    roundRect(ctx, x, y, w, h, 4);
    ctx.stroke();

    // Label
    ctx.fillStyle = "#666";
    ctx.font = "8px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("VRAM", x, y - 3);
  }

  function drawComputeSlots(x, y, w, h) {
    const slotW = (w - 12) / 4;
    const slots = gpuAnim.compute_slots;

    // Label
    ctx.fillStyle = "#666";
    ctx.font = "8px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Compute Slots", x, y - 3);

    for (let i = 0; i < 4; i++) {
      const sx = x + i * (slotW + 4);
      const slot = slots[i];
      const active = slot && slot.active;
      const uid = slot ? slot.user_id : null;

      // Slot background
      if (active && uid !== null && uid !== undefined) {
        // Pulsing glow
        const pulse =
          0.6 + 0.4 * Math.sin((animFrame * 0.08 + i * 0.5) * Math.PI);
        ctx.globalAlpha = pulse;
        ctx.fillStyle = USER_COLORS[uid];
        roundRect(ctx, sx, y, slotW, h, 4);
        ctx.fill();
        ctx.globalAlpha = 1;

        // Glow shadow
        ctx.shadowColor = USER_COLORS[uid];
        ctx.shadowBlur = 8;
        ctx.strokeStyle = USER_COLORS[uid];
        ctx.lineWidth = 1.5;
        roundRect(ctx, sx, y, slotW, h, 4);
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Label
        ctx.fillStyle = "#fff";
        ctx.font = "bold 9px monospace";
        ctx.textAlign = "center";
        ctx.fillText("U" + uid, sx + slotW / 2, y + h / 2 + 3);
      } else {
        ctx.fillStyle = "#1a1a2e";
        roundRect(ctx, sx, y, slotW, h, 4);
        ctx.fill();
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        roundRect(ctx, sx, y, slotW, h, 4);
        ctx.stroke();
        ctx.fillStyle = "#444";
        ctx.font = "8px monospace";
        ctx.textAlign = "center";
        ctx.fillText("--", sx + slotW / 2, y + h / 2 + 3);
      }
    }
  }

  function drawRequestQueue(x, y, w, h) {
    // Label
    ctx.fillStyle = "#666";
    ctx.font = "8px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Request Queue", x, y - 3);

    // Background
    ctx.fillStyle = "#111";
    roundRect(ctx, x, y, w, h, 4);
    ctx.fill();
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    roundRect(ctx, x, y, w, h, 4);
    ctx.stroke();

    // Waiting requests as dots
    const waiting = gpuAnim.active_users.filter(
      (u) => u.phase === "waiting"
    );
    let dx = x + 8;
    for (const u of waiting) {
      ctx.fillStyle = USER_COLORS[u.user_id] || "#888";
      ctx.beginPath();
      ctx.arc(dx, y + h / 2, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "7px monospace";
      ctx.textAlign = "center";
      ctx.fillText("" + u.user_id, dx, y + h / 2 + 3);
      dx += 16;
    }

    if (waiting.length === 0) {
      ctx.fillStyle = "#444";
      ctx.font = "8px monospace";
      ctx.textAlign = "center";
      ctx.fillText("empty", x + w / 2, y + h / 2 + 3);
    }
  }

  function drawAlgorithmSpecific(gpuX, gpuY, gpuW, gpuH) {
    const algo = gpuAnim.algorithm || currentAlgo;
    const innerY = gpuY + 140;
    const innerX = gpuX + 14;
    const innerW = gpuW - 28;
    const innerH = gpuH - 155;

    if (innerH < 20) return;

    switch (algo) {
      case "paged_attention":
        drawPagedBlocks(innerX, innerY, innerW, innerH);
        break;
      case "speculative":
        drawSpeculativeCycle(innerX, innerY, innerW, innerH);
        break;
      case "chunked_prefill":
        drawChunkedPrefill(innerX, innerY, innerW, innerH);
        break;
      case "quantized":
        drawQuantizedIndicator(innerX, innerY, innerW, innerH);
        break;
      case "naive":
        drawNaiveIndicator(innerX, innerY, innerW, innerH);
        break;
      case "static_batch":
        drawStaticBatchIndicator(innerX, innerY, innerW, innerH);
        break;
      case "continuous_batch":
        drawContinuousBatchIndicator(innerX, innerY, innerW, innerH);
        break;
    }
  }

  // ── Helper: draw caption strip at bottom of algo area ──
  function drawCaption(x, y, w, text) {
    ctx.fillStyle = "#888";
    ctx.font = "italic 9px 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(text, x + w / 2, y);
  }

  // ── 1. NAIVE: Gantt Chart ──────────────────────────────────────
  function drawNaiveIndicator(x, y, w, h) {
    const activeUser = gpuAnim.active_users.find(
      (u) => u.phase === "prefill" || u.phase === "decode"
    );
    const uid = activeUser ? activeUser.user_id : null;
    const totalUsers = gpuAnim.total_users || 4;
    const idleSlots = gpuAnim.idle_slots || (uid !== null ? totalUsers - 1 : 0);

    // Title
    ctx.fillStyle = "#888";
    ctx.font = "bold 9px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Gantt Chart \u2014 One User at a Time", x, y + 8);

    // GPU Utilization badge
    const utilPct = uid !== null ? Math.round(100 / Math.max(totalUsers, 1)) : 0;
    ctx.fillStyle = utilPct < 50 ? "#ff4444" : "#00e676";
    ctx.font = "bold 10px monospace";
    ctx.textAlign = "right";
    ctx.fillText("GPU Util: " + utilPct + "%", x + w, y + 8);

    // Gantt lanes
    const laneH = Math.min(18, (h - 30) / 4);
    const laneGap = 3;
    const laneX = x + 24;
    const laneW = w - 28;

    for (let i = 0; i < 4; i++) {
      const ly = y + 16 + i * (laneH + laneGap);
      const isActive = uid === i;
      const au = gpuAnim.active_users.find((u) => u.user_id === i);
      const tokens = au ? au.tokens_generated || 0 : 0;

      // Lane label
      ctx.fillStyle = USER_COLORS[i];
      ctx.globalAlpha = isActive ? 1 : 0.4;
      ctx.font = "bold 8px monospace";
      ctx.textAlign = "right";
      ctx.fillText("U" + i, laneX - 4, ly + laneH / 2 + 3);

      // Lane background
      ctx.fillStyle = "#1a1a2e";
      ctx.globalAlpha = 0.6;
      roundRect(ctx, laneX, ly, laneW, laneH, 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      if (isActive) {
        // Active bar growing based on tokens
        const barW = Math.min(laneW, Math.max(8, (tokens / 128) * laneW));
        ctx.fillStyle = USER_COLORS[i];
        ctx.globalAlpha = 0.8 + 0.15 * Math.sin(animFrame * 0.1);
        roundRect(ctx, laneX, ly, barW, laneH, 2);
        ctx.fill();
        ctx.globalAlpha = 1;

        // Active pulse edge
        ctx.fillStyle = "#fff";
        ctx.globalAlpha = 0.3 + 0.3 * Math.sin(animFrame * 0.15);
        ctx.fillRect(laneX + barW - 3, ly, 3, laneH);
        ctx.globalAlpha = 1;
      } else if (au && au.phase === "idle" && tokens > 0) {
        // Completed: show full bar dimmed
        const barW = Math.min(laneW, Math.max(8, (tokens / 128) * laneW));
        ctx.fillStyle = USER_COLORS[i];
        ctx.globalAlpha = 0.3;
        roundRect(ctx, laneX, ly, barW, laneH, 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#00e676";
        ctx.font = "bold 7px monospace";
        ctx.textAlign = "center";
        ctx.fillText("\u2713", laneX + barW - 8, ly + laneH / 2 + 3);
      } else {
        // Idle: hatched pattern
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 0.5;
        for (let sx = laneX + 4; sx < laneX + laneW; sx += 10) {
          ctx.beginPath();
          ctx.moveTo(sx, ly);
          ctx.lineTo(sx - 4, ly + laneH);
          ctx.stroke();
        }
        ctx.fillStyle = "#555";
        ctx.font = "7px monospace";
        ctx.textAlign = "center";
        ctx.fillText("IDLE", laneX + laneW / 2, ly + laneH / 2 + 2);
      }
    }

    // Caption
    if (uid !== null) {
      drawCaption(x, y + h - 2, w,
        "User " + uid + " processing. " + idleSlots + " slot(s) idle. GPU " + (100 - utilPct) + "% wasted.");
    } else {
      drawCaption(x, y + h - 2, w, "Waiting for prompt. Only 1 user processed at a time.");
    }
  }

  // ── 2. STATIC BATCH: Padded Bars with Waste Counter ────────────
  function drawStaticBatchIndicator(x, y, w, h) {
    const paddedLen = gpuAnim.padded_length || 0;
    const perUserActual = gpuAnim.per_user_actual_tokens || [];
    const eosReached = gpuAnim.eos_reached || [];
    const activeUids = gpuAnim.active_uids || [0, 1, 2, 3];

    // Title
    ctx.fillStyle = "#888";
    ctx.font = "bold 9px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Static Batch \u2014 Padded to Max Length", x, y + 8);

    // Waste percentage
    let totalWaste = 0, totalCompute = 0;
    if (paddedLen > 0 && perUserActual.length > 0) {
      for (let i = 0; i < perUserActual.length; i++) {
        if (eosReached[i]) {
          totalWaste += paddedLen - perUserActual[i];
        }
        totalCompute += paddedLen;
      }
    }
    const wastePct = totalCompute > 0 ? Math.round((totalWaste / totalCompute) * 100) : 0;
    if (wastePct > 0) {
      ctx.fillStyle = "#ff4444";
      ctx.font = "bold 10px monospace";
      ctx.textAlign = "right";
      ctx.fillText("Waste: " + wastePct + "%", x + w, y + 8);
    }

    const barH = Math.min(14, (h - 35) / activeUids.length);
    const gap = 3;
    const maxBarW = w * 0.82;

    for (let idx = 0; idx < activeUids.length; idx++) {
      const uid = activeUids[idx];
      const by = y + 16 + idx * (barH + gap);
      const au = gpuAnim.active_users.find((u) => u.user_id === uid);
      const tokens = au ? au.tokens_generated || 0 : 0;
      const isDone = eosReached[idx] || false;
      const actualTokens = perUserActual[idx] || tokens;

      // Label
      ctx.fillStyle = USER_COLORS[uid] || "#888";
      ctx.font = "bold 8px monospace";
      ctx.textAlign = "right";
      ctx.fillText("U" + uid, x + 16, by + barH / 2 + 3);

      // Full padded bar background
      ctx.fillStyle = "#1a1a2e";
      ctx.globalAlpha = 0.6;
      roundRect(ctx, x + 20, by, maxBarW, barH, 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      // Content portion (actual tokens)
      const contentRatio = paddedLen > 0 ? Math.min(1, actualTokens / paddedLen) : (tokens > 0 ? 0.5 : 0.1);
      const contentW = Math.max(4, maxBarW * contentRatio);

      ctx.fillStyle = isDone ? "rgba(100,100,100,0.5)" : (USER_COLORS[uid] || "#888");
      ctx.globalAlpha = isDone ? 0.4 : 0.8;
      roundRect(ctx, x + 20, by, contentW, barH, 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      // Padding waste portion (red striped)
      if (isDone && paddedLen > 0) {
        const padX = x + 20 + contentW;
        const padW = maxBarW - contentW;
        if (padW > 2) {
          ctx.fillStyle = "rgba(255,60,60,0.25)";
          ctx.fillRect(padX, by, padW, barH);

          // Red stripes
          ctx.strokeStyle = "rgba(255,60,60,0.5)";
          ctx.lineWidth = 0.7;
          for (let sx = padX + 4; sx < padX + padW; sx += 7) {
            ctx.beginPath();
            ctx.moveTo(sx, by);
            ctx.lineTo(sx - 3, by + barH);
            ctx.stroke();
          }

          // WASTED label
          if (padW > 35) {
            ctx.fillStyle = "#ff4444";
            ctx.font = "bold 7px monospace";
            ctx.textAlign = "center";
            ctx.fillText("WASTED", padX + padW / 2, by + barH / 2 + 2);
          }
        }
      }

      // Border
      ctx.strokeStyle = isDone ? "#ff4444" : "#444";
      ctx.lineWidth = isDone ? 1 : 0.5;
      roundRect(ctx, x + 20, by, maxBarW, barH, 2);
      ctx.stroke();
    }

    // Caption
    const doneCount = eosReached.filter(Boolean).length;
    if (doneCount > 0 && doneCount < activeUids.length) {
      drawCaption(x, y + h - 2, w,
        doneCount + " user(s) done but batch still running. Compute wasted on padding.");
    } else if (doneCount === 0 && paddedLen > 0) {
      drawCaption(x, y + h - 2, w,
        "All users padded to " + paddedLen + " tokens. Shorter sequences waste compute.");
    } else {
      drawCaption(x, y + h - 2, w, "All users batched together. Everyone waits for the slowest.");
    }
  }

  // ── 3. CONTINUOUS BATCH: Independent Lanes + Slot Recycling ────
  function drawContinuousBatchIndicator(x, y, w, h) {
    const slotStatus = gpuAnim.slot_status || ["idle", "idle", "idle", "idle"];

    // Title
    ctx.fillStyle = "#888";
    ctx.font = "bold 9px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Continuous Batching \u2014 Independent Completion", x, y + 8);

    const barH = Math.min(16, (h - 30) / 4);
    const gap = 3;
    const maxBarW = w * 0.68;

    let activeCount = 0, freeCount = 0;

    for (let i = 0; i < 4; i++) {
      const by = y + 16 + i * (barH + gap);
      const au = gpuAnim.active_users.find((u) => u.user_id === i);
      const isActive = au && (au.phase === "prefill" || au.phase === "decode");
      const tokens = au ? au.tokens_generated || 0 : 0;
      const status = slotStatus[i] || "idle";

      // Label
      ctx.fillStyle = USER_COLORS[i];
      ctx.globalAlpha = isActive ? 1 : 0.4;
      ctx.font = "bold 8px monospace";
      ctx.textAlign = "right";
      ctx.fillText("U" + i, x + 16, by + barH / 2 + 3);
      ctx.globalAlpha = 1;

      if (isActive) {
        activeCount++;
        // Active bar: independent length, no padding
        const barW = Math.min(maxBarW, Math.max(8, (tokens / 128) * maxBarW));
        ctx.fillStyle = USER_COLORS[i];
        ctx.globalAlpha = 0.8;
        roundRect(ctx, x + 20, by, barW, barH, 3);
        ctx.fill();
        ctx.globalAlpha = 1;

        // Growing edge
        ctx.fillStyle = "#fff";
        ctx.globalAlpha = 0.2 + 0.3 * Math.sin(animFrame * 0.12 + i);
        ctx.fillRect(x + 20 + barW - 4, by, 4, barH);
        ctx.globalAlpha = 1;

        // Status label
        ctx.fillStyle = USER_COLORS[i];
        ctx.font = "8px monospace";
        ctx.textAlign = "left";
        ctx.fillText(tokens + " tok", x + 24 + maxBarW, by + barH / 2 + 3);

      } else if (status === "free") {
        freeCount++;
        // Free slot: green dashed border with "SLOT FREE"
        ctx.strokeStyle = "#00e676";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.globalAlpha = 0.5 + 0.3 * Math.sin(animFrame * 0.08 + i);
        roundRect(ctx, x + 20, by, maxBarW, barH, 3);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;

        ctx.fillStyle = "#00e676";
        ctx.font = "bold 8px monospace";
        ctx.textAlign = "center";
        ctx.fillText("\u2713 SLOT FREE", x + 20 + maxBarW / 2, by + barH / 2 + 3);

      } else {
        // Idle: empty slot
        ctx.fillStyle = "#1a1a2e";
        ctx.globalAlpha = 0.4;
        roundRect(ctx, x + 20, by, maxBarW, barH, 3);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#444";
        ctx.font = "7px monospace";
        ctx.textAlign = "center";
        ctx.fillText("empty", x + 20 + maxBarW / 2, by + barH / 2 + 2);
      }
    }

    // Caption
    if (freeCount > 0) {
      drawCaption(x, y + h - 2, w,
        freeCount + " slot(s) freed instantly. No waiting for other users. No padding waste.");
    } else if (activeCount > 0) {
      drawCaption(x, y + h - 2, w,
        activeCount + " user(s) active. Each finishes independently \u2014 no padding needed.");
    } else {
      drawCaption(x, y + h - 2, w, "Users complete independently. Freed slots reused immediately.");
    }
  }

  // ── 4. PAGED ATTENTION: Physical Memory Grid + Page Table ──────
  function drawPagedBlocks(x, y, w, h) {
    const ml = gpuAnim.memory_layout || {};
    const pageMap = ml.page_map || [];
    const totalPages = ml.total_pages || 64;
    const pagesPerUser = ml.pages_per_user || [0, 0, 0, 0];
    const utilPct = ml.page_utilization || 0;
    const fragPct = ml.page_fragmentation || 0;

    // Title + metrics
    ctx.fillStyle = "#888";
    ctx.font = "bold 9px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("PagedAttention \u2014 Virtual Memory for KV Cache", x, y + 8);

    ctx.font = "8px monospace";
    ctx.textAlign = "right";
    ctx.fillStyle = "#4ECDC4";
    ctx.fillText("Util:" + utilPct.toFixed(0) + "%  Frag:" + fragPct.toFixed(0) + "%", x + w, y + 8);

    // Physical memory grid
    const gridY = y + 14;
    const blockSize = 10;
    const gapB = 2;
    const displayPages = Math.min(totalPages, 80);
    const cols = Math.floor(w / (blockSize + gapB));
    const gridRows = Math.min(Math.ceil(displayPages / cols), Math.floor((h * 0.5) / (blockSize + gapB)));

    // Build page owner lookup
    const pageOwner = {};
    for (const pm of pageMap) {
      pageOwner[pm.page] = pm.user_id;
    }

    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < cols; c++) {
        const pageIdx = r * cols + c;
        if (pageIdx >= displayPages) break;
        const bx = x + c * (blockSize + gapB);
        const by = gridY + r * (blockSize + gapB);
        const owner = pageOwner[pageIdx];

        if (owner !== undefined) {
          ctx.fillStyle = USER_COLORS[owner] || "#666";
          ctx.globalAlpha = 0.75;
        } else {
          ctx.fillStyle = "#0a0a1e";
          ctx.globalAlpha = 0.5;
        }
        ctx.fillRect(bx, by, blockSize, blockSize);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "#2a2a3e";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(bx, by, blockSize, blockSize);
      }
    }

    // Page table: per-user rows showing page counts
    const tableY = gridY + gridRows * (blockSize + gapB) + 8;
    const rowH = Math.min(12, (h - (tableY - y) - 14) / 4);
    if (rowH > 4) {
      ctx.fillStyle = "#666";
      ctx.font = "7px 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("Page Table Mapping:", x, tableY - 2);

      for (let i = 0; i < 4; i++) {
        const ry = tableY + 2 + i * (rowH + 2);
        const pages = pagesPerUser[i] || 0;

        ctx.fillStyle = USER_COLORS[i];
        ctx.globalAlpha = pages > 0 ? 0.9 : 0.3;
        ctx.font = "bold 7px monospace";
        ctx.textAlign = "left";
        ctx.fillText("U" + i + ":", x, ry + rowH / 2 + 2);
        ctx.globalAlpha = 1;

        // Draw small page blocks for this user
        const maxPageDraw = Math.min(pages, 20);
        for (let p = 0; p < maxPageDraw; p++) {
          const px = x + 22 + p * (8);
          ctx.fillStyle = USER_COLORS[i];
          ctx.globalAlpha = 0.7;
          ctx.fillRect(px, ry, 6, rowH);
          ctx.globalAlpha = 1;
        }
        if (pages > maxPageDraw) {
          ctx.fillStyle = "#888";
          ctx.font = "6px monospace";
          ctx.fillText("+" + (pages - maxPageDraw), x + 22 + maxPageDraw * 8, ry + rowH / 2 + 2);
        }

        // Page count
        ctx.fillStyle = "#888";
        ctx.font = "7px monospace";
        ctx.textAlign = "right";
        ctx.fillText(pages + " pages", x + w, ry + rowH / 2 + 2);
      }
    }

    // Caption
    drawCaption(x, y + h - 2, w,
      "Non-contiguous pages \u2014 like OS virtual memory. No fragmentation overhead.");
  }

  // ── 5. SPECULATIVE: Token Sequence with Draft/Verify ───────────
  function drawSpeculativeCycle(x, y, w, h) {
    const phase = gpuAnim.cycle_phase || "";
    const draftTokens = gpuAnim.draft_tokens || [];
    const verifyResults = gpuAnim.verified_results || [];
    const acceptRate = gpuAnim.acceptance_rate || 0;
    const draftK = gpuAnim.draft_k || 4;

    // Update token history from latest data
    if (phase === "verifying" && verifyResults.length > 0 && draftTokens.length > 0) {
      // Replace any pending drafts with verified results
      // Remove old drafts
      specTokenHistory = specTokenHistory.filter(t => t.status !== "draft");
      for (let i = 0; i < draftTokens.length; i++) {
        specTokenHistory.push({
          text: draftTokens[i],
          status: verifyResults[i] || "rejected",
        });
      }
      if (specTokenHistory.length > 16) {
        specTokenHistory = specTokenHistory.slice(-16);
      }
    } else if (phase === "drafting" && draftTokens.length > 0) {
      // Show drafts as pending
      specTokenHistory = specTokenHistory.filter(t => t.status !== "draft");
      for (let i = 0; i < draftTokens.length; i++) {
        specTokenHistory.push({ text: draftTokens[i], status: "draft" });
      }
      if (specTokenHistory.length > 16) {
        specTokenHistory = specTokenHistory.slice(-16);
      }
    }

    // Title
    ctx.fillStyle = "#888";
    ctx.font = "bold 9px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Speculative Decoding \u2014 Draft K, Verify in 1 Pass", x, y + 8);

    // Phase indicator
    if (phase === "drafting") {
      ctx.fillStyle = "#ffd740";
      ctx.font = "bold 9px monospace";
      ctx.textAlign = "right";
      ctx.fillText("\u26A1 DRAFTING", x + w, y + 8);
    } else if (phase === "verifying") {
      ctx.fillStyle = "#00e676";
      ctx.font = "bold 9px monospace";
      ctx.textAlign = "right";
      ctx.fillText("\u2714 VERIFYING", x + w, y + 8);
    }

    // Token sequence display
    const tokenY = y + 16;
    const tokenW = 22;
    const tokenH = Math.min(20, (h - 45) * 0.5);
    const tokenGap = 3;
    const maxTokens = Math.min(specTokenHistory.length, Math.floor(w / (tokenW + tokenGap)));
    const startIdx = Math.max(0, specTokenHistory.length - maxTokens);

    for (let i = startIdx; i < specTokenHistory.length; i++) {
      const t = specTokenHistory[i];
      const tx = x + (i - startIdx) * (tokenW + tokenGap);

      let bgColor, borderColor, label;
      switch (t.status) {
        case "draft":
          bgColor = "rgba(255,215,64,0.3)";
          borderColor = "#ffd740";
          label = "D";
          break;
        case "accepted":
          bgColor = "rgba(0,230,118,0.25)";
          borderColor = "#00e676";
          label = "\u2713";
          break;
        case "rejected":
          bgColor = "rgba(255,60,60,0.25)";
          borderColor = "#ff4444";
          label = "\u2717";
          break;
        case "corrected":
          bgColor = "rgba(100,100,255,0.25)";
          borderColor = "#6c63ff";
          label = "\u21BB";
          break;
        default:
          bgColor = "#1a1a2e";
          borderColor = "#333";
          label = "";
      }

      ctx.fillStyle = bgColor;
      roundRect(ctx, tx, tokenY, tokenW, tokenH, 3);
      ctx.fill();
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 1.2;
      roundRect(ctx, tx, tokenY, tokenW, tokenH, 3);
      ctx.stroke();

      // Status label
      ctx.fillStyle = borderColor;
      ctx.font = "bold 9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(label, tx + tokenW / 2, tokenY + tokenH / 2 + 3);
    }

    // Acceptance rate gauge
    const gaugeY = tokenY + tokenH + 8;
    const gaugeW = w * 0.6;
    const gaugeH = 10;

    ctx.fillStyle = "#1a1a2e";
    roundRect(ctx, x, gaugeY, gaugeW, gaugeH, 3);
    ctx.fill();

    const fillW = gaugeW * (acceptRate / 100);
    ctx.fillStyle = acceptRate > 70 ? "#00e676" : acceptRate > 40 ? "#ffd740" : "#ff4444";
    ctx.globalAlpha = 0.8;
    roundRect(ctx, x, gaugeY, Math.max(2, fillW), gaugeH, 3);
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.fillStyle = "#ccc";
    ctx.font = "bold 8px monospace";
    ctx.textAlign = "left";
    ctx.fillText("Accept Rate: " + acceptRate.toFixed(0) + "%", x + gaugeW + 8, gaugeY + gaugeH / 2 + 3);

    // Caption
    drawCaption(x, y + h - 2, w,
      "Draft " + draftK + " tokens (greedy), verify in 1 pass. High accept rate = " + draftK + "x speedup.");
  }

  // ── 6. CHUNKED PREFILL: Interleaved Work Timeline ─────────────
  function drawChunkedPrefill(x, y, w, h) {
    const iterLog = gpuAnim.iteration_log || [];
    const prefillProg = gpuAnim.prefill_progress || {};

    // Update timeline from iteration log
    if (iterLog.length > 0) {
      chunkedTimeline = iterLog;
    }

    // Title
    ctx.fillStyle = "#888";
    ctx.font = "bold 9px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Chunked Prefill \u2014 Interleaved Timeline", x, y + 8);

    // Prefill progress bars (right side)
    const progX = x + w * 0.72;
    const progW = w * 0.26;
    ctx.fillStyle = "#666";
    ctx.font = "7px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Prefill Progress:", progX, y + 8);

    for (let i = 0; i < 4; i++) {
      const py = y + 14 + i * 11;
      const prog = prefillProg[i];
      ctx.fillStyle = USER_COLORS[i];
      ctx.globalAlpha = prog ? 0.9 : 0.3;
      ctx.font = "7px monospace";
      ctx.textAlign = "left";
      ctx.fillText("U" + i, progX, py + 7);
      ctx.globalAlpha = 1;

      if (prog) {
        const pct = prog.total > 0 ? prog.pos / prog.total : 0;
        ctx.fillStyle = "#1a1a2e";
        ctx.fillRect(progX + 16, py, progW - 16, 8);
        ctx.fillStyle = USER_COLORS[i];
        ctx.globalAlpha = 0.7;
        ctx.fillRect(progX + 16, py, (progW - 16) * pct, 8);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(progX + 16, py, progW - 16, 8);
      }
    }

    // Timeline (left side) - show recent work items as rows
    const tlX = x;
    const tlW = w * 0.68;
    const tlY = y + 16;
    const maxRows = Math.min(chunkedTimeline.length, Math.floor((h - 30) / 12));
    const startIdx = Math.max(0, chunkedTimeline.length - maxRows);
    const rowH = 10;
    const rowGap = 2;

    for (let i = startIdx; i < chunkedTimeline.length; i++) {
      const item = chunkedTimeline[i];
      const ry = tlY + (i - startIdx) * (rowH + rowGap);
      const uid = item.uid;
      const color = USER_COLORS[uid] || "#888";

      if (item.type === "prefill") {
        // Wider block for prefill
        const blockW = tlW * 0.6;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.7;
        roundRect(ctx, tlX, ry, blockW, rowH, 2);
        ctx.fill();
        ctx.globalAlpha = 1;

        ctx.fillStyle = "#fff";
        ctx.font = "bold 6px monospace";
        ctx.textAlign = "center";
        ctx.fillText("PREFILL U" + uid, tlX + blockW / 2, ry + rowH / 2 + 2);
      } else {
        // Narrower block for decode
        const blockW = tlW * 0.3;
        const offset = (uid % 3) * tlW * 0.22;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.6;
        roundRect(ctx, tlX + offset, ry, blockW, rowH, 2);
        ctx.fill();
        ctx.globalAlpha = 1;

        ctx.fillStyle = "#fff";
        ctx.font = "6px monospace";
        ctx.textAlign = "center";
        ctx.fillText("D:U" + uid, tlX + offset + blockW / 2, ry + rowH / 2 + 2);
      }
    }

    // Caption
    const prefillUsers = Object.keys(prefillProg).length;
    if (prefillUsers > 0) {
      drawCaption(x, y + h - 2, w,
        "Prefill split into chunks. Decode steps interleaved \u2014 no starvation.");
    } else {
      drawCaption(x, y + h - 2, w,
        "Long prompts chunked. Other users keep decoding between chunks.");
    }
  }

  // ── 7. QUANTIZED: Memory Comparison ────────────────────────────
  function drawQuantizedIndicator(x, y, w, h) {
    // Title
    ctx.fillStyle = "#888";
    ctx.font = "bold 9px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("4-bit Quantization \u2014 Memory Savings", x, y + 8);

    const barH = 16;
    const fullW = w * 0.65;
    const modelMb = gpuAnim.memory_layout ? gpuAnim.memory_layout.model_mb || 0 : 0;

    // BF16 bar (reference)
    const fp16Y = y + 20;
    ctx.fillStyle = "#3a3a5a";
    roundRect(ctx, x, fp16Y, fullW, barH, 3);
    ctx.fill();
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 0.5;
    roundRect(ctx, x, fp16Y, fullW, barH, 3);
    ctx.stroke();
    ctx.fillStyle = "#aaa";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("BF16: ~8000 MB", x + fullW / 2, fp16Y + barH / 2 + 3);

    // INT4 bar (quantized, ~1/4 size)
    const int4Y = fp16Y + barH + 8;
    const int4W = fullW * 0.28;
    ctx.fillStyle = "#6c63ff";
    ctx.globalAlpha = 0.8;
    roundRect(ctx, x, int4Y, int4W, barH, 3);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#6c63ff";
    ctx.lineWidth = 0.5;
    roundRect(ctx, x, int4Y, int4W, barH, 3);
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("INT4: ~" + (modelMb || "2000") + " MB", x + int4W / 2, int4Y + barH / 2 + 3);

    // Freed space (green)
    const freeX = x + int4W + 4;
    const freeW = fullW - int4W - 4;
    ctx.fillStyle = "rgba(0,230,118,0.15)";
    roundRect(ctx, freeX, int4Y, freeW, barH, 3);
    ctx.fill();
    ctx.strokeStyle = "#00e676";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    roundRect(ctx, freeX, int4Y, freeW, barH, 3);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#00e676";
    ctx.font = "bold 8px monospace";
    ctx.textAlign = "center";
    ctx.fillText("FREE for KV cache", freeX + freeW / 2, int4Y + barH / 2 + 3);

    // Arrow + savings label
    ctx.fillStyle = "#00e676";
    ctx.font = "bold 11px 'Segoe UI', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("\u2193 ~4x memory savings", x + fullW + 10, fp16Y + barH + 6);
    ctx.fillStyle = "#888";
    ctx.font = "9px 'Segoe UI', sans-serif";
    ctx.fillText("= more users / longer context", x + fullW + 10, fp16Y + barH + 20);

    // Caption
    drawCaption(x, y + h - 2, w,
      "4x less VRAM for model = 4x more room for KV caches and concurrent users.");
  }

  // ── Utility: Round Rect ────────────────────────────────────────
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ── Initialization ─────────────────────────────────────────────
  function init() {
    initControls();
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    renderBenchmarks();
    drawGPU();
    connect();

    // Check if a report already exists
    fetch("/report/data").then(r => {
      if (r.ok) {
        const btn = $("#reportBtn");
        if (btn) btn.classList.add("visible");
      }
    }).catch(() => {});
  }

  // Start when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
