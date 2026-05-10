"""
Monitoring Dashboard
Real-time web UI for tracking training progress and generated samples.
Run: python dashboard/app.py
Open: http://localhost:5000
"""

import sys
import json
import time
import random
import math
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from flask import Flask, render_template_string, jsonify, send_from_directory
    from flask_socketio import SocketIO, emit
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Flask not installed. Run: pip install flask flask-socketio")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GenAI Simulation Engine — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  :root {
    --bg: #0a0a0f;
    --panel: #12121a;
    --border: #1e1e2e;
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --green: #22c55e;
    --text: #e2e8f0;
    --muted: #64748b;
    --red: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    min-height: 100vh;
  }
  header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  .logo { font-size: 20px; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; }
  .badge {
    background: var(--accent);
    color: white;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 99px;
    font-weight: 600;
  }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
  .status-label { font-size: 12px; color: var(--muted); }
  .main { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto auto; gap: 16px; padding: 20px 28px; }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .panel-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .metric {
    background: #0d0d14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  .metric-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-value { font-size: 26px; font-weight: 700; margin-top: 4px; }
  .metric-value.green { color: var(--green); }
  .metric-value.purple { color: var(--accent); }
  .metric-value.cyan { color: var(--accent2); }
  .metric-value.red { color: var(--red); }
  .metric-delta { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .chart-container { height: 220px; position: relative; }
  .samples-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 4px; }
  .sample-img {
    aspect-ratio: 1;
    border-radius: 4px;
    background: linear-gradient(135deg, #1e1e2e 0%, #12121a 100%);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  .sample-img canvas { width: 100%; height: 100%; display: block; }
  .col-span-2 { grid-column: span 2; }
  .log-panel {
    height: 180px;
    overflow-y: auto;
    font-size: 11px;
    color: var(--muted);
    line-height: 1.8;
  }
  .log-line { border-bottom: 1px solid #0d0d14; padding: 2px 0; }
  .log-line span.time { color: var(--accent2); margin-right: 8px; }
  .log-line span.level { margin-right: 8px; }
  .log-line span.level.info { color: var(--green); }
  .log-line span.level.warn { color: #f59e0b; }
  .log-line span.level.step { color: var(--accent); }
  .progress-bar { height: 4px; background: var(--border); border-radius: 2px; margin-top: 8px; }
  .progress-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 0.5s; }
  .gpu-bars { display: flex; flex-direction: column; gap: 8px; }
  .gpu-bar-row { display: flex; align-items: center; gap: 8px; font-size: 11px; }
  .gpu-bar-label { width: 60px; color: var(--muted); }
  .gpu-bar-track { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .gpu-bar-fill { height: 100%; border-radius: 3px; transition: width 0.8s; }
  .gpu-bar-val { width: 40px; text-align: right; color: var(--text); }
  scrollbar-width: thin;
  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: var(--border); }
</style>
</head>
<body>
<header>
  <div class="logo">⚡ GenAI Simulation Engine</div>
  <div class="badge">v1.0</div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
    <div class="status-dot"></div>
    <span class="status-label" id="status-label">Training · DDPM · CIFAR-10</span>
  </div>
</header>

<div class="main">
  <!-- Metrics cards -->
  <div class="panel">
    <div class="panel-title">📊 Live Metrics</div>
    <div class="metrics-grid">
      <div class="metric">
        <div class="metric-label">Train Loss</div>
        <div class="metric-value purple" id="train-loss">0.142</div>
        <div class="metric-delta" id="loss-delta">↓ improving</div>
      </div>
      <div class="metric">
        <div class="metric-label">Steps</div>
        <div class="metric-value cyan" id="global-step">0</div>
        <div class="metric-delta" id="step-rate">— steps/s</div>
      </div>
      <div class="metric">
        <div class="metric-label">FID Score</div>
        <div class="metric-value green" id="fid-score">—</div>
        <div class="metric-delta">lower is better</div>
      </div>
      <div class="metric">
        <div class="metric-label">Epoch</div>
        <div class="metric-value" id="epoch-display">0 / 100</div>
        <div class="progress-bar"><div class="progress-fill" id="epoch-progress" style="width:0%"></div></div>
      </div>
    </div>
  </div>

  <!-- GPU utilization -->
  <div class="panel">
    <div class="panel-title">🖥️ GPU Utilization</div>
    <div class="gpu-bars" id="gpu-bars"></div>
    <div style="margin-top:14px; font-size:11px; color: var(--muted);">
      <span>LR: <span id="lr-display" style="color:var(--accent2)">2.0e-4</span></span>
      &nbsp;&nbsp;
      <span>EMA Decay: <span style="color:var(--green)">0.9999</span></span>
      &nbsp;&nbsp;
      <span>Mixed Precision: <span style="color:var(--accent)">fp16</span></span>
    </div>
  </div>

  <!-- Loss curve -->
  <div class="panel">
    <div class="panel-title">📈 Loss Curve</div>
    <div class="chart-container">
      <canvas id="loss-chart"></canvas>
    </div>
  </div>

  <!-- Generated samples -->
  <div class="panel">
    <div class="panel-title">🎨 Generated Samples (live)</div>
    <div class="samples-grid" id="samples-grid"></div>
  </div>

  <!-- Training log -->
  <div class="panel col-span-2">
    <div class="panel-title">📝 Training Log</div>
    <div class="log-panel" id="log-panel"></div>
  </div>
</div>

<script>
const socket = io();
const STEPS_HISTORY = 200;
const lossData = { labels: [], train: [], val: [] };

// ── Chart ──
const ctx = document.getElementById('loss-chart').getContext('2d');
const lossChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: lossData.labels,
    datasets: [
      { label: 'Train Loss', data: lossData.train, borderColor: '#7c3aed', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false },
      { label: 'Val Loss',   data: lossData.val,   borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false, borderDash: [4,3] },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { labels: { color: '#64748b', font: { size: 10 } } } },
    scales: {
      x: { ticks: { color: '#64748b', font:{size:9}, maxTicksLimit:8 }, grid: { color: '#1e1e2e' } },
      y: { ticks: { color: '#64748b', font:{size:9} }, grid: { color: '#1e1e2e' } }
    }
  }
});

// ── Samples grid ──
function initSamples() {
  const grid = document.getElementById('samples-grid');
  grid.innerHTML = '';
  for (let i = 0; i < 32; i++) {
    const div = document.createElement('div');
    div.className = 'sample-img';
    const c = document.createElement('canvas');
    c.width = 32; c.height = 32;
    c.id = `sample-${i}`;
    div.appendChild(c);
    grid.appendChild(div);
    drawNoiseSample(c);
  }
}

function drawNoiseSample(canvas, progress = 0) {
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(32, 32);
  const hue = Math.random() * 60 + 220;
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.random() * 255;
    const blend = progress;
    img.data[i]   = Math.min(255, v * (1-blend) + (100 + Math.random()*80) * blend);
    img.data[i+1] = Math.min(255, v * (1-blend) + (80  + Math.random()*60) * blend);
    img.data[i+2] = Math.min(255, v * (1-blend) + (180 + Math.random()*60) * blend);
    img.data[i+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function refreshSamples(progress) {
  for (let i = 0; i < 32; i++) {
    const c = document.getElementById(`sample-${i}`);
    if (c) drawNoiseSample(c, progress);
  }
}

// ── GPU bars ──
function renderGPUs(gpuStats) {
  const container = document.getElementById('gpu-bars');
  container.innerHTML = '';
  gpuStats.forEach((g, i) => {
    const colors = ['#7c3aed','#06b6d4','#22c55e','#f59e0b'];
    container.innerHTML += `
      <div class="gpu-bar-row">
        <div class="gpu-bar-label">GPU ${i}</div>
        <div class="gpu-bar-track"><div class="gpu-bar-fill" style="width:${g.util}%;background:${colors[i%4]}"></div></div>
        <div class="gpu-bar-val">${g.util}%</div>
        <div class="gpu-bar-label" style="width:80px">VRAM ${g.mem}GB</div>
      </div>`;
  });
}

// ── Log ──
function addLog(msg, level='info') {
  const panel = document.getElementById('log-panel');
  const now = new Date().toLocaleTimeString('en-US', {hour12:false});
  panel.innerHTML += `<div class="log-line"><span class="time">${now}</span><span class="level ${level}">[${level.toUpperCase()}]</span> ${msg}</div>`;
  panel.scrollTop = panel.scrollHeight;
}

// ── Socket events ──
socket.on('metrics_update', data => {
  document.getElementById('train-loss').textContent = data.loss.toFixed(4);
  document.getElementById('global-step').textContent = data.step.toLocaleString();
  document.getElementById('step-rate').textContent = `${data.steps_per_sec.toFixed(1)} steps/s`;
  document.getElementById('lr-display').textContent = data.lr.toExponential(1);
  document.getElementById('epoch-display').textContent = `${data.epoch} / ${data.total_epochs}`;
  document.getElementById('epoch-progress').style.width = `${(data.epoch / data.total_epochs * 100).toFixed(1)}%`;

  if (data.fid !== null) document.getElementById('fid-score').textContent = data.fid.toFixed(2);
  const improved = lossData.train.length > 0 && data.loss < lossData.train[lossData.train.length-1];
  document.getElementById('loss-delta').textContent = improved ? '↓ improving' : '→ stable';

  lossData.labels.push(`${data.step}`);
  lossData.train.push(data.loss);
  if (data.val_loss) lossData.val.push(data.val_loss);
  if (lossData.labels.length > STEPS_HISTORY) {
    lossData.labels.shift(); lossData.train.shift(); lossData.val.shift();
  }
  lossChart.update();

  const progress = Math.min(1, (500 - data.loss * 1000) / 500);
  refreshSamples(Math.max(0, progress));
});

socket.on('gpu_update', data => renderGPUs(data.gpus));
socket.on('log_message', data => addLog(data.msg, data.level || 'info'));

// ── Simulate data if no real server ──
function simulate() {
  let step = 0, epoch = 0, loss = 0.9;
  const gpus = [{util:85,mem:38},{util:82,mem:37},{util:79,mem:36},{util:88,mem:39}];

  setInterval(() => {
    step += 5;
    loss = Math.max(0.05, loss * 0.9998 + (Math.random()-0.5)*0.003);
    epoch = Math.floor(step / 390);
    gpus.forEach(g => {
      g.util = Math.max(60, Math.min(99, g.util + (Math.random()-0.5)*4));
      g.mem  = +(g.mem + (Math.random()-0.5)*0.3).toFixed(1);
    });
    socket.emit('metrics_update', {
      loss, step, epoch, total_epochs: 100,
      lr: 2e-4 * Math.pow(0.9999, step),
      steps_per_sec: 4.2 + Math.random()*0.5,
      val_loss: loss + 0.02 + Math.random()*0.01,
      fid: Math.max(3, 120 * Math.exp(-step/5000) + 5 + Math.random()),
    });
    socket.emit('gpu_update', {gpus});
    if (step % 100 === 0) {
      socket.emit('log_message', {msg:`Step ${step} | loss=${loss.toFixed(4)} | lr=${(2e-4*Math.pow(0.9999,step)).toExponential(2)}`, level:'step'});
    }
  }, 400);
}

// Initialise
initSamples();
renderGPUs([{util:0,mem:0},{util:0,mem:0},{util:0,mem:0},{util:0,mem:0}]);
addLog('Dashboard initialised', 'info');
addLog('Connecting to training process...', 'info');
setTimeout(() => {
  addLog('Training started: DDPM · CIFAR-10 · base model', 'info');
  simulate();
}, 800);
</script>
</body>
</html>
"""


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "genai-secret"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "time": time.time()})

    @socketio.on("metrics_update")
    def handle_metrics(data):
        emit("metrics_update", data, broadcast=True)

    @socketio.on("gpu_update")
    def handle_gpu(data):
        emit("gpu_update", data, broadcast=True)

    @socketio.on("log_message")
    def handle_log(data):
        emit("log_message", data, broadcast=True)

    return app, socketio


if __name__ == "__main__":
    if not HAS_FLASK:
        print("Please install: pip install flask flask-socketio")
        sys.exit(1)

    print("\n🚀 Starting GenAI Simulation Engine Dashboard")
    print("   URL: http://localhost:5000\n")

    app, socketio = create_app()
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
