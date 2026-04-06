from __future__ import annotations

import os
from importlib import import_module

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:  # pragma: no cover
    create_fastapi_app = import_module("openenv_core.env_server").create_fastapi_app

try:
    from irce.environment import FoodCrisisEnv
    from irce.models import FoodCrisisAction, FoodCrisisObservation
except ImportError:  # pragma: no cover
    from environment import FoodCrisisEnv
    from models import FoodCrisisAction, FoodCrisisObservation

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def create_environment() -> FoodCrisisEnv:
    return FoodCrisisEnv()


app = create_fastapi_app(create_environment, FoodCrisisAction, FoodCrisisObservation)

# ─────────────────────────────────────────────────────────────────────────────
# Web UI — served at GET /
# ─────────────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>FoodCrisisEnv · Incident Command</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Oswald:wght@400;500;600;700&family=Space+Mono&display=swap" rel="stylesheet">
<style>
  /* ── Reset & Base ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #080e08;
    --bg2:      #0d1a0d;
    --bg3:      #112011;
    --border:   #1e3a1e;
    --border2:  #2a4a2a;
    --text:     #c8e6c8;
    --text2:    #7ab87a;
    --text3:    #4a7a4a;
    --accent:   #39ff14;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --blue:     #38bdf8;
    --clean:    #22c55e;
    --contam:   #ef4444;
    --suspect:  #f59e0b;
    --unknown:  #4a7a4a;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-head: 'Oswald', sans-serif;
  }

  html, body {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    line-height: 1.5;
    overflow-x: hidden;
  }

  /* ── Scanline overlay ── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,0,0,0.08) 2px,
      rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
    z-index: 1000;
  }

  /* ── Header ── */
  header {
    background: var(--bg2);
    border-bottom: 2px solid var(--border);
    padding: 0 24px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .logo-icon {
    width: 32px; height: 32px;
    background: var(--accent);
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    filter: drop-shadow(0 0 8px var(--accent));
  }
  .logo-text {
    font-family: var(--font-head);
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--accent);
    text-shadow: 0 0 20px rgba(57,255,20,0.4);
  }
  .logo-sub {
    font-size: 10px;
    color: var(--text3);
    letter-spacing: 3px;
    text-transform: uppercase;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid var(--border2);
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--accent); }
    50%       { opacity: 0.4; box-shadow: none; }
  }
  .hdr-link {
    color: var(--text3);
    text-decoration: none;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1px solid var(--border2);
    padding: 4px 10px;
    border-radius: 3px;
    transition: all .15s;
  }
  .hdr-link:hover { color: var(--accent); border-color: var(--accent); }

  /* ── Layout ── */
  .layout {
    display: grid;
    grid-template-columns: 280px 1fr 280px;
    grid-template-rows: auto 1fr;
    height: calc(100vh - 56px);
    gap: 0;
  }

  /* ── Panels ── */
  .panel {
    background: var(--bg2);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  .panel:last-child { border-right: none; border-left: 1px solid var(--border); }
  .panel-title {
    font-family: var(--font-head);
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text3);
    padding: 12px 16px 8px;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--bg2);
    z-index: 10;
  }
  .panel-body { padding: 12px; flex: 1; }

  /* ── Center (graph area) ── */
  .center {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg);
  }
  .center-toolbar {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }

  /* ── Cards ── */
  .card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 8px;
    overflow: hidden;
  }
  .card-title {
    font-size: 9px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--text3);
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.3);
  }
  .card-body { padding: 8px 10px; }

  /* ── Metrics ── */
  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }
  .metric-label { color: var(--text3); font-size: 11px; }
  .metric-val {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
  }
  .metric-val.good { color: var(--clean); }
  .metric-val.warn { color: var(--amber); }
  .metric-val.bad  { color: var(--red); }

  /* ── Trust bar ── */
  .trust-wrap { margin: 6px 0; }
  .trust-bar-bg {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .trust-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width .5s ease, background .5s ease;
  }

  /* ── Budget bar ── */
  .budget-bar-bg {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 3px;
  }
  .budget-bar-fill {
    height: 100%;
    background: var(--blue);
    border-radius: 2px;
    transition: width .5s ease;
  }

  /* ── Buttons ── */
  .btn {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 1px;
    padding: 6px 12px;
    border-radius: 3px;
    border: 1px solid;
    cursor: pointer;
    transition: all .15s;
    text-transform: uppercase;
    font-weight: 500;
  }
  .btn-primary {
    background: transparent;
    border-color: var(--accent);
    color: var(--accent);
  }
  .btn-primary:hover {
    background: var(--accent);
    color: var(--bg);
    box-shadow: 0 0 12px rgba(57,255,20,0.4);
  }
  .btn-danger {
    background: transparent;
    border-color: var(--red);
    color: var(--red);
  }
  .btn-danger:hover { background: var(--red); color: #fff; }
  .btn-warn {
    background: transparent;
    border-color: var(--amber);
    color: var(--amber);
  }
  .btn-warn:hover { background: var(--amber); color: var(--bg); }
  .btn-ghost {
    background: transparent;
    border-color: var(--border2);
    color: var(--text2);
  }
  .btn-ghost:hover { border-color: var(--text2); color: var(--text); }
  .btn:disabled { opacity: 0.35; cursor: not-allowed; }

  /* ── Select ── */
  select {
    font-family: var(--font-mono);
    font-size: 11px;
    background: var(--bg3);
    border: 1px solid var(--border2);
    color: var(--text);
    padding: 5px 8px;
    border-radius: 3px;
    cursor: pointer;
  }
  select:focus { outline: none; border-color: var(--accent); }

  /* ── Action Grid ── */
  .action-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-bottom: 8px;
  }
  .action-btn {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 1.5px;
    padding: 8px 6px;
    border-radius: 3px;
    border: 1px solid;
    cursor: pointer;
    text-transform: uppercase;
    font-weight: 600;
    transition: all .15s;
    text-align: center;
  }
  .act-inspect  { border-color: var(--blue);   color: var(--blue); background: transparent; }
  .act-quarantine { border-color: var(--amber); color: var(--amber); background: transparent; }
  .act-lift     { border-color: var(--clean);  color: var(--clean); background: transparent; }
  .act-recall   { border-color: #a855f7;       color: #a855f7;     background: transparent; }
  .act-trace    { border-color: var(--text2);  color: var(--text2); background: transparent; }
  .act-wait     { border-color: var(--text3);  color: var(--text3); background: transparent; grid-column: span 2; }
  .act-inspect:hover    { background: var(--blue);  color: var(--bg); }
  .act-quarantine:hover { background: var(--amber); color: var(--bg); }
  .act-lift:hover       { background: var(--clean); color: var(--bg); }
  .act-recall:hover     { background: #a855f7;      color: #fff; }
  .act-trace:hover      { background: var(--text2); color: var(--bg); }
  .act-wait:hover       { background: var(--bg3);   color: var(--text); }

  /* ── Target input ── */
  .target-row {
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
    align-items: center;
  }
  .target-row label { color: var(--text3); font-size: 10px; letter-spacing: 1px; white-space: nowrap; }
  .target-row select { flex: 1; }

  /* ── Result flash ── */
  .result-flash {
    padding: 8px 10px;
    border-radius: 3px;
    font-size: 11px;
    margin-bottom: 8px;
    border-left: 3px solid;
    display: none;
    animation: fadeIn .2s ease;
  }
  @keyframes fadeIn { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:none; } }
  .result-flash.success { border-color: var(--clean); background: rgba(34,197,94,.1); color: var(--clean); }
  .result-flash.error   { border-color: var(--red);   background: rgba(239,68,68,.1);   color: var(--red); }
  .result-flash.ambig   { border-color: var(--amber); background: rgba(245,158,11,.1); color: var(--amber); }

  /* ── Graph ── */
  #graph-wrap {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    overflow: auto;
    position: relative;
  }
  #supply-graph {
    width: 100%;
    height: 100%;
    min-height: 320px;
  }

  /* SVG node styles */
  .g-node rect {
    rx: 6;
    stroke-width: 2;
    transition: all .4s ease;
  }
  .g-node text { font-family: var(--font-mono); fill: #fff; }
  .g-edge { stroke-width: 1.5; fill: none; opacity: 0.4; }
  .g-edge.contaminated { stroke: var(--red); opacity: 0.7; stroke-width: 2.5; }
  .g-edge.active { animation: flowAnim 1.5s linear infinite; }
  @keyframes flowAnim {
    to { stroke-dashoffset: -20; }
  }
  .node-badge {
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity: 0.7;
  }
  .node-sensor { font-size: 11px; font-weight: 600; font-family: 'Space Mono', monospace; }

  /* ── Log ── */
  #episode-log {
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text3);
    max-height: 220px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border2) transparent;
  }
  .log-entry {
    padding: 3px 0;
    border-bottom: 1px solid rgba(30,58,30,0.4);
    display: flex;
    gap: 6px;
    align-items: flex-start;
  }
  .log-step { color: var(--text3); min-width: 28px; }
  .log-action { color: var(--text2); flex: 1; }
  .log-reward.pos { color: var(--clean); }
  .log-reward.neg { color: var(--red); }

  /* ── Lab results / illness pills ── */
  .pill-list { display: flex; flex-wrap: wrap; gap: 4px; }
  .pill {
    font-size: 9px;
    padding: 2px 7px;
    border-radius: 10px;
    border: 1px solid;
    letter-spacing: 0.5px;
    font-weight: 500;
  }
  .pill.contaminated { border-color: var(--red);   color: var(--red);   background: rgba(239,68,68,.1); }
  .pill.clean        { border-color: var(--clean); color: var(--clean); background: rgba(34,197,94,.1); }
  .pill.quarantined  { border-color: var(--amber); color: var(--amber); background: rgba(245,158,11,.1); }
  .pill.illness      { border-color: #a855f7;      color: #a855f7;     background: rgba(168,85,247,.1); }
  .pill.traced       { border-color: var(--blue);  color: var(--blue);  background: rgba(56,189,248,.1); }

  /* ── Illness reports ── */
  .illness-item {
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
  }
  .illness-item:last-child { border-bottom: none; }
  .illness-retailer { color: var(--amber); font-weight: 600; }
  .illness-cases { color: var(--red); }

  /* ── Step counter ── */
  .step-counter {
    font-family: var(--font-head);
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
    text-shadow: 0 0 20px rgba(57,255,20,0.3);
    line-height: 1;
  }
  .step-max { font-size: 13px; color: var(--text3); }

  /* ── Progress ── */
  .progress-ring { position: relative; display: inline-flex; align-items: center; justify-content: center; }
  .progress-ring svg { transform: rotate(-90deg); }
  .progress-ring .ring-bg { fill: none; stroke: var(--border); stroke-width: 4; }
  .progress-ring .ring-fill { fill: none; stroke: var(--accent); stroke-width: 4; stroke-linecap: round; transition: stroke-dashoffset .5s ease; }
  .ring-label { position: absolute; font-family: var(--font-head); font-size: 14px; font-weight: 700; color: var(--text); }

  /* ── Code snippet ── */
  .code-block {
    background: #050a05;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px;
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text2);
    white-space: pre;
    overflow-x: auto;
    line-height: 1.7;
    scrollbar-width: thin;
  }
  .code-kw  { color: #c792ea; }
  .code-str { color: #c3e88d; }
  .code-cmt { color: #546e7a; }
  .code-fn  { color: var(--blue); }
  .code-num { color: #f78c6c; }

  /* ── Task selector tabs ── */
  .task-tabs { display: flex; gap: 4px; }
  .task-tab {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 1px;
    padding: 5px 12px;
    border-radius: 3px;
    border: 1px solid var(--border2);
    background: transparent;
    color: var(--text3);
    cursor: pointer;
    transition: all .15s;
    text-transform: uppercase;
  }
  .task-tab.active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(57,255,20,.05);
  }
  .task-tab:hover:not(.active) { border-color: var(--text3); color: var(--text2); }

  /* ── Endpoint list ── */
  .endpoint-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
  }
  .endpoint-row:last-child { border-bottom: none; }
  .method {
    font-size: 9px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 2px;
    min-width: 34px;
    text-align: center;
  }
  .method.get  { background: rgba(34,197,94,.2);  color: var(--clean); }
  .method.post { background: rgba(56,189,248,.2); color: var(--blue); }
  .endpoint-path { color: var(--text2); font-family: var(--font-mono); }
  .endpoint-desc { color: var(--text3); font-size: 10px; }

  /* ── Done overlay ── */
  #done-overlay {
    display: none;
    position: absolute;
    inset: 0;
    background: rgba(8,14,8,0.85);
    backdrop-filter: blur(4px);
    z-index: 50;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 16px;
    font-family: var(--font-head);
  }
  #done-overlay.show { display: flex; }
  .done-title { font-size: 36px; font-weight: 700; letter-spacing: 4px; color: var(--accent); text-shadow: 0 0 30px var(--accent); }
  .done-score { font-size: 64px; font-weight: 700; color: #fff; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* ── Spinner ── */
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid var(--border2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .6s linear infinite;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Tooltip ── */
  [data-tip] { position: relative; cursor: help; }
  [data-tip]:hover::after {
    content: attr(data-tip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg2);
    border: 1px solid var(--border2);
    color: var(--text);
    padding: 4px 8px;
    border-radius: 3px;
    font-size: 10px;
    white-space: nowrap;
    pointer-events: none;
    z-index: 200;
  }

  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; grid-template-rows: auto; }
    .panel:last-child { border-left: none; border-top: 1px solid var(--border); }
  }
</style>
</head>
<body>

<!-- ── HEADER ── -->
<header>
  <div class="logo">
    <div class="logo-icon">🥬</div>
    <div>
      <div class="logo-text">FOODCRISISENV</div>
      <div class="logo-sub">Supply Chain Contamination Benchmark</div>
    </div>
  </div>
  <div class="header-right">
    <div class="status-pill" id="api-status">
      <div class="status-dot"></div>
      <span id="api-status-text">CONNECTING</span>
    </div>
    <a class="hdr-link" href="/docs" target="_blank">API DOCS</a>
    <a class="hdr-link" href="/redoc" target="_blank">REDOC</a>
  </div>
</header>

<!-- ── MAIN LAYOUT ── -->
<div class="layout">

  <!-- LEFT PANEL: State + Controls -->
  <div class="panel">
    <div class="panel-title">⬡ MISSION CONTROL</div>
    <div class="panel-body">

      <!-- Task Selector -->
      <div class="card">
        <div class="card-title">Task Selection</div>
        <div class="card-body">
          <div class="task-tabs" style="margin-bottom:8px;">
            <button class="task-tab active" onclick="selectTask(1)">T-1 Easy</button>
            <button class="task-tab" onclick="selectTask(2)">T-2 Med</button>
            <button class="task-tab" onclick="selectTask(3)">T-3 Hard</button>
          </div>
          <div id="task-desc" style="font-size:10px;color:var(--text3);line-height:1.5;margin-bottom:8px;">
            Single source, low noise, generous budgets.
          </div>
          <button class="btn btn-primary" style="width:100%;" onclick="resetEnv()">⟳ RESET EPISODE</button>
        </div>
      </div>

      <!-- Step + Progress -->
      <div class="card">
        <div class="card-title">Episode Status</div>
        <div class="card-body" style="display:flex;align-items:center;gap:16px;">
          <div class="progress-ring">
            <svg width="60" height="60" viewBox="0 0 60 60">
              <circle class="ring-bg" cx="30" cy="30" r="24"/>
              <circle class="ring-fill" id="progress-ring" cx="30" cy="30" r="24"
                stroke-dasharray="150.8"
                stroke-dashoffset="150.8"/>
            </svg>
            <div class="ring-label" id="progress-pct">0%</div>
          </div>
          <div>
            <div style="color:var(--text3);font-size:10px;letter-spacing:1px;">STEP</div>
            <div class="step-counter"><span id="step-cur">0</span></div>
            <div class="step-max">/ <span id="step-max">48</span></div>
          </div>
        </div>
      </div>

      <!-- Budgets + Trust -->
      <div class="card">
        <div class="card-title">Resources</div>
        <div class="card-body">
          <div class="metric-row">
            <span class="metric-label" data-tip="Lab inspections remaining">🔬 Lab Budget</span>
            <span class="metric-val" id="lab-budget">—</span>
          </div>
          <div class="budget-bar-bg"><div class="budget-bar-fill" id="lab-bar" style="width:100%;background:var(--blue)"></div></div>
          <div style="margin-top:8px;"></div>
          <div class="metric-row">
            <span class="metric-label" data-tip="Recall budget remaining">📦 Recall Budget</span>
            <span class="metric-val" id="recall-budget">—</span>
          </div>
          <div class="budget-bar-bg"><div class="budget-bar-fill" id="recall-bar" style="width:100%;background:#a855f7"></div></div>
          <div style="margin-top:10px;"></div>
          <div class="metric-row">
            <span class="metric-label" data-tip="Public trust level (0–1)">👥 Public Trust</span>
            <span class="metric-val" id="trust-val">—</span>
          </div>
          <div class="trust-wrap">
            <div class="trust-bar-bg"><div class="trust-bar-fill" id="trust-bar" style="width:100%"></div></div>
          </div>
        </div>
      </div>

      <!-- Last reward -->
      <div class="card">
        <div class="card-title">Last Action Result</div>
        <div class="card-body">
          <div id="result-flash" class="result-flash">—</div>
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:var(--text3);font-size:10px;">Last Reward</span>
            <span id="last-reward" style="font-size:18px;font-weight:700;color:var(--text);">—</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
            <span style="color:var(--text3);font-size:10px;">Tool Result</span>
            <span id="tool-result" style="font-size:11px;color:var(--text2);">—</span>
          </div>
        </div>
      </div>

    </div><!-- /panel-body -->
  </div><!-- /left panel -->


  <!-- CENTER: Supply chain graph -->
  <div class="center">
    <div class="center-toolbar">
      <span style="color:var(--text3);font-size:10px;letter-spacing:2px;text-transform:uppercase;">Supply Chain Graph</span>
      <div style="flex:1;"></div>
      <div style="display:flex;gap:8px;align-items:center;font-size:10px;">
        <span style="color:var(--clean);">■ Clean</span>
        <span style="color:var(--amber);">■ Suspect</span>
        <span style="color:var(--red);">■ Contaminated</span>
        <span style="color:var(--text3);">■ Unknown</span>
        <span style="border:1.5px solid var(--amber);padding:1px 5px;border-radius:2px;color:var(--amber);">Q=Quarantined</span>
      </div>
      <button class="btn btn-ghost" style="padding:4px 8px;font-size:9px;" onclick="fetchState()">↺ REFRESH</button>
      <span id="refresh-spinner" style="display:none;"><span class="spinner"></span></span>
    </div>

    <div id="graph-wrap" style="position:relative;">
      <svg id="supply-graph" viewBox="0 0 800 480" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#2a4a2a"/>
          </marker>
          <marker id="arrowhead-red" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--red)"/>
          </marker>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <text x="400" y="240" text-anchor="middle" fill="#1e3a1e" font-family="IBM Plex Mono" font-size="14">
          Loading supply chain...
        </text>
      </svg>
      <div id="done-overlay">
        <div class="done-title">EPISODE COMPLETE</div>
        <div id="done-label" style="color:var(--text3);font-size:14px;letter-spacing:2px;">SCORE</div>
        <div class="done-score" id="done-score">—</div>
        <button class="btn btn-primary" onclick="resetEnv()">⟳ NEW EPISODE</button>
      </div>
    </div>
  </div>

  <!-- RIGHT PANEL: Actions + Intel -->
  <div class="panel">
    <div class="panel-title">⬡ OPERATIONS</div>
    <div class="panel-body">

      <!-- LLM Mode Selector -->
      <div class="card">
        <div class="card-title">🤖 LLM Control</div>
        <div class="card-body">
          <div class="target-row">
            <label>LLM Mode</label>
            <select id="llm-mode" onchange="updateLLMMode()">
              <option value="manual">Manual Control</option>
              <option value="llm-sample">LLM (Single Step)</option>
              <option value="llm-auto">LLM Auto-Play</option>
            </select>
          </div>
          <button class="btn btn-primary" id="llm-action-btn" style="width:100%;margin-top:8px;display:none;" onclick="runLLMStep()">▶ LLM Decide</button>
          <div id="llm-status" style="margin-top:8px;font-size:10px;color:var(--text3);"></div>
        </div>
      </div>

      <!-- Action sender -->
      <div class="card">
        <div class="card-title">Execute Action</div>
        <div class="card-body">
          <div class="target-row">
            <label>TARGET NODE</label>
            <select id="target-node">
              <option value="">— select node —</option>
            </select>
          </div>
          <div class="target-row">
            <label>TARGET BATCH</label>
            <select id="target-batch">
              <option value="">— select batch —</option>
            </select>
          </div>
          <div class="action-grid">
            <button class="action-btn act-inspect"    onclick="sendAction('INSPECT',   'node')">🔬 INSPECT</button>
            <button class="action-btn act-quarantine" onclick="sendAction('QUARANTINE','node')">🚫 QUARANTINE</button>
            <button class="action-btn act-lift"       onclick="sendAction('LIFT',      'node')">✓ LIFT</button>
            <button class="action-btn act-recall"     onclick="sendAction('RECALL',    'batch')">📦 RECALL</button>
            <button class="action-btn act-trace"      onclick="sendAction('TRACE',     'batch')">🔍 TRACE</button>
            <button class="action-btn act-wait"       onclick="sendAction('WAIT',      'none')">⏸ WAIT</button>
          </div>
        </div>
      </div>

      <!-- Lab Results -->
      <div class="card">
        <div class="card-title">Lab Results</div>
        <div class="card-body">
          <div class="pill-list" id="lab-results-pills">
            <span style="color:var(--text3);font-size:10px;">No results yet.</span>
          </div>
        </div>
      </div>

      <!-- Quarantined nodes -->
      <div class="card">
        <div class="card-title">Quarantined Nodes</div>
        <div class="card-body">
          <div class="pill-list" id="quarantine-pills">
            <span style="color:var(--text3);font-size:10px;">None.</span>
          </div>
        </div>
      </div>

      <!-- Illness Reports -->
      <div class="card">
        <div class="card-title">⚠ Illness Reports</div>
        <div class="card-body" id="illness-reports">
          <span style="color:var(--text3);font-size:10px;">No reports yet.</span>
        </div>
      </div>

      <!-- Traced Batches -->
      <div class="card">
        <div class="card-title">Traced Batches</div>
        <div class="card-body">
          <div class="pill-list" id="traced-pills">
            <span style="color:var(--text3);font-size:10px;">None traced yet.</span>
          </div>
          <div id="trace-paths" style="margin-top:6px;font-size:10px;color:var(--text3);"></div>
        </div>
      </div>

      <!-- Episode Log -->
      <div class="card">
        <div class="card-title">Episode Log</div>
        <div class="card-body" style="padding:6px 8px;">
          <div id="episode-log"><span style="color:var(--text3);">Awaiting first action...</span></div>
        </div>
      </div>

      <!-- API Endpoints -->
      <div class="card">
        <div class="card-title">API Endpoints</div>
        <div class="card-body">
          <div class="endpoint-row">
            <span class="method get">GET</span>
            <div><div class="endpoint-path">/health</div><div class="endpoint-desc">Liveness check</div></div>
          </div>
          <div class="endpoint-row">
            <span class="method post">POST</span>
            <div><div class="endpoint-path">/reset</div><div class="endpoint-desc">Start new episode</div></div>
          </div>
          <div class="endpoint-row">
            <span class="method post">POST</span>
            <div><div class="endpoint-path">/step</div><div class="endpoint-desc">Take one action</div></div>
          </div>
          <div class="endpoint-row">
            <span class="method get">GET</span>
            <div><div class="endpoint-path">/state</div><div class="endpoint-desc">Full env state</div></div>
          </div>
        </div>
      </div>

      <!-- Connect Your Agent -->
      <div class="card">
        <div class="card-title">Connect Your Agent</div>
        <div class="card-body" style="padding:6px;">
          <div class="code-block"><span class="code-kw">from</span> irce.client <span class="code-kw">import</span> FoodCrisisEnvClient
<span class="code-kw">from</span> irce.models <span class="code-kw">import</span> FoodCrisisAction

BASE = <span class="code-str">"https://&lt;your-space&gt;.hf.space"</span>

<span class="code-kw">with</span> <span class="code-fn">FoodCrisisEnvClient</span>(BASE).<span class="code-fn">sync</span>() <span class="code-kw">as</span> env:
    obs = env.<span class="code-fn">reset</span>(task_id=<span class="code-num">1</span>)
    <span class="code-kw">while not</span> obs.done:
        action = <span class="code-fn">FoodCrisisAction</span>(
            action_type=<span class="code-str">"WAIT"</span>
        )
        obs = env.<span class="code-fn">step</span>(action)</div>
        </div>
      </div>

    </div><!-- /panel-body -->
  </div><!-- /right panel -->

</div><!-- /layout -->

<script>
// ── State ──
let currentTask = 1;
let lastState   = null;
let episodeLog  = [];
let refreshTimer = null;
let labBudgetMax   = {1:10, 2:6, 3:4};
let recallBudgetMax = {1:100, 2:60, 3:40};
let maxSteps       = {1:48, 2:60, 3:72};
let llmMode = 'manual';
let autoPlayLLM = false;
const taskDescs = {
  1: "Single source, low noise, fast reports, generous budgets. Perfect for learning the environment.",
  2: "Multi-source outbreak with delayed reports and tighter budgets. Prioritization is key.",
  3: "Adversarial false spikes, re-seeding, delayed reports, and high trust pressure. Frontier difficulty."
};

// ── Init ──
window.addEventListener('load', async () => {
  await checkHealth();
  const r = await fetch('/reset', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({task_id: 1, seed: 7})
  });
  const data = await r.json();
  const obs = data.observation || data;
  updateUI(obs, null, null);
  startAutoRefresh();
});

async function checkHealth() {
  try {
    const r = await fetch('/health');
    if (r.ok) {
      document.getElementById('api-status-text').textContent = 'LIVE';
      document.querySelector('.status-dot').style.background = 'var(--clean)';
    }
  } catch {
    document.getElementById('api-status-text').textContent = 'OFFLINE';
    document.querySelector('.status-dot').style.background = 'var(--red)';
  }
}

function selectTask(t) {
  currentTask = t;
  document.querySelectorAll('.task-tab').forEach((el,i) => {
    el.classList.toggle('active', i+1 === t);
  });
  document.getElementById('task-desc').textContent = taskDescs[t];
  document.getElementById('step-max').textContent = maxSteps[t];
}

async function resetEnv() {
  episodeLog = [];
  updateLog();
  document.getElementById('done-overlay').classList.remove('show');
  
  // Pause auto-refresh during reset
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = null;
  
  try {
    const r = await fetch('/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task_id: currentTask, seed: 7})
    });
    const data = await r.json();
    const obs = data.observation || data;
    updateUI(obs, null, null);
    
    // Restart auto-refresh after a short delay to ensure state is ready
    setTimeout(startAutoRefresh, 500);
  } catch(e) {
    console.error('Reset failed', e);
    startAutoRefresh();
  }
}

async function fetchState() {
  document.getElementById('refresh-spinner').style.display = 'inline';
  try {
    const r = await fetch('/state');
    const data = await r.json();
    const obs = data.observation || data;
    updateUI(obs, null, null);
  } catch(e) { console.error('State fetch failed', e); }
  finally { document.getElementById('refresh-spinner').style.display = 'none'; }
}

async function sendAction(verb, targetType) {
  let target = '';
  if (targetType === 'node') {
    const sel = document.getElementById('target-node');
    target = sel.value;
    if (!target) { showFlash('Select a target node first.', 'error'); return; }
  } else if (targetType === 'batch') {
    const sel = document.getElementById('target-batch');
    target = sel.value;
    if (!target) { showFlash('Select a target batch first.', 'error'); return; }
  }
  const actionStr = target ? `${verb} ${target}` : verb;
  try {
    const r = await fetch('/step', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: {action_type: actionStr}})
    });
    const data = await r.json();
    const obs  = data.observation || data;
    const rew  = data.reward ?? obs.reward ?? 0;
    const done = data.done   ?? obs.done   ?? false;
    updateUI(obs, rew, actionStr);
    addLogEntry(obs.step_count ?? episodeLog.length+1, actionStr, rew, obs.tool_result);
    if (done) { 
      autoPlayLLM = false;
      showDone(obs);
    } else if (autoPlayLLM && llmMode === 'llm-auto') {
      setTimeout(runLLMStep, 1500);
    }
  } catch(e) {
    showFlash('Request failed: ' + e.message, 'error');
    autoPlayLLM = false;
  }
}

function updateLLMMode() {
  const mode = document.getElementById('llm-mode').value;
  llmMode = mode;
  const btn = document.getElementById('llm-action-btn');
  if (mode === 'llm-sample') {
    btn.style.display = 'block';
    autoPlayLLM = false;
  } else if (mode === 'llm-auto') {
    btn.style.display = 'none';
    if (!autoPlayLLM && lastState && !lastState.done) {
      autoPlayLLM = true;
      runLLMStep();
    }
  } else {
    btn.style.display = 'none';
    autoPlayLLM = false;
  }
}

async function runLLMStep() {
  if (!lastState) return;
  const statusEl = document.getElementById('llm-status');
  statusEl.textContent = '⏳ LLM thinking...';
  
  try {
    const prompt = buildPrompt(lastState);
    statusEl.textContent = '📡 Querying LLM...';
    
    const llmResponse = await queryLLM(prompt);
    const actionStr = parseLLMAction(llmResponse);
    
    statusEl.textContent = `💡 LLM: ${actionStr}`;
    
    const r = await fetch('/step', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: {action_type: actionStr}})
    });
    const data = await r.json();
    const obs  = data.observation || data;
    const rew  = data.reward ?? obs.reward ?? 0;
    
    updateUI(obs, rew, actionStr);
    addLogEntry(obs.step_count ?? episodeLog.length+1, actionStr, rew, '');
    
    if (obs.done) {
      autoPlayLLM = false;
      statusEl.textContent = '✓ Episode complete';
      showDone(obs);
    } else if (autoPlayLLM) {
      setTimeout(runLLMStep, 1500);
    }
  } catch(e) {
    statusEl.textContent = '❌ LLM Error: ' + e.message;
    autoPlayLLM = false;
  }
}

function buildPrompt(obs) {
  const summary = obs.natural_language_summary || '';
  const nodes = (obs.nodes || []).map(n => `${n.node_id} (${n.node_type})`).join(', ');
  const infected = (obs.lab_results || {});
  const qNodes = Object.entries(obs.quarantine_status || {}).filter(([,v]) => v).map(([k]) => k);
  
  return `You are a food safety incident responder with this situation:

${summary}

Current state:
- Timestep: ${obs.timestep || obs.step_count || 0}
- Nodes: ${nodes}
- Confirmed infected: ${Object.keys(infected).length ? Object.keys(infected).join(', ') : 'None'}
- Quarantined: ${qNodes.length ? qNodes.join(', ') : 'None'}
- Lab budget: ${obs.lab_budget || 0}
- Recall budget: ${obs.recall_budget || 0}
- Public trust: ${(obs.public_trust || 1.0).toFixed(2)}

Your options: INSPECT <node>, QUARANTINE <node>, LIFT <node>, RECALL <batch>, TRACE <batch>, WAIT

Respond with ONLY ONE action string, like: "TRACE batch_001" or "WAIT"
Response:`;
}

async function queryLLM(prompt) {
  try {
    const r = await fetch('/llm/decide', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: prompt})
    });
    
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || `LLM API error: ${r.status}`);
    }
    
    const data = await r.json();
    return data.action || 'WAIT';
  } catch(e) {
    throw new Error('LLM connection failed: ' + e.message);
  }
}

function parseLLMAction(text) {
  const actions = ['INSPECT', 'QUARANTINE', 'LIFT', 'RECALL', 'TRACE', 'WAIT'];
  const upper = text.toUpperCase();
  
  for (const act of actions) {
    if (upper.includes(act)) {
      if (act === 'WAIT') return 'WAIT';
      const match = text.match(new RegExp(act + '\\s+([\\w_]+)','i'));
      if (match) return `${act} ${match[1]}`;
      return act;
    }
  }
  return 'WAIT';
}

function updateUI(obs, reward, action) {
  if (!obs) return;
  lastState = obs;

  // Step counter
  const step = obs.step_count ?? obs.timestep ?? 0;
  document.getElementById('step-cur').textContent = step;
  const mx = maxSteps[currentTask];
  const pct = Math.round((step / mx) * 100);
  document.getElementById('progress-pct').textContent = pct + '%';
  const circ = 2 * Math.PI * 24;
  document.getElementById('progress-ring').style.strokeDashoffset = circ - (circ * pct / 100);

  // Budgets
  const lb = obs.lab_budget ?? 0;
  const rb = obs.recall_budget ?? 0;
  const lbMax = labBudgetMax[currentTask];
  const rbMax = recallBudgetMax[currentTask];
  document.getElementById('lab-budget').textContent = lb;
  document.getElementById('lab-budget').className = 'metric-val ' + (lb === 0 ? 'bad' : lb <= 2 ? 'warn' : 'good');
  document.getElementById('lab-bar').style.width = Math.round((lb/lbMax)*100) + '%';
  document.getElementById('recall-budget').textContent = rb;
  document.getElementById('recall-budget').className = 'metric-val ' + (rb < 10 ? 'bad' : rb < 30 ? 'warn' : '');
  document.getElementById('recall-bar').style.width = Math.round((rb/rbMax)*100) + '%';

  // Trust
  const trust = obs.public_trust ?? 1.0;
  document.getElementById('trust-val').textContent = trust.toFixed(2);
  document.getElementById('trust-val').className = 'metric-val ' + (trust < 0.5 ? 'bad' : trust < 0.75 ? 'warn' : 'good');
  const trustBar = document.getElementById('trust-bar');
  trustBar.style.width = Math.round(trust * 100) + '%';
  trustBar.style.background = trust < 0.5 ? 'var(--red)' : trust < 0.75 ? 'var(--amber)' : 'var(--clean)';

  // Reward
  if (reward !== null && reward !== undefined) {
    const rEl = document.getElementById('last-reward');
    rEl.textContent = (reward >= 0 ? '+' : '') + reward.toFixed(2);
    rEl.style.color = reward > 0 ? 'var(--clean)' : reward < 0 ? 'var(--red)' : 'var(--text3)';
  }
  if (obs.tool_result) {
    document.getElementById('tool-result').textContent = obs.tool_result;
    showFlash(obs.natural_language_summary
      ? obs.natural_language_summary.split('.')[0] + '.'
      : obs.tool_result,
      obs.tool_result === 'SUCCESS' ? 'success' : obs.tool_result === 'ERROR' ? 'error' : 'ambig'
    );
  }

  // Populate node/batch dropdowns
  const nodes = obs.nodes || [];
  console.log('DEBUG: obs.nodes =', nodes, 'length =', nodes.length);
  const nodeSelect = document.getElementById('target-node');
  const batchSelect = document.getElementById('target-batch');
  
  if (!nodeSelect || !batchSelect) {
    console.error('ERROR: target-node or target-batch select not found in DOM');
    return;
  }
  
  const prevNode = nodeSelect.value;
  const prevBatch = batchSelect.value;
  nodeSelect.innerHTML = '<option value="">— select node —</option>';
  batchSelect.innerHTML = '<option value="">— select batch —</option>';
  
  if (nodes.length === 0) {
    console.warn('WARNING: obs.nodes is empty! Full obs:', obs);
  }
  
  const seenBatches = new Set();
  nodes.forEach(n => {
    const opt = document.createElement('option');
    opt.value = n.node_id;
    const q = (obs.quarantine_status || {})[n.node_id] ? ' [Q]' : '';
    const r = (obs.lab_results || {})[n.node_id] ? ` [${obs.lab_results[n.node_id][0].toUpperCase()}]` : '';
    opt.textContent = `${n.node_id}${q}${r}`;
    if (opt.value === prevNode) opt.selected = true;
    nodeSelect.appendChild(opt);
    console.log('Added node option:', opt.textContent);
    (n.batch_ids || []).forEach(bid => {
      if (!seenBatches.has(bid)) {
        seenBatches.add(bid);
        const bopt = document.createElement('option');
        bopt.value = bid;
        bopt.textContent = bid;
        if (bopt.value === prevBatch) bopt.selected = true;
        batchSelect.appendChild(bopt);
        console.log('Added batch option:', bid);
      }
    });
  });

  // Lab results pills
  const labResults = obs.lab_results || {};
  const labPills = document.getElementById('lab-results-pills');
  if (Object.keys(labResults).length === 0) {
    labPills.innerHTML = '<span style="color:var(--text3);font-size:10px;">No results yet.</span>';
  } else {
    labPills.innerHTML = Object.entries(labResults).map(([node, res]) =>
      `<span class="pill ${res}">${node}: ${res}</span>`
    ).join('');
  }

  // Quarantined pills
  const qStatus = obs.quarantine_status || {};
  const qNodes = Object.entries(qStatus).filter(([,v]) => v).map(([k]) => k);
  const qPills = document.getElementById('quarantine-pills');
  qPills.innerHTML = qNodes.length
    ? qNodes.map(n => `<span class="pill quarantined">🚫 ${n}</span>`).join('')
    : '<span style="color:var(--text3);font-size:10px;">None quarantined.</span>';

  // Illness reports
  const illness = obs.illness_reports || [];
  const illnessEl = document.getElementById('illness-reports');
  illnessEl.innerHTML = illness.length
    ? illness.map(r => `
        <div class="illness-item">
          <span class="illness-retailer">${r.retailer_id}</span>
          — <span class="illness-cases">${r.case_count} case(s)</span>
          <span style="color:var(--text3)"> @ t${r.timestep_reported}</span>
        </div>`).join('')
    : '<span style="color:var(--text3);font-size:10px;">No reports yet.</span>';

  // Traced batches
  const traced = obs.traced_batches || {};
  const tracedPills = document.getElementById('traced-pills');
  const tracedPaths = document.getElementById('trace-paths');
  if (Object.keys(traced).length === 0) {
    tracedPills.innerHTML = '<span style="color:var(--text3);font-size:10px;">None traced yet.</span>';
    tracedPaths.innerHTML = '';
  } else {
    tracedPills.innerHTML = Object.keys(traced).map(bid =>
      `<span class="pill traced">🔍 ${bid}</span>`
    ).join('');
    tracedPaths.innerHTML = Object.entries(traced).map(([bid, path]) =>
      `<div style="margin-top:4px;"><span style="color:var(--blue)">${bid}:</span><br>
       <span style="color:var(--text3)">${Array.isArray(path) ? path.join(' → ') : path}</span></div>`
    ).join('');
  }

  // Draw graph
  drawGraph(obs);
}

// ── Graph rendering ──
function nodeColor(nodeId, obs) {
  const labRes = (obs.lab_results || {})[nodeId];
  const sensor = (obs.sensor_readings || {})[nodeId] ?? 0;
  if (labRes === 'contaminated') return { fill: '#3a0f0f', stroke: 'var(--red)',   glow: true };
  if (labRes === 'clean')        return { fill: '#0f2a0f', stroke: 'var(--clean)', glow: false };
  if (sensor >= 0.7)             return { fill: '#2a1a0f', stroke: 'var(--amber)', glow: false };
  if (sensor >= 0.4)             return { fill: '#1a200a', stroke: '#84cc16',      glow: false };
  return                                { fill: '#0d1a0d', stroke: 'var(--border2)', glow: false };
}

function drawGraph(obs) {
  const svg = document.getElementById('supply-graph');
  const nodes = obs.nodes || [];
  if (!nodes.length) return;

  // Group by type, maintaining order
  const byType = { farm: [], processing: [], warehouse: [], retailer: [] };
  nodes.forEach(n => { if (byType[n.node_type]) byType[n.node_type].push(n); });

  const W = 800, H = 480;
  const cols = ['farm', 'processing', 'warehouse', 'retailer'];
  const colX = [80, 260, 450, 650];
  const nodeW = 120, nodeH = 52;
  const positions = {};

  cols.forEach((type, ci) => {
    const group = byType[type];
    const total = group.length;
    const span = Math.max(total * 70, 60);
    const startY = H/2 - span/2;
    group.forEach((n, i) => {
      positions[n.node_id] = {
        x: colX[ci] - nodeW/2,
        y: startY + i * (span / Math.max(total-1, 1)) - nodeH/2
      };
      if (total === 1) positions[n.node_id].y = H/2 - nodeH/2;
    });
  });

  // Build SVG
  let svgContent = `
    <defs>
      <marker id="arr" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
        <polygon points="0 0, 7 2.5, 0 5" fill="#2a4a2a"/>
      </marker>
      <marker id="arr-red" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
        <polygon points="0 0, 7 2.5, 0 5" fill="var(--red)"/>
      </marker>
      <filter id="glow-fx">
        <feGaussianBlur stdDeviation="4" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
  `;

  // Column labels
  const colLabels = ['FARMS', 'PROCESSING', 'WAREHOUSES', 'RETAILERS'];
  cols.forEach((type, ci) => {
    svgContent += `<text x="${colX[ci]}" y="22" text-anchor="middle"
      font-family="IBM Plex Mono" font-size="9" fill="#2a4a2a" letter-spacing="2">
      ${colLabels[ci]}
    </text>`;
  });

  // Edges first (behind nodes)
  const labRes = obs.lab_results || {};
  const qStatus = obs.quarantine_status || {};

  nodes.forEach(n => {
    const from = positions[n.node_id];
    if (!from) return;
    (n.connected_to || []).forEach(toId => {
      const to = positions[toId];
      if (!to) return;
      const isContam = labRes[n.node_id] === 'contaminated';
      const isQ = qStatus[n.node_id];
      const x1 = from.x + nodeW;
      const y1 = from.y + nodeH/2;
      const x2 = to.x;
      const y2 = to.y + nodeH/2;
      const mx = (x1 + x2) / 2;
      const strokeColor = isContam && !isQ ? 'var(--red)' : isQ ? '#2a4a2a' : '#1e3a1e';
      const marker = isContam && !isQ ? 'arr-red' : 'arr';
      const dasharray = isQ ? '4,4' : '';
      const dash = dasharray ? `stroke-dasharray="${dasharray}"` : '';
      svgContent += `<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}"
        stroke="${strokeColor}" fill="none" stroke-width="${isContam && !isQ ? 2.5 : 1.5}"
        ${dash} marker-end="url(#${marker})" opacity="${isQ ? 0.25 : 0.6}"/>`;
    });
  });

  // Nodes
  nodes.forEach(n => {
    const pos = positions[n.node_id];
    if (!pos) return;
    const { x, y } = pos;
    const col = nodeColor(n.node_id, obs);
    const sensor = (obs.sensor_readings || {})[n.node_id] ?? 0;
    const isQ = qStatus[n.node_id];
    const labR = labRes[n.node_id];
    const glowFilter = col.glow ? 'filter="url(#glow-fx)"' : '';

    // Short label
    const label = n.node_id.length > 14 ? n.node_id.slice(0,13)+'…' : n.node_id;

    // Status icon
    let icon = '';
    if (labR === 'contaminated') icon = '☣';
    else if (labR === 'clean')   icon = '✓';
    else if (isQ)                icon = '⚠';
    else if (sensor >= 0.7)     icon = '!';

    svgContent += `
      <g class="g-node" ${glowFilter}>
        <rect x="${x}" y="${y}" width="${nodeW}" height="${nodeH}"
          rx="5" fill="${col.fill}" stroke="${col.stroke}" stroke-width="${isQ ? 2.5 : 1.5}"
          stroke-dasharray="${isQ ? '5,3' : ''}"/>
        ${icon ? `<text x="${x+nodeW-10}" y="${y+14}" text-anchor="middle"
            font-family="IBM Plex Mono" font-size="12" fill="${col.stroke}">${icon}</text>` : ''}
        <text x="${x+nodeW/2}" y="${y+20}" text-anchor="middle"
          font-family="IBM Plex Mono" font-size="10" font-weight="600" fill="#c8e6c8">
          ${label}
        </text>
        <text x="${x+nodeW/2}" y="${y+32}" text-anchor="middle"
          font-family="IBM Plex Mono" font-size="9" fill="${col.stroke}">
          ${n.node_type.toUpperCase()}
        </text>
        <text x="${x+nodeW/2}" y="${y+46}" text-anchor="middle"
          font-family="Space Mono" font-size="11" font-weight="600"
          fill="${sensor >= 0.7 ? 'var(--red)' : sensor >= 0.4 ? 'var(--amber)' : '#4a7a4a'}">
          ${sensor.toFixed(2)}
        </text>
        ${(n.batch_ids||[]).length > 0 ? `
        <text x="${x+4}" y="${y+46}" text-anchor="start"
          font-family="IBM Plex Mono" font-size="8" fill="#2a4a2a">
          ${(n.batch_ids||[]).length}b
        </text>` : ''}
      </g>`;
  });

  svg.innerHTML = svgContent;
}

// ── Episode Log ──
function addLogEntry(step, action, reward, result) {
  episodeLog.push({ step, action, reward, result });
  updateLog();
}

function updateLog() {
  const el = document.getElementById('episode-log');
  if (!episodeLog.length) {
    el.innerHTML = '<span style="color:var(--text3);">Awaiting first action...</span>';
    return;
  }
  el.innerHTML = [...episodeLog].reverse().slice(0,30).map(e => {
    const rClass = (e.reward ?? 0) >= 0 ? 'pos' : 'neg';
    const rStr = e.reward != null ? ((e.reward >= 0 ? '+' : '') + Number(e.reward).toFixed(2)) : '—';
    return `<div class="log-entry">
      <span class="log-step">t${e.step}</span>
      <span class="log-action">${e.action}</span>
      <span class="log-reward ${rClass}">${rStr}</span>
    </div>`;
  }).join('');
}

// ── Flash message ──
function showFlash(msg, type) {
  const el = document.getElementById('result-flash');
  el.textContent = msg;
  el.className = 'result-flash ' + type;
  el.style.display = 'block';
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.style.display = 'none'; }, 4000);
}

// ── Done overlay ──
function showDone(obs) {
  const overlay = document.getElementById('done-overlay');
  overlay.classList.add('show');
  const prog = (obs.progress_hint ?? 0);
  document.getElementById('done-score').textContent = prog.toFixed(3);
  document.getElementById('done-score').style.color =
    prog >= 0.7 ? 'var(--clean)' : prog >= 0.5 ? 'var(--amber)' : 'var(--red)';
}

// ── Auto refresh ──
function startAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(fetchState, 3000);
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Endpoint — Call Groq model for decisions
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/llm/decide")
async def llm_decide(prompt: dict):
    """
    Call the LLM with a prompt and return a decision.
    Expects: {"prompt": "..."}
    Reads credentials from environment variables (HF_TOKEN, API_BASE_URL, MODEL_NAME)
    """
    if not OpenAI:
        raise HTTPException(status_code=500, detail="OpenAI client not installed")
    
    api_key = os.getenv("HF_TOKEN")
    api_base = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="HF_TOKEN not configured in Space secrets")
    
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a food safety incident responder. Respond with exactly one action."},
                {"role": "user", "content": prompt.get("prompt", "")}
            ],
            temperature=0.3,
            max_tokens=50,
        )
        decision = response.choices[0].message.content.strip()
        return {"action": decision}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui_root() -> HTMLResponse:
    return HTMLResponse(content=_HTML, status_code=200)