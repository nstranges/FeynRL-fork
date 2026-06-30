"""
Interactive Snake browser UI.

Usage:
    python examples/vlm/sft/snake/smolvlm2-500m-video-instruct/ui.py [--port 5002] [--grid_inner 10] [--seed 0]

Then open http://localhost:5002

Controls:
    Arrow keys / WASD   steer (buffered until next tick)
    A                   toggle oracle auto-play
    R                   reset

The snake ticks at a fixed rate (150 ms/step). Keypresses buffer the next
direction; if no key was pressed since the last tick the snake continues in
its current direction (NONE).  Hitting a wall or your own body ends the game.

Requires: pip install flask
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))  # examples/vlm/ — for games/

from games.snake import SnakeEnv, GRID_INNER

try:
    from flask import Flask, jsonify
except ImportError:
    sys.exit("Flask not found — install with: pip install flask")

app = Flask(__name__)

# ── module-level game state ────────────────────────────────────────────────────
_rng: random.Random = random.Random(0)
_env: SnakeEnv = SnakeEnv()


def _init(grid_inner: int, seed: int) -> None:
    global _rng, _env
    _rng = random.Random(seed)
    _env = SnakeEnv(grid_inner=grid_inner)
    _env.reset(_rng)


def _frame_b64() -> str:
    buf = io.BytesIO()
    _env.render().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _snap():
    return jsonify(
        frame=_frame_b64(),
        score=_env.score,
        step=_env.step_count,
        done=_env.done,
        length=len(_env.snake),
    )


# ── HTML ──────────────────────────────────────────────────────────────────────
_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Snake — FeynRL</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d0d0d;color:#e5e7eb;font-family:'Courier New',monospace;
       display:flex;flex-direction:column;align-items:center;justify-content:center;
       min-height:100vh;gap:16px}
  h1{color:#22c55e;font-size:2rem;letter-spacing:.2em;text-shadow:0 0 20px #16a34a44}
  #game{image-rendering:pixelated;border:2px solid #1f2937;
        width:672px;height:672px;display:block;border-radius:4px}
  #hud{display:flex;gap:40px;font-size:1.05rem}
  .lbl{color:#4b5563}
  .val{color:#d1fae5;font-weight:bold;min-width:3ch;display:inline-block}
  #status{font-size:1rem;font-weight:bold;letter-spacing:.12em;min-height:1.5em}
  #hint{color:#374151;font-size:.78rem;text-align:center;line-height:2}
</style>
</head>
<body>
<h1>SNAKE</h1>
<img id="game" alt="snake">
<div id="hud">
  <span><span class="lbl">SCORE </span><span class="val" id="score">0</span></span>
  <span><span class="lbl">LENGTH </span><span class="val" id="length">3</span></span>
  <span><span class="lbl">STEPS </span><span class="val" id="steps">0</span></span>
</div>
<div id="status"></div>
<div id="hint">
  Arrow&nbsp;keys&nbsp;/&nbsp;WASD&nbsp;steer
  &nbsp;·&nbsp; Shift&#8209;A&nbsp;oracle&nbsp;auto&#8209;play
  &nbsp;·&nbsp; R&nbsp;reset
</div>
<script>
const TICK_MS = 150;

let pendingDir = null;   // direction queued by last keypress; null = NONE
let oracleMode = false;  // Shift-A toggles oracle driving every tick
let ticker     = null;
let gameOver   = false;
let started    = false;  // true after the first keypress of a game
let busy       = false;  // prevent overlapping fetch requests

const $status = document.getElementById('status');
const $score  = document.getElementById('score');
const $length = document.getElementById('length');
const $steps  = document.getElementById('steps');
const $game   = document.getElementById('game');

function applySnap(d) {
  $game.src            = 'data:image/png;base64,' + d.frame;
  $score.textContent   = d.score;
  $length.textContent  = d.length;
  $steps.textContent   = d.step;
  if (d.done && !gameOver) {
    gameOver     = true;
    oracleMode   = false;
    stopTicker();
    $status.textContent = 'FAIL  ·  R to restart';
    $status.style.color = '#ef4444';
  }
}

async function request(path) {
  const r = await fetch(path);
  return r.json();
}

function startTicker() {
  stopTicker();
  ticker = setInterval(async () => {
    if (busy || gameOver) return;
    busy = true;
    try {
      const action = oracleMode ? 'oracle' : (pendingDir || 'NONE');
      pendingDir = null;
      applySnap(await request('/step/' + action));
    } finally {
      busy = false;
    }
  }, TICK_MS);
}

function stopTicker() {
  if (ticker) { clearInterval(ticker); ticker = null; }
}

async function reset() {
  stopTicker();
  oracleMode  = false;
  gameOver    = false;
  started     = false;
  pendingDir  = null;
  busy        = false;
  $status.textContent = 'Press any arrow key to start';
  $status.style.color = '#4b5563';
  applySnap(await request('/reset'));
  // Ticker starts on the first keypress, not here.
}

document.addEventListener('keydown', e => {
  // Arrow keys and WASD steer (lowercase 'a' = LEFT; Shift-A = oracle toggle)
  const dirs = {
    ArrowUp:'UP', ArrowDown:'DOWN', ArrowLeft:'LEFT', ArrowRight:'RIGHT',
    w:'UP', s:'DOWN', a:'LEFT', d:'RIGHT'
  };
  if (dirs[e.key] !== undefined) {
    pendingDir = dirs[e.key];
    if (!started && !gameOver) {
      started = true;
      $status.textContent = '';
      $status.style.color = '';
      startTicker();
    }
    e.preventDefault();
  } else if (e.key === 'A') {          // Shift-A
    if (!gameOver) {
      oracleMode = !oracleMode;
      if (oracleMode) {
        $status.textContent = 'AUTO';
        $status.style.color = '#f87171';
        if (!started) { started = true; startTicker(); }
      } else {
        $status.textContent = started ? '' : 'Press any arrow key to start';
        $status.style.color = started ? '' : '#4b5563';
      }
    }
    e.preventDefault();
  } else if (/^[rR]$/.test(e.key)) {
    reset();
    e.preventDefault();
  }
});

reset();
</script>
</body>
</html>
"""


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return _HTML


@app.get("/reset")
def reset():
    _env.reset(_rng)
    return _snap()


@app.get("/step/<action>")
def step(action: str):
    if not _env.done:
        a = action.upper()
        if a == "ORACLE":
            a = _env.oracle_action(_rng)
        _env.step(a, _rng)
    return _snap()


# ── entry point ───────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Snake browser UI")
    p.add_argument("--port",       type=int, default=5002)
    p.add_argument("--grid_inner", type=int, default=GRID_INNER)
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _init(args.grid_inner, args.seed)
    print(f"Snake UI →  http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=False)
