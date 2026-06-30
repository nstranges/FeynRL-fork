"""
Run a trained VLM checkpoint on the Snake game until the snake dies,
then export the full episode as a video (MP4 if imageio is available, GIF otherwise).

Usage:
    python examples/vlm/sft/snake/smolvlm2-500m-video-instruct/rollout.py \
        --checkpoint_dir ./ckps/snake-sft/smolvlm2-500m/iter000050 \
        --output_dir     ./rollouts/snake \
        --num_games      5 \
        --max_steps      500 \
        --fps            8

The script runs inference step-by-step:
  1. Render current frame
  2. Feed frame + instruction to the model (greedy decode, max 4 new tokens)
  3. Parse the predicted action word
  4. Execute in the environment
  5. Repeat until done or max_steps reached
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel
from transformers import AutoProcessor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))   # examples/vlm/ — for games/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", ".."))  # repo root

from games.snake import SnakeEnv, PROMPT, ACTIONS
from misc.model_loading import build_hf_model


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Path to a saved FeynRL SFT checkpoint (may contain adapter_config.json for LoRA).")
    p.add_argument("--output_dir",     type=str, default="./rollouts/snake")
    p.add_argument("--num_games",      type=int, default=5)
    p.add_argument("--max_steps",      type=int, default=500)
    p.add_argument("--fps",            type=int, default=8)
    p.add_argument("--grid_inner",     type=int, default=10)
    p.add_argument("--seed",           type=int, default=0)
    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────
def load_model_and_processor(checkpoint_dir: str):
    model = build_hf_model(
        model_path=checkpoint_dir,
        model_dtype="bfloat16" if torch.cuda.is_available() else "float32",
        model_class="vlm",
        trust_remote_code=False,
        attn_impl="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()
    return model, processor, device


# ── inference ─────────────────────────────────────────────────────────────────
_VALID = set(ACTIONS)  # includes "NONE"

def predict_action(model, processor, pil_image: Image.Image, device) -> str:
    """Greedy-decode one action token from the current frame."""
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    enc  = processor(text=text, images=[pil_image], return_tensors="pt")
    enc  = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=4)

    new_ids  = out[0, enc["input_ids"].shape[1]:]
    decoded  = processor.tokenizer.decode(new_ids, skip_special_tokens=True).upper().strip()

    for word in decoded.split():
        if word in _VALID:
            return word
    return "UP"   # safe fallback


# ── video export ──────────────────────────────────────────────────────────────
def _annotate(frame: Image.Image, action: str, score: int, step: int) -> Image.Image:
    """Overlay action + HUD text on a copy of frame."""
    out  = frame.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14
        )
    except (IOError, OSError):
        font = ImageFont.load_default()

    hud = f"→ {action:<5}   SCORE {score:03d}   STEP {step:04d}"
    # semi-transparent black bar over the top border row
    draw.rectangle([0, 0, out.width, 28], fill=(0, 0, 0))
    draw.text((6, 7), hud, fill=(220, 220, 220), font=font)
    return out


def save_video(frames: list[Image.Image], output_path: str, fps: int) -> str:
    arr = [np.array(f) for f in frames]
    try:
        import imageio
        imageio.mimwrite(output_path, arr, fps=fps)
        return output_path
    except ImportError:
        gif_path = os.path.splitext(output_path)[0] + ".gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0,
        )
        return gif_path


# ── rollout loop ──────────────────────────────────────────────────────────────
def run_game(
    model, processor, device,
    *,
    game_idx: int,
    rng: random.Random,
    grid_inner: int,
    max_steps: int,
    fps: int,
    output_dir: str,
) -> dict:
    env = SnakeEnv(grid_inner=grid_inner)
    env.reset(rng)

    frames: list[Image.Image] = []
    actions: list[str] = []

    while not env.done and env.step_count < max_steps:
        frame  = env.render()
        action = predict_action(model, processor, frame, device)
        annotated = _annotate(frame, action, env.score, env.step_count)
        frames.append(annotated)
        actions.append(action)
        env.step(action, rng)

    # Final "game over" frame with no action overlay
    final = env.render()
    draw  = ImageDraw.Draw(final)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18
        )
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.rectangle([0, 0, final.width, 28], fill=(0, 0, 0))
    draw.text((6, 5), f"GAME OVER  SCORE {env.score:03d}", fill=(239, 68, 68), font=font)
    frames.append(final)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"snake_game_{game_idx:03d}.mp4")
    saved    = save_video(frames, out_path, fps)

    summary = {
        "game":        game_idx,
        "score":       env.score,
        "steps":       env.step_count,
        "snake_length": len(env.snake),
        "saved":       saved,
    }
    print(json.dumps(summary))
    return summary


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint_dir}")
    model, processor, device = load_model_and_processor(args.checkpoint_dir)
    print(f"Model on {device}. Running {args.num_games} game(s) …\n")

    rng      = random.Random(args.seed)
    results  = []
    for i in range(args.num_games):
        r = run_game(
            model, processor, device,
            game_idx=i,
            rng=rng,
            grid_inner=args.grid_inner,
            max_steps=args.max_steps,
            fps=args.fps,
            output_dir=args.output_dir,
        )
        results.append(r)

    avg_score = sum(r["score"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    print(f"\nSummary over {args.num_games} games:")
    print(f"  avg score : {avg_score:.1f}")
    print(f"  avg steps : {avg_steps:.1f}")
    print(f"  videos    : {args.output_dir}")

    with open(os.path.join(args.output_dir, "rollout_summary.json"), "w") as fh:
        json.dump({"results": results, "avg_score": avg_score, "avg_steps": avg_steps}, fh, indent=2)


if __name__ == "__main__":
    main()
