"""
Generate the classic Snake SFT dataset.

Each parquet row is one environment step:
  prompt       list[dict]   chat message with the task instruction
  answer       str          oracle action word (UP / DOWN / LEFT / RIGHT)
  image_bytes  bytes        PNG-encoded 336×336 frame *before* the action is executed

The oracle uses BFS to find the shortest path to food, falling back to a
flood-fill survival heuristic when no direct path exists.

Run:
    python examples/vlm/sft/snake/smolvlm2-500m-video-instruct/prepare_data.py \
        --output_dir ./data/vla-games/snake_sft \
        --num_train_episodes 4000 \
        --num_val_episodes 400
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))   # examples/vlm/ — for games/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", ".."))  # repo root

from games.snake import SnakeEnv, PROMPT, GRID_INNER


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",          type=str, default="./data/snake_sft")
    p.add_argument("--num_train_episodes",  type=int, default=4000)
    p.add_argument("--num_val_episodes",    type=int, default=400)
    p.add_argument("--grid_inner",          type=int, default=GRID_INNER)
    p.add_argument("--max_steps",           type=int, default=500)
    p.add_argument("--seed",                type=int, default=42)
    return p.parse_args()


def _build_prompt() -> list[dict[str, str]]:
    return [{"role": "user", "content": PROMPT}]


def generate_episodes(
    num_episodes: int,
    *,
    start_idx: int,
    rng: random.Random,
    grid_inner: int,
    max_steps: int,
) -> list[dict]:
    rows: list[dict] = []
    prompt = _build_prompt()

    for offset in range(num_episodes):
        ep_id = start_idx + offset
        env   = SnakeEnv(grid_inner=grid_inner)
        env.reset(rng)

        for t in range(max_steps):
            action = env.oracle_action(rng)
            rows.append(
                {
                    "prompt":       prompt,
                    "answer":       action,
                    "image_bytes":  env.to_bytes(),
                    "episode_id":   f"ep_{ep_id:06d}",
                    "frame_index":  t,
                    "task":         "snake",
                    "score":        env.score,
                    "snake_length": len(env.snake),
                    "game_state":   json.dumps(env.get_state()),
                }
            )
            _, done = env.step(action, rng)
            if done:
                break

    return rows


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng  = random.Random(args.seed)

    print(f"Generating {args.num_train_episodes} train episodes …")
    train_rows = generate_episodes(
        args.num_train_episodes,
        start_idx=0,
        rng=rng,
        grid_inner=args.grid_inner,
        max_steps=args.max_steps,
    )

    print(f"Generating {args.num_val_episodes} val episodes …")
    val_rows = generate_episodes(
        args.num_val_episodes,
        start_idx=args.num_train_episodes,
        rng=rng,
        grid_inner=args.grid_inner,
        max_steps=args.max_steps,
    )

    pd.DataFrame(train_rows).to_parquet(os.path.join(args.output_dir, "train.parquet"), index=False)
    pd.DataFrame(val_rows).to_parquet(os.path.join(args.output_dir, "val.parquet"),   index=False)

    avg_train_len = len(train_rows) / max(1, args.num_train_episodes)
    meta = {
        "task":               "snake",
        "grid_inner":         args.grid_inner,
        "max_steps":          args.max_steps,
        "num_train_episodes": args.num_train_episodes,
        "num_val_episodes":   args.num_val_episodes,
        "num_train_rows":     len(train_rows),
        "num_val_rows":       len(val_rows),
        "avg_episode_length": round(avg_train_len, 1),
        "seed":               args.seed,
        "image_key":          "image_bytes",
        "prompt_example":     _build_prompt(),
        "answer_example":     "RIGHT",
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
