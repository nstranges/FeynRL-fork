# Snake Game

<video src="smolvlm2-500m-video-instruct/results/snake_progression.mp4" autoplay muted playsinline width="720"></video>

_Left to right: base model (score 0.3) → epoch 1 (score 4.4) → epoch 50 (score 31.8). Base and epoch 1 loop while the final game plays once._

Teaching a vision-language model to play classic Snake from pixel observations via supervised imitation of a BFS oracle.

---

## Files

```text
snake/
├── snake.py                              # Environment, renderer, and BFS oracle
├── prepare_data.py                       # Generate the oracle-imitation dataset
└── smolvlm2-500m-video-instruct/        # SFT experiment with SmolVLM2-500M-Video-Instruct
    ├── train.yaml                        # Training config
    ├── rollout.py                        # Evaluate a checkpoint by playing games
    ├── ui.py                             # Interactive browser UI
    └── results/                          # Pre-generated rollout videos
```

---

## The Environment (`snake.py`)

The game runs as a simple Python class with no gym dependency. `snake.py` contains the environment, the pixel renderer, and the BFS oracle — everything needed to generate data or play the game.

**Layout:** 10×10 interior playfield surrounded by a 1-cell wall border, rendered at 28 px/cell → **336×336 px PNG** per frame.

**Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT` (and `NONE` to continue the current direction). The model predicts exactly one of these words per step.

**Oracle:** BFS finds the shortest path from the snake's head to food. When no safe path to food exists, it falls back to a flood-fill survival heuristic that picks the move maximising the number of reachable empty cells, keeping the snake alive as long as possible.

---

## Data Preparation (`prepare_data.py`)

The dataset is generated entirely from oracle rollouts — no human labels or external data required.

```bash
python examples/vlm/sft/snake/prepare_data.py \
    --output_dir         ./data/snake_sft \
    --num_train_episodes 4000 \
    --num_val_episodes   400  \
    --max_steps          500  \
    --seed               42
```

### How it works

For each episode the oracle plays the game from start to death (or `--max_steps`). Every intermediate step is written as one row in the parquet:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `prompt` | `list[dict]` | Single-turn chat message with the task instruction (identical for every row) |
| `answer` | `str` | Oracle action: `UP`, `DOWN`, `LEFT`, or `RIGHT` |
| `image_bytes` | `bytes` | PNG-encoded 336×336 frame **before** the action is taken |
| `episode_id` | `str` | Unique episode identifier, e.g. `ep_000042` |
| `frame_index` | `int` | Step number within the episode |
| `score` | `int` | Food eaten so far at this step |
| `snake_length` | `int` | Current snake length |
| `game_state` | `str` | JSON snapshot of snake body positions and food location |

The prompt sent to the model with every frame:

> *"You are playing the classic Snake game. Move the green snake to eat the red food and grow. Avoid hitting the gray walls or your own body or the game ends. Reply with exactly one word: UP, DOWN, LEFT, RIGHT, or NONE (to continue in your current direction)."*

### Output

```
./data/snake_sft/
  train.parquet       ← ~879 301 rows (4 000 episodes × avg ~220 steps/episode)
  val.parquet         ← ~88 000 rows  (400 episodes)
  dataset_info.json   ← metadata: episode counts, grid size, seed, prompt example
```

At 4 000 train episodes the dataset is ~880k rows. The training config processes 1/10 of this per epoch (`micro_batches_per_epoch: 5496`) so checkpoints are saved frequently and you can observe learning early (scores improve significantly by epoch 1).

---

## Experiments

| Model | Algorithm | Avg Score | Avg Steps | Details |
| ----- | --------- | --------: | --------: | ------- |
| HuggingFaceTB/SmolVLM2-500M-Video-Instruct | SFT | **31.8** | **296.6** | [smolvlm2-500m-video-instruct/README.md](smolvlm2-500m-video-instruct/README.md) |
