# Snake — SmolVLM2-500M-Video-Instruct (SFT)

A 500 M VLM learns to play the classic Snake game by imitating a BFS oracle. Given a single PNG frame, the model outputs one of four action tokens (`UP` / `DOWN` / `LEFT` / `RIGHT`). After training it plays coherently from pixel observations alone.

This experiment is a self-contained walkthrough of the full FeynRL SFT pipeline: environment → dataset generation → training → evaluation. It intentionally uses a small, fun task so you can focus on understanding the pipeline rather than waiting for long training runs.

---

## Training Progression

| Checkpoint | Avg Score | Avg Steps |
| ---------- | --------: | --------: |
| Base model (no training) | 0.3 | 6.0 |
| Epoch 1 | 4.4 | 46.6 |
| Epoch 50 (final) | **31.8** | **296.6** |

<video src="results/snake_progression.mp4" autoplay muted playsinline width="720"></video>

_Base model and epoch 1 loop continuously while the best epoch-50 game plays once (score 40, 393 steps)._

---

## Files

```
examples/vlm/sft/snake/
  snake.py                              ← game environment and BFS oracle
  smolvlm2-500m-video-instruct/
    prepare_data.py                     ← generate the SFT dataset
    train.yaml                          ← training config
    rollout.py                          ← evaluate a checkpoint by playing games
    ui.py                               ← interactive browser UI
    results/                            ← pre-generated rollout videos
```

---

## Step 1 — The Environment (`snake.py`)

The game is implemented in `../snake.py` as a single self-contained class with no gym dependency.

**Layout:** 10×10 interior playfield surrounded by a 1-cell wall border. Rendered at 28 px/cell → **336×336 px PNG** per frame.

**Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT`, `NONE` (continue current direction). The model only needs to produce one of these words.

**Oracle:** A BFS shortest-path planner that finds the path to food. When no safe path to food exists, it falls back to a flood-fill survival heuristic that picks the move maximising reachable empty cells (keeping the snake alive as long as possible).

---

## Step 2 — Dataset Generation (`prepare_data.py`)

The dataset is built entirely from oracle rollouts — no human labels required.

```bash
python examples/vlm/sft/snake/smolvlm2-500m-video-instruct/prepare_data.py \
    --output_dir         ./data/snake_sft \
    --num_train_episodes 4000 \
    --num_val_episodes   400  \
    --max_steps          500  \
    --seed               42
```

### What the script does

For each episode the oracle plays the game from start to death (or `max_steps`). Every intermediate step is saved as one row:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `prompt` | `list[dict]` | Single-turn chat message with the task instruction |
| `answer` | `str` | Oracle action: `UP`, `DOWN`, `LEFT`, or `RIGHT` |
| `image_bytes` | `bytes` | PNG-encoded 336×336 frame **before** the action is taken |
| `episode_id` | `str` | Unique episode identifier, e.g. `ep_000042` |
| `frame_index` | `int` | Step number within the episode |
| `score` | `int` | Food eaten so far at this step |
| `snake_length` | `int` | Current snake length |
| `game_state` | `str` | JSON snapshot of snake positions and food location |

The prompt is the same for every row:

> *"You are playing the classic Snake game. Move the green snake to eat the red food and grow. Avoid hitting the gray walls or your own body or the game ends. Reply with exactly one word: UP, DOWN, LEFT, RIGHT, or NONE (to continue in your current direction)."*

### Output

```
./data/snake_sft/
  train.parquet       ← ~879 301 rows (4 000 episodes × avg ~220 steps/episode)
  val.parquet         ← ~88 000 rows  (400 episodes)
  dataset_info.json   ← metadata: episode counts, grid size, seed, prompt example
```

At 4 000 train episodes the dataset is ~880k rows. This is intentionally large — each epoch in `train.yaml` processes only 1/10 of it (`micro_batches_per_epoch: 5496`) so checkpoints are saved more frequently and training progress is visible early.

---

## Step 3 — Training (`train.yaml`)

```bash
python main_sft.py --config examples/vlm/sft/snake/smolvlm2-500m-video-instruct/train.yaml
```

FeynRL's SFT trainer reads `(prompt, answer, image_bytes)` from the parquet, tokenises each row as a single-turn conversation, and trains with cross-entropy on the `answer` tokens only. The image is passed through the VLM's vision encoder; the text decoder predicts the action word.

Checkpoints are saved every epoch under `./ckps/snake-sft/smolvlm2-500m/iter{N:06d}/`.

### Key settings

| Parameter | Value |
| --------- | ----- |
| Model | HuggingFaceTB/SmolVLM2-500M-Video-Instruct |
| Epochs | 50 (each epoch = 1/10 of dataset) |
| Effective batch size | 32 (2 per GPU × 8 GPUs × 2 grad accum) |
| Learning rate | 1e-5, WarmupCosineLR (5% warmup) |
| Max sequence length | 2 048 |
| LoRA | rank 16, α=32, dropout=0.05, q/k/v/o_proj |
| DeepSpeed | ZeRO stage 2, bf16 |
| Hardware | 8×A100 40 GB GPUs |

---

## Step 4 — Evaluation (`rollout.py`)

```bash
python examples/vlm/sft/snake/smolvlm2-500m-video-instruct/rollout.py \
    --checkpoint_dir ./ckps/snake-sft/smolvlm2-500m/iter000050 \
    --output_dir     ./rollouts/snake \
    --num_games      5 \
    --max_steps      500 \
    --fps            8
```

The script runs inference step-by-step: render frame → feed to model → parse predicted action → execute → repeat until death or `max_steps`. It writes one MP4 per game and a `rollout_summary.json` with scores and step counts.

---

## Step 5 — Interactive UI (`ui.py`)

```bash
python examples/vlm/sft/snake/smolvlm2-500m-video-instruct/ui.py \
    --checkpoint_dir ./ckps/snake-sft/smolvlm2-500m/iter000050
```

Then open `http://localhost:5002`. Arrow keys / WASD steer the snake; `A` toggles oracle auto-play; `R` resets.

---

## Evaluation Results

### Final checkpoint (epoch 50)

| Game | Score | Steps | Snake Length |
| ---- | ----: | ----: | -----------: |
| 0    |    38 |   395 |           41 |
| 1    |    30 |   232 |           33 |
| 2    |    18 |   169 |           21 |
| 3    |    40 |   393 |           43 |
| 4    |    33 |   294 |           36 |
| **Avg** | **31.8** | **296.6** | — |

### Epoch 1

| Game | Score | Steps | Snake Length |
| ---- | ----: | ----: | -----------: |
| 0    |     1 |     6 |            4 |
| 1    |     9 |   103 |           12 |
| 2    |     2 |    19 |            5 |
| 3    |     7 |    66 |           10 |
| 4    |     3 |    39 |            6 |
| **Avg** | **4.4** | **46.6** | — |

### Base model (untrained)

| Game | Score | Steps | Snake Length |
| ---- | ----: | ----: | -----------: |
| 0    |     1 |     6 |            4 |
| 1    |     0 |     6 |            3 |
| 2    |     0 |     6 |            3 |
| **Avg** | **0.3** | **6.0** | — |
