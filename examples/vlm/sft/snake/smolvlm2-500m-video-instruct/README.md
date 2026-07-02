# Snake — SmolVLM2-500M-Video-Instruct (SFT)

<video src="results/snake_progression.mp4" autoplay muted playsinline width="720"></video>

_Base model (score 0.3) and epoch 1 (score 4.4) loop continuously while the best epoch-50 game (score 40, 393 steps) plays once._

A 500 M VLM learns to play the classic Snake game by imitating a BFS oracle. Given a single PNG frame, the model outputs one of four action tokens (`UP` / `DOWN` / `LEFT` / `RIGHT`). After training it plays coherently from pixel observations alone.

See [`../README.md`](../README.md) for environment details and dataset preparation.

---

## Training Progression

| Checkpoint | Avg Score | Avg Steps |
| ---------- | --------: | --------: |
| Base model (no training) | 0.3 | 6.0 |
| Epoch 1 | 4.4 | 46.6 |
| Epoch 50 (final) | **31.8** | **296.6** |

---

## Quickstart

### 1 — Generate the SFT dataset

```bash
python examples/vlm/sft/snake/prepare_data.py \
    --output_dir ./data/snake_sft \
    --num_train_episodes 4000 \
    --num_val_episodes 400
```

Produces `./data/snake_sft/train.parquet` and `./data/snake_sft/val.parquet`. Update `data.train_files_path` / `data.val_files_path` in `train.yaml` if you place them elsewhere.

### 2 — Train

```bash
python main_sft.py --config examples/vlm/sft/snake/smolvlm2-500m-video-instruct/train.yaml
```

Checkpoints are saved every epoch under `./ckps/snake-sft/smolvlm2-500m/`.

### 3 — Evaluate (rollout)

```bash
python examples/vlm/sft/snake/smolvlm2-500m-video-instruct/rollout.py \
    --checkpoint_dir ./ckps/snake-sft/smolvlm2-500m/iter000050 \
    --output_dir     ./rollouts/snake \
    --num_games      5 \
    --max_steps      500 \
    --fps            8
```

Writes per-game MP4 files and `rollout_summary.json` under the output directory.

### 4 — Interactive UI

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

---

## Key Training Settings

| Parameter               | Value                                         |
| ----------------------- | --------------------------------------------- |
| Model                   | HuggingFaceTB/SmolVLM2-500M-Video-Instruct    |
| Epochs                  | 50 (each epoch = 1/10 of data)                |
| Effective batch size    | 32 (2 per GPU × 8 GPUs × 2 grad accum)       |
| Learning rate           | 1e-5 with WarmupCosineLR (5% warmup)          |
| Max sequence length     | 2 048                                         |
| LoRA rank               | 16 (α=32, dropout=0.05, q/k/v/o_proj)        |
| DeepSpeed               | ZeRO stage 2, bf16                            |
| Full dataset size       | ~879 301 steps across 4 000 episodes          |
| Hardware                | 8×A100 40 GB GPUs                             |
