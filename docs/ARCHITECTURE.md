# FeynRL Architecture Overview

FeynRL is designed with a **separation of concerns** between algorithmic logic and system-level orchestration. This modularity allows researchers and engineers to focus on developing new methods while leveraging a scalable, high-performance training stack.

## System Components

### 🛰️ Orchestration (Ray)
Ray serves as the central orchestrator, managing the lifecycle of distributed workers across a cluster. It schedules:
- **Training Workers**: Handle distributed training using DeepSpeed.
- **Rollout Workers**: Generate trajectories using vLLM rollout engines.

### 🖥️ Training Engine (DeepSpeed)
The training engine utilizes **DeepSpeed** for distributed training, supporting:
- **ZeRO Stage 1/2/3**: Efficient parameter, gradient, and optimizer state partitioning.
- **CPU Offloading**: Optional offloading of optimizer states and parameters to CPU memory to handle larger models.
- **LoRA Support**: Parameter-efficient fine-tuning via PEFT integration.

### 🎲 Rollout Engine (vLLM)
Trajectory generation is powered by **vLLM**, which provides:
- **High Throughput Generation**: Optimized kernels and PagedAttention for fast inference.
- **Tensor Parallelism**: Capability to shard large models across multiple GPUs for rollout.
- **Dynamic Loading**: Support for directly updating policy weights during training.

### 🔄 Weight Synchronization
FeynRL supports three methods for syncing weights from the training engine to the rollout workers, with an automatic fallback chain (NCCL to direct to disk):
1. **NCCL Broadcast** (fastest): Training rank 0 broadcasts weights directly to all rollout engine TP workers over a dedicated NCCL process group. Zero-copy, no serialization overhead. Requires all participants to be idle (no pending Ray tasks in their mailboxes).
2. **Direct Sync**: Weights are gathered from training engines via DeepSpeed's ZeRO gather and pushed to rollout workers through Ray's shared-memory object store. No disk I/O, but involves serialization.
3. **Disk Sync**: Weights are saved as a checkpoint to disk, and rollout workers reload from the saved path. Slowest, but always available as a last-resort fallback.

If NCCL sync fails, FeynRL automatically falls back to direct sync. If direct sync also fails, it falls back to disk. This ensures rollout engines always receive updated weights.

### 🔁 Training↔Rollout Scheduling
FeynRL supports two execution modes that control how rollout generation and training are scheduled:

1. **Synchronous** (`main_rl.py: run_epoch_sync`): Each epoch generates all rollouts (blocking), trains on them, syncs weights, and repeats. Fully on-policy, simple to debug, and the right choice when data freshness matters more than throughput.

2. **Overlap** (`main_rl.py: run_epoch_overlap`): Generation and training run concurrently on separate GPUs within the same epoch. This mode significantly reduces GPU idle time and improves throughput. It works as follows:

   - **Chunk-based dispatch**: The dataloader is consumed in small chunks (configurable `chunk_size`). Only one chunk is ever in-flight at a time, keeping each rollout engine's FIFO mailbox shallow so NCCL sync can execute immediately between chunks.
   - **Interleaved training**: While a chunk generates on the rollout engines, the driver runs as many training steps as fit on the training engines using already-collected data. When the chunk finishes (or training is done), the driver finalizes the chunk, adds results to the replay buffer, and dispatches the next chunk.
   - **ESS-driven weight sync**: An Effective Sample Size (ESS) factor, computed from importance weights inside the training step, monitors how much the current policy has diverged from the rollout policy. When ESS drops below a configurable threshold (`ess_sync_threshold`), training stops early and an NCCL weight sync is triggered at the chunk boundary while engines are idle. This is an **adaptive** approach: unlike fixed sync intervals used in other frameworks, ESS responds to the actual divergence between the training and rollout policies, so syncs happen exactly when they are needed rather than on a rigid schedule. When the policy changes slowly, syncs are infrequent and training proceeds uninterrupted; when the policy shifts quickly, syncs happen sooner to keep rollout data fresh. A fixed sync interval can also be used as a fallback when ESS is not available.
   - **Staleness control**: A configurable `max_lag` controls how many policy versions the rollout data can lag behind. The replay buffer retains samples from recent policies and evicts stale ones, enabling mild off-policy reuse without unbounded staleness.
   - **Fallback chain**: Weight sync uses NCCL broadcast (fastest), falling back to direct transfer via Ray object store, then to disk-based checkpoint reload if needed. This ensures rollout engines always receive updated weights regardless of the cluster environment.
   - **Bulk training**: If generation finishes before all training steps are consumed (e.g., small prompts, few chunks), the remaining steps run in a bulk training phase after all chunks are finalized.

   **When to use each mode:**
   - Use **overlap mode** when generation is the bottleneck (large models, long sequences) and you want to fill training GPU idle time with useful work.
   - Use **sync mode** when you need strict on-policy data, are debugging, or when training and generation take roughly the same time (overlap provides little benefit in that case).

## 🧩 Modularity & Extensibility

- **Algorithm Agnostic**: The system is designed to support various algorithms by providing a common interface for data handling and model updates.
- **Pluggable Rewards**: Custom reward functions can be easily integrated in the configuration.
- **Flexible Data Processing**: The data pipeline supports mixed-dataset sampling with configurable ratios, allowing for complex training recipes.

## 📂 Repository Structure

```text
FeynRL/
├── algs/               # Implementation of various algorithms such as PPO, GRPO, CISPO, DPO, SFT
├── configs/            # YAML configuration files and Pydantic schema validation
├── data_feeds/         # Data loading, mixed-dataset sampling, and dataset construction
├── data_prep/          # Scripts for processing raw datasets
├── docs/               # Documentation files (Installation, FAQ, Architecture, Troubleshooting)
├── experiments/        # Experiment configurations and documentation
├── misc/               # Utility modules (logging, trackers, helpers)
├── rewards/            # Reward functions for RL training
├── rollouts/           # vLLM-powered rollout engine and weight synchronization
├── main_rl.py          # Entry point for Reinforcement Learning training
├── main_sl.py          # Entry point for Supervised Fine-Tuning (SFT)
├── main_cl.py          # Entry point for Preference Learning (e.g., DPO)
├── main_eval.py        # Entry point for standalone model evaluation
├── requirements.txt    # Project dependencies
├── CONTRIBUTING.md     # Contribution guidelines
├── LICENSE             # Project license
├── .gitignore          # Git ignore rules
└── README.md           # Main project landing page
```