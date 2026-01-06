import os
import random
import numpy as np
import argparse
import deepspeed
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as torch_dist
from tqdm import tqdm
import gc
import ray

# imports local methods, classes, etc.
import config.load as cfg # all config arguments
from custom_datasets.prompt_only_dataset import PromptOnlyDataset # our custom pytorch dataset
from misc.utils import safe_string_to_torch_dtype
from rollout_engine import VLLMRolloutEngine

def set_random_seeds(seed):
    '''
        Set random seeds, etc., to make it easier to reproduce results eventhough it is not 100% guaranteed.
        In particualr, since we do distributed training, floating-point arithmetic, non-deterministic operations (e.g., torch.Tensor.index_add_),
        setting the seed is not enough, just make things a bit "predictable".
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rank_setup(world_size):
    '''
        Detect rank from environment variables.
    '''
    # world_size is the total number of gpus (e.g, 2 nodes x 4 gpus = 8 gpus in total). 
    # world size need to be at least 1
    assert world_size >= 1, 'world_size need to be at least 1'

    # Unique id of gpu in the ENTIRE WORLD. It ranges from 0 to world_size - 1
    rank = int(os.environ.get('RANK', 0))

    # Unique id of gpu in the LOCAL node (or simply one node). It ranges from 0 to local_node_size - 1
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # add some checks to make sure number of gpus and local rank are correct.
    if not torch.cuda.is_available():
        if rank == 0:
            print("Warning: CUDA is not available, running on CPU. Sorry!")
    else:
        num_local_gpus = torch.cuda.device_count()
        if local_rank >= num_local_gpus:
            raise RuntimeError(f"LOCAL_RANK {local_rank} >= available GPUs {num_local_gpus}")

        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def setup_ray(ray_address):
    '''
       Initialize ray cluster and setup master address.
    '''
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)

    else:
        ray.init(ignore_reinit_error=True)

    try:
        master_addr = ray.util.get_node_ip_address()

    except Exception:
        print("Warning: Could not get master address, using localhost")
        master_addr = "127.0.0.1"

    return ray, master_addr

def training_runner_setup(model_path,
                           ref_model_path,
                           model_dtype,
                           trust_remote_code,
                           attn_impl,
                           world_size,
                           master_addr,
                           master_port,
                           alg,
                           kl_coeff,
                           clip_low,
                           clip_high,
                           entropy_coeff,
                           use_cache,
                           micro_batch_size_per_gpu,
                           update_after_full_replay):
    '''
        This function is responsible for running the training engine.
    '''
    ray_runners = []
    for rank in range(world_size):
        ray_vars = {
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": "0",
                   }

        runner = alg.options(num_gpus=1,
                            runtime_env={"env_vars": ray_vars},
                            ).remote(
                                     model_path=model_path,
                                     ref_model_path=ref_model_path,
                                     model_dtype=model_dtype,
                                     trust_remote_code=trust_remote_code,
                                     attn_impl=attn_impl,
                                     kl_coeff=kl_coeff,
                                     clip_low=clip_low,
                                     clip_high=clip_high,
                                     entropy_coeff=entropy_coeff,
                                     use_cache=use_cache,
                                     micro_batch_size_per_gpu=micro_batch_size_per_gpu,
                                     update_after_full_replay=update_after_full_replay,
                                     )
        ray_runners.append(runner)

    return ray_runners

def inference_engine_setup(model_path,
                           trust_remote_code,
                           temperature,
                           max_tokens,
                           n_samples,
                           top_p,
                           top_k,
                           seed,
                           ignore_eos,
                           stop,
                           stop_token_ids,
                           prompt_logprobs,
                           force_strict_on_policy,
                           reward_func,
                           tensor_parallel_size,
                           eos_id,
                           reward_broadcast,
                           eps_reward_norm,
                           gen_gpus,
                           ):
    '''
        This function is responsible for setting up distributed inference engine.
    '''
    tp = int(tensor_parallel_size)
    num_rollout_engines = max(1, int(gen_gpus) // tp)

    kwargs = { "model_path": model_path,
               "trust_remote_code": trust_remote_code,
               "temperature": temperature,
               "max_tokens": max_tokens,
               "n_samples": n_samples,
               "top_p": top_p,
               "top_k": top_k,
               "seed": seed,
               "ignore_eos": ignore_eos,
               "stop": stop,
               "stop_token_ids": stop_token_ids,
               "prompt_logprobs": prompt_logprobs,
               "force_strict_on_policy": force_strict_on_policy,
               "reward_func": reward_func,
               "tensor_parallel_size": tensor_parallel_size,
               "eos_id": eos_id,
               "reward_broadcast": reward_broadcast,
               "eps_reward_norm": eps_reward_norm,
            }
    rollout_engines = [
                       VLLMRolloutEngine.options(num_gpus=tp).remote(**kwargs)
                                                                        for _ in range(num_rollout_engines)
                      ]

    return rollout_engines

def load_tokenizer(model_name,
                   trust_remote_code=False,
                   rank=0):
    '''
       Load tokenizer from huggingface.
    '''
    # 1. Tokenizer initialization
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=trust_remote_code)

    # if pad token is not present, we use eos token as pad token
    # log warning if pad token is not present.
    if tokenizer.pad_token_id is None:
        if rank == 0:
            print("Warning: Pad token is not present, using eos token as pad token")

        if getattr(tokenizer, 'eos_token', None) is not None:
            # prefer explicit token if available
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

        else:
            # fallback to eos token id
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

def data_loader_setup(dnames,
                      dataset_ratios,
                      files_path,
                      num_workers,
                      max_seq_len,
                      prompt_key,
                      answer_key,
                      batch_size,
                      tokenizer,
                      seed,
                      split='train',
                      world_size=1,
                      rank=0):
    '''
       Setup DataLoader for distributed training.
       Notes:
           - batch_size is the per-gpu-micro-batch size. Global batch size = batch_size * world_size * gradient_accumulation_steps.
           - Sampler is DistributedSampler; caller must call sampler.set_epoch(epoch) each epoch.
    '''
    # 1. Initialize our custom datasets
    prompt_ds = PromptOnlyDataset(prompt_key=prompt_key,
                            max_seq_len=max_seq_len,
                            tokenizer=tokenizer,
                            data_path=files_path,
                            return_text=False)

    dataloader = DataLoader(
                            dataset=prompt_ds,
                            batch_size=rollout_batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=prompt_ds.collate_fn,
                            )

    return dataloader

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/dummy.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    args = parser.parse_args()

    ########
    # 1. Miscellaneous setups
    ########
    rank, local_rank = rank_setup()
    world_size = config.run.world_size
    config = cfg.load_and_verify(input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 )
    set_random_seeds(seed=config.run.seed)

    ########
    # 2. initialize ray
    ########
    ray_engine, master_addr = setup_ray(ray_address=config.run.ray_address)

    ########
    # 3. initialize training engine
    ########
    if str.lower(config.train.alg_name) in {'pg', 'ppo', 'grpo', 'cispo'}:
        if str.lower(config.train.alg_name) == 'pg':
            import algs.PG.pg as calg
            alg = calg.PG
        elif str.lower(config.train.alg_name) == 'ppo':
            import algs.PPO.ppo as calg
            alg = calg.PPO
        elif str.lower(config.train.alg_name) == 'grpo':
            import algs.GRPO.grpo as calg
            alg = calg.GRPO
        elif str.lower(config.train.alg_name) == 'cispo':
            import algs.CISPO.cispo as calg
            alg = calg.CISPO
    else:
        raise ValueError(f"Unknown algorithm: {config.train.alg_name}")

    training_engine_runner = training_runner_setup(model_path=config.model.name,
                                                  ref_model_path=config.model.ref_model,
                                                  model_dtype=config.model.dtype,
                                                  trust_remote_code=config.model.trust_remote_code,
                                                  attn_impl=config.model.attn_implementation,
                                                  world_size=world_size,
                                                  master_addr=master_addr,
                                                  master_port=config.run.ray_master_port,
                                                  alg=alg,
                                                  kl_coeff=config.train.kl_coeff,
                                                  clip_low=config.train.clip_low,
                                                  clip_high=config.train.clip_high,
                                                  entropy_coeff=config.train.entropy_coeff,
                                                  use_cache=config.model.use_cache,
                                                  train_batch_size_per_gpu=config.train.train_batch_size_per_gpu,
                                                  update_after_full_replay=config.train.update_after_full_replay)
    ########
    # 5. load tokenizer
    ########
    tokenizer = load_tokenizer()

    ########
    # 6. initialize inference engine
    ########
    rollout_engines = inference_engine_setup(model_path=config.model.name,
                                             trust_remote_code=config.model.trust_remote_code,
                                             temperature=config.inference_engine.temperature,
                                             max_tokens=config.inference_engine.max_tokens,
                                             n_samples=config.inference_engine.n_samples,
                                             top_p=config.inference_engine.top_p,
                                             top_k=config.inference_engine.top_k,
                                             seed=config.run.seed,
                                             ignore_eos=config.inference_engine.ignore_eos,
                                             stop=config.inference_engine.stop,
                                             stop_token_ids=config.inference_engine.stop_token_ids,
                                             prompt_logprobs=config.inference_engine.prompt_logprobs,
                                             force_strict_on_policy=config.inference_engine.force_strict_on_policy,
                                             reward_func=config.inference_engine.reward_func,
                                             tensor_parallel_size=config.inference_engine.tensor_parallel_size,
                                             eos_id=tokenizer.eos_token_id,
                                             reward_broadcast=config.inference_engine.reward_broadcast,
                                             eps_reward_norm=config.inference_engine.eps_reward_norm,
                                             gen_gpus=config.run.gen_gpus,)

    ########
    # 6. Build env or data loader
    ########
    train_dataloader, train_sampler = data_loader_setup(dnames=config.data.train_dnames,
                                                        dataset_ratios=config.data.train_ratios,
                                                        files_path=config.data.train_files_path,
                                                        num_workers=config.data.num_workers,
                                                        max_seq_len=config.data.max_seq_len,
                                                        prompt_key=config.data.prompt_key,
                                                        answer_key=config.data.answer_key,
                                                        batch_size=config.train.train_batch_size_per_gpu,
                                                        tokenizer=tokenizer,
                                                        seed=config.run.seed,
                                                        split='train',
                                                        world_size=world_size,
                                                        rank=rank)

    val_dataloader, _ = data_loader_setup(dnames=config.data.train_dnames,
                                          dataset_ratios=config.data.train_ratios,
                                          files_path=config.data.val_files_path,
                                          num_workers=config.data.num_workers,
                                          max_seq_len=config.data.max_seq_len,
                                          prompt_key=config.data.prompt_key,
                                          answer_key=config.data.answer_key,
                                          batch_size=config.train.val_batch_size_per_gpu,
                                          tokenizer=tokenizer,
                                          seed=config.run.seed,
                                          split='val',
                                          world_size=world_size,
                                          rank=rank)

    ########
    # 7. Intitate the learning algorithm (e.g., ppo)
    ########
    if str.lower(config.train.alg_name) in {'pg'}:
        if str.lower(config.train.alg_name) == 'pg':
            import algs.PG.pg as calg
            alg = calg.PG(
                           model_engine=model_engine,
                           optimizer=optimizer,
                           device=model_engine.device,
                           use_cache=config.model.use_cache,
                           normalize_loss=config.train.normalize_loss)

    else:
        raise ValueError(f"Unknown algorithm: {config.train.alg_name}")

    ########
    # 8. Training and evaluation loop
    ########
    if rank == 0:
        print("Starting training...")

    global_step = 0

    # Sync before starting
    # Ensure all nodes have loaded the model and data before anyone starts iterating
    torch.distributed.barrier()

    for epoch in range(config.train.total_number_of_epochs):
        train_sampler.set_epoch(epoch)

        ########
        # 8.1 Training loop
        ########
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.train.total_number_of_epochs}", disable=(rank != 0))

        for step, micro_batch in enumerate(progress_bar):
            # Move batch to gpu (deepspeed engine device)
            micro_batch = {k: v.to(model_engine.device) for k, v in micro_batch.items()}

            # Run one train step for micro-batch.
            metric = alg.train_step(micro_batch)
            global_step += 1

            # logging
            if rank == 0 and step % config.deepspeed.steps_per_print == 0:
                progress_bar.set_postfix(loss=metric['loss'])

        # Sync before validation to ensure consistent state
        torch.distributed.barrier()

        # Clear graph and to reclaim fragmented memory from training ONCE per epoch
        torch.cuda.empty_cache()
        gc.collect()

        ########
        # 8.2 Validation loop
        ########
        # to be safe and caculate loss average across batches and across GPUs correctly, we use
        # the following instead computes per-rank average and then all-reduces averages
        local_sum = torch.tensor(0.0, device=model_engine.device)
        local_count = torch.tensor(0.0, device=model_engine.device)

        for data in val_dataloader:
            val_batch = {k: v.to(model_engine.device) for k, v in data.items()}
            val_metric = alg.eval_step(val_batch)
            local_sum += float(val_metric['loss'])
            local_count += 1

        # Aggregate across all ranks
        torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(local_count, op=torch.distributed.ReduceOp.SUM)

        global_avg_loss = (local_sum / torch.clamp(local_count, min=1.0)).item()
        if rank == 0:
            print(f"Epoch {epoch+1}, Validation Loss: {global_avg_loss}")

        ########
        # 8.3 Save checkpoint
        ########
        # Sync before saving to ensure no one is still writing
        torch.distributed.barrier()

        tag = f"epoch_{epoch+1}"
        # DeepSpeed handles the collective saving internally so we don't need to worry about different ranks.
        model_engine.save_checkpoint(config.deepspeed.monitor_config.get("tensorboard", {}).get("output_path", "./checkpoints"), tag=tag)

        # Wait for saving to complete on all ranks
        torch.distributed.barrier()

    if rank == 0:
        print("Training complete.")