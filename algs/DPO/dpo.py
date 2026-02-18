import torch
import numpy as np

class DPO:
    def __init__(self, model_engine,
                 ref_model_engine,
                 optimizer,
                 beta,
                 normalize_loss=False):

        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        # ref model would not be updated, so we put it in eval mode
        self.ref_model_engine.eval()
        self.optimizer = optimizer
        self.normalize_loss = normalize_loss
        self.beta = beta

        # use cross entropy loss
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, logprobs, ref_logprobs, loss_mask):
        '''
         This implements length-normalized dpo loss
         logprobs and ref_logprobs are [2B, T -1]
         loss_mask is [2B, T -1]
        '''
        # [2B, T -1]
        # B here is actually 2 * B_micro
        B, T = logprobs.shape

        # logprobs: [2B, T -1]
        # chosen: logprobs[:B] so [B, T -1]
        # rejected: logprobs[B:] so [B, T -1]
        chosen_logprobs   = logprobs[:B]
        rejected_logprobs = logprobs[B:]
        # reference model logprobs
        ref_chosen_logprobs   = ref_logprobs[:B]
        ref_rejected_logprobs = ref_logprobs[B:]

        # [2B, T -1] --> [B, T -1]
        chosen_loss_mask    = loss_mask[:B]
        rejected_loss_mask  = loss_mask[B:]

        # compute length-normalized dpo loss
        # loss = -E[log_sigmoid[(beta/len(c)) * log(p_c/pr_c) - (beta/len(r)) * log(p_r/pr_r)]
        len_chosen = chosen_loss_mask.sum(dim=1, keepdim=True)
        len_rejected = rejected_loss_mask.sum(dim=1, keepdim=True)
        chosen_logratios   = chosen_loss_mask * (chosen_logprobs - ref_chosen_logprobs).to(torch.float32) / len_chosen
        rejected_logratios = rejected_loss_mask * (rejected_logprobs - ref_rejected_logprobs).to(torch.float32) / len_rejected

        per_token_loss = -F.logsigmoid(self.beta * (chosen_logratios - rejected_logratios))
        denom = (chosen_loss_mask + rejected_loss_mask).sum().clamp(min=1.0)
        loss = per_token_loss.sum() / denom

        metrics = {"chosen_rewards":float(chosen_logratios.mean().item()),
                   "rejected_rewards":float(rejected_logratios.mean().item()),
                   "reward_accuracies":float((chosen_logratios > rejected_logratios).float().mean().item())
                  }
        return loss, metrics

    def forward(self, batch):
        '''
            batch[input_ids/attn_mask] are [B, 2, T]
            batch['position_ids'] is [B, 2, T] or None
            Returns:
                logits is [2B, T-1, vocab_size]
                y is [2B, T-1]
                loss_mask is [2B, T-1]
        '''
        # inputs are [B, 2, T] as chosen and rejected
        B, _, T = batch['input_ids'].shape
        # [B, 2, T] --> [2B, T]
        input_ids = batch['input_ids'].view(-1, T)
        att_mask  = batch['attn_mask'].view(-1, T)
        # loss_mask is [2B, T -1]
        loss_mask = batch['loss_mask'].contiguous()

        # if pos_ids is not provided, hf will add it automatically.
        pos_ids = batch.get('position_ids', None)
        if pos_ids is not None:
            # [B, 2, T] --> [2B, T]
            pos_ids = pos_ids.view(-1, T).to(att_mask.device)

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=False)

        # feed data to ref model without gradient
        with torch.no_grad():
            ref_output = self.ref_model_engine(input_ids=input_ids,
                                               attention_mask=att_mask,
                                               position_ids=pos_ids,
                                               use_cache=False)

        # [2B, T, vocab_size]
        every_logits     = output.logits
        ref_every_logits = ref_output.logits

        # remember we use token t to predict token t+1, hence no need to predict last
        # token's output (e.g., <eos>) and we remove it from logits.
        # [2B, T, vocab_size] -> [2B, T-1, vocab_size]
        logits     = every_logits[:, :-1, :].contiguous()
        ref_logits = ref_every_logits[:, :-1, :].contiguous()
        _, T, vocab_size = logits.shape

        # label would be input_ids shifted by one
        # so the size is [2B, T-1] --> [2B * (T-1)]
        target_ids = input_ids[:, 1:].contiguous().view(-1)

        # cross_entropy return -logprobs but we need logprobs
        # logits is [2B, T-1, vocab_size] --> [2B * (T-1), vocab_size]
        # target_ids is [B * (T-1)]
        # Compute token logprobs in float32 to avoid bf16/fp16 quantization
        neg_logprobs = self.cross_entropy(logits.to(torch.float32).view(-1, vocab_size), target_ids)
        # [2B * (T-1)] -> [2B, T-1]
        logprobs = -neg_logprobs.view(B, T_minus_1)

        neg_logprobs_ref = self.cross_entropy(ref_logits.to(torch.float32).view(-1, vocab_size), target_ids)
        # [2B * (T-1)] -> [2B, T-1]
        ref_logprobs = -neg_logprobs_ref.view(B, T_minus_1)

        return logprobs, ref_logprobs, loss_mask

    def eval_step(self, micro_batch):
        '''
           This implements a single validation step per rank/gpu.
        '''
        # we need to split data into micro batches
        self.model_engine.eval()
        with torch.no_grad():
            logprobs, ref_logprobs, loss_mask = self.forward(micro_batch)

            # 3. compute loss pass
            loss, metrics = self.compute_loss(logprobs=logprobs, ref_logprobs=ref_logprobs, loss_mask=loss_mask)

        return {"loss": float(loss.item()), **metrics}

    def train_step(self, micro_batch):
        '''
           This implements a single training step per rank/gpu
           for given micro_batch_size_per_gpu.
        '''
        # make sure model is in training mode
        self.model_engine.train()

        # Don't need to zero_grad() here as ds handles gradient zeroing
        # internally after step() when gradient_accumulation_steps boundary is reached.
        # 1. forward pass per gpu/rank
        # chosen and rejected data are stacked as[B, 2, T]
        logprobs, ref_logprobs, loss_mask = self.forward(micro_batch)

        # 3. compute loss pass
        loss, metrics = self.compute_loss(logprobs=logprobs, ref_logprobs=ref_logprobs, loss_mask=loss_mask)

        # 4. backward step
        # deepspeed aggregates gradients and only updates weights when accumulation_steps is reached.
        self.model_engine.backward(loss)

        # 5. optimizer step
        self.model_engine.step()

        # return loss
        train_loss = loss.item()

        return {"loss": float(train_loss), **metrics}