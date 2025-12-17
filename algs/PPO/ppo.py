import torch
import numpy as np

class PPO:
    def __init__(self,
                model_engine,
                optimizer,
                micro_batch_size_per_gpu,
                clip_grad_norm=None,
                use_cache=False,
                kl_coeff=0.0,
                clip_low=0.0,
                clip_high=1.0,
                use_gae=False,
                gae_lambda=0.95,
                ref_model=None,
                gamma=0.99,
                device='cpu'):

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.micro_batch_size_per_gpu = micro_batch_size_per_gpu
        self.use_cache = use_cache
        self.kl_coeff = kl_coeff
        self.ref_model = ref_model
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.device = device

        # clip gradient if required
        if clip_grad_norm is not None:
            self.clip_grad = torch.nn.utils.clip_grad_norm_
        else:
            self.clip_grad = None

    def compute_advantages(self, rewards, masks, vals, last_val):
        '''
            rewards: [B, T]
            vals: [B, T] (bootstrap value for last step if not done)
            last_val: [B, 1] contains value of the last time step (for bootstrap) 
            masks: [B, T + 1] where 1 = done/invalid, 0 = valid continuation
            discounted returns:
                if it reaches the end of horizon (done = 1), then 
                        ret_T = reward_T 
                elif it doesn't and we have shoter episode than horizon (done = 0), then 
                        ret_t = reward_t + gamma * values_{t+1}
                        values_{t+1} is bootstrap value as if we had continued the episode
                so put above together: 
                    [T]: ret_T = reward_T + gamma * values_{T+1} * (1 - mask_{T+1})
                    and
                    [t]: ret_t = reward_t + gamma * ret_{t+1} * mask_{t+1}
            GAE:
                delta = reward_t + gamma * values_{t+1} * (1 - mask_{t+1}) - values_t
                gae_t = delta + gamma * tau * gae_{t+1} * (1 - mask_{t+1})
                ret_t = gae_t + values_t
        '''
        device = rewards.device
        B, T   = rewards.shape
        rets = torch.zeros_like(rewards)
        advs = torch.zeros_like(rewards)
        
        # variables for GAE for only one time step
        val_t_plus_1 = last_val.squeeze(-1).detach()
        adv_t_plus_1 = torch.zeros(B).to(device)

        for t in reversed(range(T)): # reminder reversed loop runs from T-1 to 0 
            # just to increase readability
            val_t    = vals[:, t].detach()
            reward_t = rewards[:, t]
            mask_t_plus_1 = masks[:, t + 1]

            # delta is just TD residual (delta_t)
            delta = reward_t + self.gamma * val_t_plus_1 * (1 - mask_t_plus_1) - val_t
            if self.use_gae == True:
                adv_t_plus_1 = delta + self.gamma * self.tau * adv_t_plus_1 * (1 - mask_t_plus_1)
                advs[:, t] = adv_t_plus_1

            else:
                advs[:, t] = delta

            # adding vals_t converts it back to discounted returns
            # ret_t = reward[:, t] + self.gamma * next_val * (1 - masks[:, t + 1]) - vals_t  + vals_T
            rets[:, t] = advs[:, t] + val_t
            val_t_plus_1 = val_t         

        return rets, advs

    def compute_loss(self, logits, y, loss_mask):
        '''
         This function implements \sum_{i=1}^{N} log p(y_i|x_i)
         y is target label [B, T -1]
         logits is model prediction [B, T -1, vocab_size]
        '''
        # [B, T -1, vocab_size]
        _, _, vocab_size = logits.shape

        # flatten logits across batch and seq_len before computing loss
        # so logits is [B * (T -1), vocab_size]
        logits = logits.view(-1, vocab_size)
        # flatten y as well:  [B, T -1] -->  [B * (T -1)]
        y = y.view(-1)

        # per token loss
        per_token_loss = self.loss_fn(logits, y)

        # We need to apply mask to loss to remove any things 
        # which should not be considered in loss (e.g., padding tokens)
        loss_mask = loss_mask.view(-1).to(dtype=per_token_loss.dtype)  # [B * (T - 1)]
        masked_per_token_loss = per_token_loss * loss_mask

        # To avoid gradient accumulation error caused by loss.mean(),
        # we use sum of loss instead but play with learning rate to account for this.
        loss = masked_per_token_loss.sum()

        # Loss_accumulated \neq Loss_full_batch when sequence lengths vary.
        # to address that, we normalize by total sequence length (constant)
        # which is fixed across gpus, not valid tokens (variable) which is loss_mask.sum().
        # This solves the gradient accumulation bug.
        if self.normalize_loss:
            total_possible_tokens = logits.shape[0]
            if total_possible_tokens == 0:
                # This shouldn't happen
                raise ValueError("Cannot compute loss: total_possible_tokens is 0")

            loss = loss / total_possible_tokens

        return loss

    def forward(self, batch):
        '''
            This function implements a single forward pass for current batch.
            It returns logits, y, and mask.
            The size of batch['seq_ids']/batch['seq_attn_mask'] is [B, T]
            The size of batch['loss_mask'] is [B, T-1]
            The size of return variables logits, y, and mask would be [B, T-1]
        '''
        # batch is a dictionary, so we want to extract things we need from it
        # input_ids and att_mask are [B, T]
        input_ids = batch['seq_ids']
        att_mask  = batch['seq_attn_mask']
        # loss_mask is [B, T -1]
        loss_mask = batch['loss_mask'].contiguous()

        # if pos_ids is not provided, HF will add that automatically.
        pos_ids   = batch.get('position_ids', None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(self.device)

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, vocab_size]
        every_token_logits = output.logits

        # label would be input_ids shifted by one (input_ids[:, 1:])
        # so the size is [B, T-1]
        y = input_ids[:, 1:].contiguous()
        # it is next token prediction, so we remove last token from logits
        logits = every_token_logits[:, :-1, :].contiguous()

        return logits, y, loss_mask

    def eval_step(self, micro_batch):
        '''
           This function implements a single validation step per rank/gpu.
        '''
        # we need to split data into micro batches
        self.model_engine.eval()
        with torch.no_grad():
            # forward pass per gpu/rank
            logits, y, loss_mask = self.forward(micro_batch)

            # compute loss pass
            loss = self.compute_loss(logits=logits, y=y, loss_mask=loss_mask)
            val_loss = loss.item()

        return {"loss": float(val_loss)}

    def train_step(self, micro_batch):
        '''
           This function implements a single training step per rank/gpu.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu. 
           The DataLoader already yields micro-batches. 
        '''
        # make sure model is in training mode
        self.model_engine.train()

        # 1. For DeepSpeed, zero_grad should be called at the start of the training step
        # especially important with gradient accu.
        self.model_engine.zero_grad()

        # 2. forward pass per gpu/rank
        logits, y, loss_mask = self.forward(micro_batch)

        # 3. compute loss pass
        loss = self.compute_loss(logits=logits, y=y, loss_mask=loss_mask)

        # 4. backward step
        # deepspeed aggregates gradients and only updates weights when accumulation_steps is reached.
        self.model_engine.backward(loss)

        # 5. optimizer step
        self.model_engine.step()

        # return loss
        train_loss = loss.item()
        return {"loss": float(train_loss)}