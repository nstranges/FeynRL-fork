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

    def compute_loss(self, logits, y, mask):
        '''
         This functions implements \sum_{i=1}^{N} log p(y_i|x_i)
         y is target label [B, T -1]
         logits is model prediction [B, T -1, vocab_size]
        '''
        B, T, vocab_size = logits.shape

        # flatten logits across batch and seq_len before computing loss
        # so logits is [B * (T -1), vocab_size]
        logits = logits.view(-1, vocab_size)
        # flatten y as well:  [B, T -1] -->  [B * (T -1)]
        y = y.view(-1)

        # per token loss
        per_token_loss = self.loss_fn(logits, y)

        # We need to apply mask to loss to remove any things 
        # which should not be considered in loss (e.g., padding tokens)
        masked_loss = per_token_loss * mask

        # To avoid gardient accumulation error caused by loss.mean(),
        # we use sum of loss instead but play with learning rate to account for this.
        loss = masked_loss.sum()
        return loss

    def forward(self, batch):
        '''
            This function implements a single forward pass for current batch.
            It returns logits, y, and mask
            logits: [B, T -1, vocab_size]
            y: [B, T -1]
            mask: [B, T -1]
        '''
        # batch is a dictionary, so we want to extract things we need from it
        # input_ids/att_mask/pos_ids are [B, T]
        input_ids = batch['input_ids'].to(self.device)
        att_mask  = batch['attention_mask'].to(self.device)
        pos_ids   = batch['position_ids'].to(self.device)
        rewards   = batch['rewards'].to(self.device)
        masks     = batch['masks'].to(self.device)
        vals      = batch['vals'].to(self.device)
        last_val  = batch['last_val'].to(self.device)

        # feed data to model
        output = self.model_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, vocab_size]
        every_token_logits = output.logits

        # label is input_ids shifted by one
        # so y is [B, T -1]
        y = input_ids[:, 1:].contiguous()
        # as a result we need to remove last token from every_token_logits
        # so logits is [B, T -1, vocab_size]
        logits = every_token_logits[:, :-1, :].contiguous()

        # last input token is EOS, so we ignore loss theal  e. so mask is [B, T -1]
        mask = batch['loss_mask'][:, :-1].contiguous()

        return logits, y, mask

    def eval_step(self, data):
        '''
           This function implements a single validation step per rank/gpu.
        '''
        # we need to split data into micro batches
        micro_batches = data.split(self.micro_batch_size_per_gpu)
        num_of_micro_batches = len(micro_batches)
        val_loss = 0
        self.model_engine.eval()
        with torch.no_grad():
            for batch in micro_batches:
                ######## 
                # forward pass per gpu/rank
                ########
                logits, y, mask = self.forward(batch)

                ######## 
                # compute loss pass
                ########
                loss = self.compute_loss(logits=logits, y=y, mask=mask)

                val_loss += loss.item()/num_of_micro_batches

        return val_loss

    def train_step(self, data):
        '''
           This function implements a single training step per rank/gpu.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu. 
           So we need to split data into micro batches.  
        '''
        # we need to split data into micro batches
        micro_batches = data.split(self.micro_batch_size_per_gpu)
        num_of_micro_batches = len(micro_batches)
        step_loss = 0
        # make sure model is in training mode
        self.model_engine.train()
        for batch in micro_batches:
            ######## 
            # forward pass per gpu/rank
            ########
            logits, y, mask = self.forward(batch)

            ######## 
            # compute loss pass
            ########
            loss = self.compute_loss(logits=logits, y=y, mask=mask)

            ########    
            # backward step
            ########
            self.optimizer.zero_grad()
            loss = loss / num_of_micro_batches            
            loss.backward()
            step_loss += loss.item()

        self.model_engine.step()
        return step_loss