"""RSL-RL 5 integration: single actor/critic PPO with simultaneous supervised auxiliary losses."""
import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from .policy import CausalFeatures, ExportedPolicy

class WholeBodyActor(MLPModel):
    def __init__(self,*args,prediction=False,velocity=True,**kwargs):
        # Flags are plain booleans and may be initialized before nn.Module.
        self.prediction,self.velocity = prediction,velocity
        super().__init__(*args,**kwargs)
        if self.obs_dim != 276 or self.obs_groups != ["policy"]:
            raise ValueError("Actor requires the 276-dimensional causal pose policy group; 246 position-only observations are incompatible")
        self.features = CausalFeatures(prediction,velocity)

    def _get_latent_dim(self):
        return 276+int(self.velocity)*3+int(self.prediction)*12

    def load_state_dict(self,state_dict,strict=True,assign=False):
        first=state_dict.get("mlp.0.weight")
        legacy_dim=246+int(self.velocity)*3+int(self.prediction)*12
        if first is not None and first.shape[1]==legacy_dim:
            raise ValueError("Position-only 246-observation checkpoints are incompatible with the 276 pose contract")
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def get_latent(self,obs,masks=None,hidden_state=None):
        raw = obs["policy"]
        return self.features(self.obs_normalizer(raw),raw)

    def auxiliary_loss(self,obs):
        raw = obs["policy"]
        velocity,future = self.features.estimates(self.obs_normalizer(raw),raw)
        loss = raw.sum()*0.
        if self.velocity:
            loss = loss+(velocity-obs["velocity_label"].detach()).square().mean()
        if self.prediction:
            mask = obs["future_valid"].detach()
            error = (future-obs["future_label"].detach()).square()*mask
            loss = loss+error.sum()/mask.sum().clamp_min(1.)
        return loss

    def as_jit(self):
        output = self.distribution.as_deterministic_output_module() if self.distribution else nn.Identity()
        return ExportedPolicy(self.obs_normalizer,self.features,self.mlp,output)

class AuxiliaryPPO(PPO):
    """Feed-forward, single-GPU PPO. Auxiliary labels stay in rollout storage, outside actor input."""
    def __init__(self,*args,auxiliary_coef=1.,**kwargs):
        super().__init__(*args,**kwargs)
        self.auxiliary_coef = auxiliary_coef
        if self.is_multi_gpu or self.symmetry or self.rnd or self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("PawWeaver v1 supports feed-forward single-GPU PPO without RND/symmetry")

    def update(self):
        totals = {key:0. for key in ("value","surrogate","entropy","auxiliary","kl")}
        count = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs):
            obs = batch.observations
            self.actor(obs,stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(obs)
            entropy = self.actor.output_entropy.mean()
            with torch.no_grad():
                kl = self.actor.get_kl_divergence(batch.old_distribution_params,self.actor.output_distribution_params).mean()
                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl > 2*self.desired_kl:
                        self.learning_rate = max(1e-5,self.learning_rate/1.5)
                    elif 0 < kl < self.desired_kl/2:
                        self.learning_rate = min(1e-2,self.learning_rate*1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
            advantages = batch.advantages.squeeze(-1)
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
            ratio = (log_prob-batch.old_actions_log_prob.squeeze(-1)).exp()
            surrogate = torch.maximum(-advantages*ratio,-advantages*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
            value_error = (values-batch.returns).square()
            if self.use_clipped_value_loss:
                clipped = batch.values+(values-batch.values).clamp(-self.clip_param,self.clip_param)
                value_error = torch.maximum(value_error,(clipped-batch.returns).square())
            value_loss = value_error.mean()
            auxiliary = self.actor.auxiliary_loss(obs)
            loss = surrogate+self.value_loss_coef*value_loss-self.entropy_coef*entropy+self.auxiliary_coef*auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
            self.optimizer.step()
            for key,value in zip(totals,(value_loss,surrogate,entropy,auxiliary,kl)):
                totals[key] += value.item()
            count += 1
        self.storage.clear()
        return {key:value/max(count,1) for key,value in totals.items()}
