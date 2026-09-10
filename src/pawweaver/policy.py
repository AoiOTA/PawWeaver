"""Causal whole-body policy features, independently exportable without Isaac Lab or RSL-RL."""
import copy
import torch
from torch import nn

class LegSoftsignMean(nn.Module):
    """Bound only the twelve leg means; Gaussian samples remain unbounded."""
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        legs = logits[:, :12]
        return torch.cat((legs / (1. + legs.abs()), logits[:, 12:]), dim=-1)

def mlp(in_dim,out_dim,hidden):
    layers = []
    for width in hidden:
        layers += [nn.Linear(in_dim,width),nn.ELU()]
        in_dim = width
    return nn.Sequential(*layers,nn.Linear(in_dim,out_dim))

class CausalFeatures(nn.Module):
    def __init__(self,prediction=False,velocity=True,observation_dim=276):
        super().__init__()
        if observation_dim not in (276,279):
            raise ValueError("Expected an explicit 276 or 279 observation dimension")
        if observation_dim==279 and (prediction or velocity):
            raise ValueError("279 command observations require prediction=false and velocity=false")
        self.observation_dim=observation_dim
        self.prediction,self.velocity = prediction,velocity
        self.velocity_net = mlp(210,3,[128,64])
        self.future_net = mlp(12,12,[64,64])
        self.output_dim = observation_dim+int(velocity)*3+int(prediction)*12

    def estimates(self,normalized:torch.Tensor,raw:torch.Tensor):
        if (normalized.dim()!=2 or raw.dim()!=2 or normalized.size(-1)!=self.observation_dim
                or raw.size(-1)!=self.observation_dim):
            raise ValueError("Causal feature observation dimension differs from its contract")
        v = self.velocity_net(normalized[:,:210])
        goals = raw[:,231:243].reshape(-1,4,3)
        relative_history = (goals-goals[:,-1:]).flatten(1)
        future_delta = self.future_net(relative_history)*raw[:,243:244]
        return v,future_delta

    def forward(self,normalized:torch.Tensor,raw:torch.Tensor):
        v,future = self.estimates(normalized,raw)
        parts = [normalized]
        if self.velocity:
            parts.append(v)
        if self.prediction:
            parts.append(future)
        return torch.cat(parts,dim=-1)

class ExportedPolicy(nn.Module):
    def __init__(self,normalizer,features,network,deterministic_output,observation_dim=276):
        super().__init__()
        if observation_dim!=features.observation_dim:
            raise ValueError("Export observation dimension differs from causal features")
        self.observation_dim=observation_dim
        self.normalizer = copy.deepcopy(normalizer)
        self.features = copy.deepcopy(features)
        self.network = copy.deepcopy(network)
        self.output = copy.deepcopy(deterministic_output)

    def forward(self,obs:torch.Tensor):
        if obs.dim()!=2 or obs.size(-1)!=self.observation_dim:
            raise ValueError("Expected observation dimension "+str(self.observation_dim)+"; incompatible pose contract")
        normalized = self.normalizer(obs)
        return self.output(self.network(self.features(normalized,obs)))
