"""Causal whole-body policy features, independently exportable without Isaac Lab or RSL-RL."""
import copy
import torch
from torch import nn

def mlp(in_dim,out_dim,hidden):
    layers = []
    for width in hidden:
        layers += [nn.Linear(in_dim,width),nn.ELU()]
        in_dim = width
    return nn.Sequential(*layers,nn.Linear(in_dim,out_dim))

class CausalFeatures(nn.Module):
    def __init__(self,prediction=False,velocity=True):
        super().__init__()
        self.prediction,self.velocity = prediction,velocity
        self.velocity_net = mlp(210,3,[128,64])
        self.future_net = mlp(12,12,[64,64])
        self.output_dim = 246+int(velocity)*3+int(prediction)*12

    def estimates(self,normalized:torch.Tensor,raw:torch.Tensor):
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
    def __init__(self,normalizer,features,network,deterministic_output):
        super().__init__()
        self.normalizer = copy.deepcopy(normalizer)
        self.features = copy.deepcopy(features)
        self.network = copy.deepcopy(network)
        self.output = copy.deepcopy(deterministic_output)

    def forward(self,obs:torch.Tensor):
        normalized = self.normalizer(obs)
        return self.output(self.network(self.features(normalized,obs)))
