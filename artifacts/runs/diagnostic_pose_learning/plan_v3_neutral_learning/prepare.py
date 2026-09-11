from pathlib import Path
import json,hashlib,os
import numpy as np
import torch
from tensordict import TensorDict
from pawweaver.commanded_pose import CommandedPoseBank,load_command_suite
from pawweaver.training_inputs import training_inputs
from pawweaver.learning import WholeBodyActor,initialize_fresh_actor
from pawweaver.observations import observation_spec
O=Path(__file__).resolve().parent
E=O.parent/'plan_v3_commanded_pose'
c=json.loads((E/'config.json').read_text())
c.update(demonstrations=[p for p in c['demonstrations'] if Path(p).name.startswith('neutral_')],learn_std=True,initial_joint_std_rad=[.25]*12+[.02]*6,entropy_coef=.01)
(O/'config.json').write_text(json.dumps(c,indent=2)+'\n')
S=O/'neutral7';S.mkdir(exist_ok=True)
m=json.loads((E/'dev8/manifest.json').read_text());m['cases']=[dict(case_id=Path(p).stem,path=os.path.relpath(Path(p).resolve(),S),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in c['demonstrations']];m['scope']='Exact existing seven neutral training trajectories; train-derived development readout, not held-out test or formal acceptance'
(S/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
t=load_command_suite(S)
bank=CommandedPoseBank(128,3000,'cpu',0,c['demonstrations'],c['demonstration_group_weights']);bank.reset(torch.arange(128))
manifest,spec,provisional=training_inputs(Path('assets/generated/diagnostic'),diagnostic=True,provisional_spec=O.parent/'wbc_low_noise_learning/spec.json')
obs=TensorDict({'policy':torch.zeros(1,279)},[1])
a=WholeBodyActor(obs,{'actor':['policy']},'actor',18,hidden_dims=c['actor_hidden_dims'],obs_normalization=True,prediction=False,velocity=False,observation_dim=279,distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution','init_std':.5})
initialize_fresh_actor(a,c,spec.action_scale);a.distribution.std_param.requires_grad_(c['learn_std'])
torch.testing.assert_close(a.distribution.std_param*torch.tensor(spec.action_scale),torch.tensor(c['initial_joint_std_rad']))
assert len(t)==7 and observation_spec(c).size==279 and a.distribution.std_param.requires_grad
assert torch.equal(a.mlp[-1].weight,torch.zeros_like(a.mlp[-1].weight)) and torch.equal(a.mlp[-1].bias,torch.zeros_like(a.mlp[-1].bias))
r=dict(sampler_consumer='WholeBodyEnv -> CommandedPoseBank: config demonstrations list, not directory scan or suite manifest',group_indices=bank.demonstration_groups,group_probabilities=bank.demonstration_group_probabilities.tolist(),sample_counts=np.bincount(bank.demonstration_index,minlength=7).tolist(),cases=[v.metadata['case_id'] for v in t],initial_action_std=a.distribution.std_param.detach().tolist(),initial_joint_std_rad=c['initial_joint_std_rad'],learn_std=a.distribution.std_param.requires_grad,observation_size=279,action_size=18,config_changed_keys=[k for k in c if c[k]!=json.loads((E/'config.json').read_text())[k]],trained=False)
(O/'cpu_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
