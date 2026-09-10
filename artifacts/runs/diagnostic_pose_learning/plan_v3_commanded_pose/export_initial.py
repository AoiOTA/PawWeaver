"""CPU export of the same fresh zero-mean Actor initialization used in training."""
from pathlib import Path
import json
import torch
from tensordict import TensorDict
from pawweaver.training_inputs import training_inputs
from pawweaver.learning import WholeBodyActor,initialize_fresh_actor
from pawweaver.bundle import export_bundle,load_bundle
from pawweaver.observations import CommandObservationSpec

OUT=Path(__file__).resolve().parent
config=json.loads((OUT/'config.json').read_text())
manifest,spec,provisional=training_inputs(Path('assets/generated/diagnostic'),diagnostic=True,
    provisional_spec=OUT.parent/'wbc_low_noise_learning/spec.json')
torch.manual_seed(0)
obs=TensorDict({'policy':torch.zeros(1,279)},[1])
actor=WholeBodyActor(obs,{'actor':['policy']},'actor',18,hidden_dims=config['actor_hidden_dims'],
    obs_normalization=True,prediction=False,velocity=False,observation_dim=279,
    distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution','init_std':.5})
initialize_fresh_actor(actor,config,spec.action_scale)
export_bundle(actor,spec,manifest,config,OUT/'initial_bundle',trained=False,
    metadata=dict(seed=0,diagnostic=True,provisional_spec=provisional))
policy,_,_=load_bundle(OUT/'initial_bundle',manifest['asset_hash'],require_trained=False,
    observation_spec=CommandObservationSpec())
probe=torch.randn(7,279)
assert torch.equal(policy(probe),torch.zeros(7,18))
torch.testing.assert_close(actor.distribution.std_param*torch.tensor(spec.action_scale),torch.tensor(config['initial_joint_std_rad']))
(OUT/'initial_export_check.json').write_text(json.dumps(dict(observation_dim=279,action_dim=18,
    zero_mean_outputs=True,joint_std_rad=config['initial_joint_std_rad'],trained=False,
    evidence_limit='Deterministic q0 is identical; later fresh train RNG follows environment construction, so random hidden weights are not claimed identical.'),indent=2)+'\n')
print('Initial zero-mean 279 bundle verified on CPU')
