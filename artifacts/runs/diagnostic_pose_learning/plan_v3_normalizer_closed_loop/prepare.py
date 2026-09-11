"""Bounded CPU-only diagnostic normalizer exchange; original artifacts are read-only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from pawweaver.learning import WholeBodyActor
from pawweaver.bundle import load_bundle
from pawweaver.contracts import canonical_hash
from pawweaver.observations import observation_spec
O=Path(__file__).resolve().parent; P=O.parent; A=P/'plan_v3_pose_amplitude25'; N=P/'plan_v3_neutral_learning'
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
sources={'old':(N/'train500/bundle',N/'train500/checkpoint_000499.pt'),'new':(A/'train500/bundle',A/'train500/checkpoint_000325.pt')}
original_hashes={str(p):sha(p) for b,c in sources.values() for p in (b/'policy.pt',b/'manifest.json',c)}
policies={k:torch.jit.load(str(b/'policy.pt'),map_location='cpu').eval() for k,(b,c) in sources.items()}
checkpoints={k:torch.load(c,map_location='cpu',weights_only=False) for k,(b,c) in sources.items()}
obs=torch.from_numpy(np.concatenate([np.load(A/'initial_mujoco_dev11'/case/'trace.npz')['observations'][::100] for case in ('neutral_forward','neutral_backward')])).float()
keys=['normalizer._mean','normalizer._var','normalizer._std','normalizer.count']
results={}
for name,w,n in [('newW_oldN','new','old'),('oldW_newN','old','new')]:
    out=O/(name+'_bundle');out.mkdir(exist_ok=False)
    policy=torch.jit.load(str(sources[w][0]/'policy.pt'),map_location='cpu').eval()
    state=policy.state_dict();donor=policies[n].state_dict()
    for key in keys:state[key]=donor[key].clone()
    policy.load_state_dict(state,strict=True)
    policy.save(str(out/'policy.pt'))
    manifest=json.loads((sources[w][0]/'manifest.json').read_text());manifest.pop('bundle_hash')
    provenance={'derived_diagnostic':True,'trained':False,'purpose':'CPU closed-loop normalizer exchange, no training; not an original checkpoint export','weights_source':str(sources[w][0]),'normalizer_source':str(sources[n][0]),'source_hashes':original_hashes,'exchanged_buffers':keys,'source_counts':{k:int(v.normalizer.count.item()) for k,v in policies.items()}}
    manifest['derived_diagnostic']=provenance;manifest['trained']=False;manifest['policy_sha256']=sha(out/'policy.pt');manifest['bundle_hash']=canonical_hash(manifest);write(out/'manifest.json',manifest)
    loaded,_,_=load_bundle(out,manifest['asset_hash'],require_trained=False,observation_spec=observation_spec(manifest['training_config']))
    verified=[]
    for key,val in loaded.state_dict().items():
        assert torch.equal(val,(donor if key in keys else policies[w].state_dict())[key]),key
        verified.append(key)
    cp=checkpoints[w];md=cp['metadata'];cfg=md['config'];spec=observation_spec(cfg)
    actor=WholeBodyActor(TensorDict({'policy':torch.zeros(1,spec.size)},batch_size=[1]),{'actor':['policy']},'actor',18,hidden_dims=cfg['actor_hidden_dims'],obs_normalization=True,observation_dim=spec.size,prediction=cfg['trajectory_prediction'],velocity=cfg['velocity_estimation'],leg_mean_transform=cfg.get('leg_mean_transform','identity'),distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution','init_std':.5})
    ast=copy.deepcopy(cp['algorithm']['actor_state_dict'])
    for key in keys:
        akey=key.replace('normalizer.','obs_normalizer.')
        ast[akey]=checkpoints[n]['algorithm']['actor_state_dict'][akey].clone()
    actor.load_state_dict(ast,strict=True);actor.eval()
    with torch.no_grad():
        actual=loaded(obs); expected=actor.as_jit().eval()(obs)
    error=float((actual-expected).abs().max());assert error<1e-6,error
    results[name]={'action_max_abs_error_vs_checkpoint_swap':error,'saved_old_observation_count':len(obs),'all_deployed_tensors_verified':verified,'provenance':provenance}
subset=O/'dev2';subset.mkdir(exist_ok=False)
m=json.loads((A/'dev11/manifest.json').read_text());m['cases']=[c for c in m['cases'] if c['case_id'] in ('neutral_forward','neutral_backward')]
for c in m['cases']:
    source=(A/'dev11'/c['path']).resolve();dest=subset/source.name
    shutil.copyfile(source,dest);assert sha(dest)==c['sha256'];c['path']=dest.name
write(subset/'manifest.json',m)
assert all(sha(Path(p))==h for p,h in original_hashes.items())
write(O/'transformation_validation.json',results)
print(json.dumps({k:{'max_action_error':v['action_max_abs_error_vs_checkpoint_swap'],'observations':v['saved_old_observation_count']} for k,v in results.items()}))
