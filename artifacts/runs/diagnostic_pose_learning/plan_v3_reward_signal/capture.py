"""Bounded frozen-policy reward capture through the actual PhysX reward consumer."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('artifacts/runs/diagnostic_pose_learning')
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, default=ROOT/'wbc_low_noise_learning/train1000/checkpoint_000250.pt')
parser.add_argument('--num-envs', type=int, default=128)
parser.add_argument('--seconds', type=float, default=60.)
args = parser.parse_args()
torch.set_num_threads(1)
saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
config = dict(saved['metadata']['config'], reward_diagnostics=True)
spec = ROOT/'wbc_low_noise_learning/spec.json'
args.output.mkdir(parents=True, exist_ok=False)
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, device='cuda:0')
try:
    import pawweaver.isaac_env as module
    from pawweaver.learning import WholeBodyActor
    from pawweaver.math import quat_angle_error
    from pawweaver.training_inputs import check_training_identity
    env = module.WholeBodyEnv(Path('assets/generated/diagnostic'), config, args.num_envs,
        'cuda:0', 0, diagnostic=True, provisional_spec=spec)
    check_training_identity(saved['metadata'], dict(asset_hash=env.manifest['asset_hash'],
        diagnostic=True, provisional_spec=json.loads(spec.read_text())))
    obs = env.get_observations()
    actor = WholeBodyActor(obs, {'actor':['policy'], 'critic':['critic']}, 'actor', 18,
        hidden_dims=config['actor_hidden_dims'], obs_normalization=True,
        prediction=config['trajectory_prediction'], velocity=config['velocity_estimation'],
        leg_mean_transform=config.get('leg_mean_transform','identity'),
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution','init_std':.5})
    actor.load_state_dict(saved['algorithm']['actor_state_dict'], strict=True)
    actor.to('cuda:0').eval()
    before = {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    env.umi_pose_reward.load_state_dict(saved['umi_pose_reward'])
    env.umi_pose_reward.on_reset()
    initial_reward_state = env.umi_pose_reward.state_dict()
    # The tap observes actual input before auto-reset; no reward or state modification.
    original = module.reward_terms
    latest = {}
    def tap(**kwargs):
        state = env.state()
        goal = env.reference.current(env.episode_length_buf)-env.scene.env_origins
        latest.clear()
        latest.update(position_error=kwargs['error'].norm(dim=-1).detach().clone(),
            orientation_error=kwargs['orientation_error'].detach().clone(),
            goal_xyz=goal.detach().clone(), episode_age_s=env.episode_length_buf*.02,
            no_positive_foot_z=(kwargs['foot_force_z'].clamp_min(0).sum(-1)<=1e-8),
            nonfoot_net_contact=kwargs['collision'].clone(), fallen=kwargs['fallen'].clone(),
            action_clip_fraction=(actions.abs()>1).float().mean(-1),
            effective_target=env.pd.target.detach().clone())
        return original(**kwargs)
    module.reward_terms = tap
    rows = {}
    falls = resets = saturation = torque_samples = 0
    start = time.perf_counter()
    with torch.no_grad():
        for step in range(round(args.seconds/.02)):
            actions = actor(obs, stochastic_output=True)
            obs, reward, done, extras = env.step(actions)
            d = extras['reward_diagnostics']
            values = {**d['weighted_nonterminal_terms'],
                'preclip':d['nonterminal_preclip'], 'postclip':d['nonterminal_postclip'],
                'returned_reward':reward, **latest}
            for key,value in values.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(key)
                rows.setdefault(key,[]).append(value.detach().cpu().numpy().copy())
            expected = d['nonterminal_postclip']*.02+config['reward_weights']['termination']*latest['fallen']
            torch.testing.assert_close(reward,expected,rtol=0,atol=0)
            if any(not torch.isfinite(value).all() for value in obs.values()):
                raise FloatingPointError('observations')
            falls += extras['falls']; resets += extras['resets']
            saturation += extras['torque_saturated']; torque_samples += extras['torque_samples']
            if (step+1)%500==0:
                print(json.dumps({'seconds':(step+1)*.02,'falls':falls,'wall_s':time.perf_counter()-start}),flush=True)
    module.reward_terms = original
    for key,value in actor.state_dict().items():
        assert torch.equal(value.detach().cpu(), before[key]), key
    data = {k:np.concatenate(v,axis=0) for k,v in rows.items()}
    np.savez_compressed(args.output/'reward_samples.npz', **data)
    categories = json.loads((ROOT/'wbc_random_paths/generation_summary.json').read_text())['pool_stats']['train']['categories']
    goal = data['goal_xyz']
    masks = {'all':np.ones(len(goal),dtype=bool),
        'low':goal[:,2]<=categories['low_height_upper_quantile_m'],
        'high':goal[:,2]>=categories['high_height_lower_quantile_m'],
        'lateral':np.abs(goal[:,1])>=categories['lateral_abs_y_lower_quantile_m'],
        'far_radius_ge_1p7':np.linalg.norm(goal[:,:2],axis=-1)>=1.7,
        'no_positive_foot_z':data['no_positive_foot_z'],
        'negative_nonterminal':data['preclip']<0,
        'nonnegative_nonterminal':data['preclip']>=0}
    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        return dict(mean=float(np.mean(values)),p05=float(np.quantile(values,.05)),
            median=float(np.median(values)),p95=float(np.quantile(values,.95)))
    groups = {}
    for name,mask in masks.items():
        groups[name] = {'samples':int(mask.sum())}
        if mask.any():
            groups[name].update({k:stats(v[mask]) for k,v in data.items() if v.ndim==1})
            groups[name]['clipped_fraction'] = float((data['preclip'][mask]<0).mean())
    result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        config=config, seed=0, num_envs=args.num_envs, global_seconds=args.seconds,
        transitions=len(goal), optimizer_updates=0, actor_and_normalizer_unchanged=True,
        sampling='Frozen actor, stochastic actions from saved std, natural training GoalBank; episodes auto-reset.',
        reward_state_initial=initial_reward_state,reward_state_final=env.umi_pose_reward.state_dict(),
        grouping='Overlapping instantaneous target position bins using original generation thresholds; not task success classes or exact source IDs.',
        categories=categories, groups=groups, falls=falls, resets=resets,
        torque_saturation_fraction=saturation/torque_samples,
        effective_joint_target_min_rad=data['effective_target'].min(0).tolist(),
        effective_joint_target_max_rad=data['effective_target'].max(0).tolist(),
        wall_seconds=time.perf_counter()-start,
        evidence_limit='New provisional-model frozen-policy mechanism evidence, not training, causal failure attribution, hardware validity or acceptance.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('transitions','falls','resets','wall_seconds')}),flush=True)
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
