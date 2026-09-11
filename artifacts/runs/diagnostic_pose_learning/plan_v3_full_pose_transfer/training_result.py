"""Summarize the actually exited bounded transfer using its saved training evidence."""
from pathlib import Path
import hashlib,json,math
O=Path(__file__).resolve().parent
p=O/'train500'
code=int((O/'train500_exitcode.txt').read_text())
rows=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
run=json.loads((p/'run.json').read_text())
assert rows and [r['iteration'] for r in rows]==list(range(len(rows)))
def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,list):return all(finite(x) for x in value)
    return True
n=len(rows);transitions=n*run['num_envs']*run['config']['rollout_steps']
progress=run['training_progress']
result={'process_exitcode':code,'requested_iterations':500,'recorded_iterations':n,
    'completed_transitions':transitions,'optimizer_steps':sum(r['optimizer_steps'] for r in rows),
    'budget_completed':code==0 and n==500,'finite_checks_all_passed':all(r['finite_checks_passed'] for r in rows),
    'all_recorded_numeric_fields_finite':finite(rows),'training_progress':progress,
    'falls':sum(r['falls'] for r in rows),'last100_falls':sum(r['falls'] for r in rows[-100:]),
    'resets':sum(r['resets'] for r in rows),'torque_saturated':sum(r['torque_saturated'] for r in rows),
    'torque_samples':sum(r['torque_samples'] for r in rows),
    'stand_transitions':sum(r['demonstration_stand_transitions'] for r in rows),
    'move_transitions':sum(r['demonstration_move_transitions'] for r in rows),
    'initialize_from':run['initialize_from'],'seed':run['seed'],'config':run['config'],
    'source_sha256':run['source_sha256'],'asset_hash':run['asset_hash'],'observation':run['observation'],
    'checkpoints':{},'trend':[],
    'scope':'Actual fixed500 transfer execution; mixed training metrics do not establish task learning or causal attribution. Diagnostic provisional hardware, trained=false.'}
assert result['stand_transitions']+result['move_transitions']==transitions
for i in (0,25,50,100,150,200,250,300,350,400,450,499):
    if i<n:result['trend'].append({k:rows[i][k] for k in ('iteration','learning_rate','leg_action_std_mean','arm_action_std_mean','tracking_error_mean_m','orientation_error_mean_rad','reward_weighted_umi_pose_mean','reward_weighted_base_linear_velocity_tracking_mean','reward_weighted_base_yaw_rate_tracking_mean','umi_position_sigma_m2','umi_orientation_sigma_rad','falls')})
for name in ('checkpoint_000250.pt',f'checkpoint_{n-1:06d}.pt'):
    f=p/name
    if f.is_file():result['checkpoints'][name]=hashlib.sha256(f.read_bytes()).hexdigest()
with (O/'training_summary.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ('process_exitcode','recorded_iterations','completed_transitions','optimizer_steps','budget_completed','finite_checks_all_passed','falls','last100_falls')}))
