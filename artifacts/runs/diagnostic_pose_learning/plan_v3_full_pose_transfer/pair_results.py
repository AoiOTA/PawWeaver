"""Compare saved transfer development and neutral retention evidence; no simulations."""
from pathlib import Path
import json
import numpy as np
O=Path(__file__).resolve().parent
N=O.parent/'plan_v3_neutral_learning'
def read(p):return json.loads(p.read_text())
def value(x):
    w=x['windows']['pose_hold'];v=x['windows']['steady_command']
    return dict(case_id=x['case_id'],completed=x['completed'],fallen=x['fallen'],actual_duration_s=x['actual_duration_s'],
        full_pose_hold=w['full_window_metrics'],fragment_pose_hold=w['executed_fragment_metrics'],
        full_command_metrics=v['full_window_metrics'],fragment_command_metrics=v['executed_fragment_metrics'],
        full_command_response=v.get('full_window_response'),fragment_command_response=v.get('executed_fragment_response'))
def support(x):
    return {k:x[k] for k in ('windows','hip_height','movement','whole_trace_nonfoot_force_longest','alignment_checks')}
def pair(before,after,br,ar,bs,ass):
    assert br['suite_sha256']==ar['suite_sha256']
    assert [v['case_id'] for v in br['episodes']]==[v['case_id'] for v in ar['episodes']]
    result=dict(before=str(before),after=str(after),suite_sha256=br['suite_sha256'],
        before_policy_sha256=br['policy_sha256'],after_policy_sha256=ar['policy_sha256'],
        before_completed=br['completed_episodes'],after_completed=ar['completed_episodes'],cases=[])
    for b,a,sb,sa in zip(br['episodes'],ar['episodes'],bs['cases'],ass['cases']):
        assert b['case_id']==a['case_id']==sb['case_id']==sa['case_id']
        row=dict(case_id=b['case_id'],before=value(b),after=value(a),before_support=support(sb),after_support=support(sa),common_windows={})
        with np.load(before/b['case_id']/'trace.npz',allow_pickle=False) as bt,np.load(after/a['case_id']/'trace.npz',allow_pickle=False) as at:
            n=min(len(bt['times']),len(at['times']));ticks=np.arange(n)
            for name,start in (('steady_command',8.),('pose_hold',20.)):
                mask=ticks>=int(round(start/.02))
                r={'samples':int(mask.sum()),'target_time_bounds_s':[float(ticks[mask][0]*.02),float(ticks[mask][-1]*.02)] if mask.any() else None,
                   'poststate_time_bounds_s':[float((ticks[mask][0]+1)*.02),float((ticks[mask][-1]+1)*.02)] if mask.any() else None,
                   'full_requested_window':bool(n==3000 and not b['fallen'] and not a['fallen'])}
                for stage,t in (('before',bt),('after',at)):
                    d={'ee_position_m':t['errors'][:n],'ee_orientation_rad':t['orientation_errors_rad'][:n],
                       'base_linear_velocity_mps':np.linalg.norm(t['base_velocity_yaw'][:n,:2]-t['velocity_commands_yaw'][:n,:2],axis=1),
                       'base_yaw_rate_radps':np.abs(t['base_yaw_rate'][:n]-t['velocity_commands_yaw'][:n,2])}
                    r[stage]={k:{'rmse':float(np.sqrt(np.mean(v[mask]**2))),'p95':float(np.quantile(v[mask],.95))} for k,v in d.items()} if mask.any() else None
                row['common_windows'][name]=r
        result['cases'].append(row)
    return result
results={}
for engine in ('mujoco','physx'):
    for suite in ('dev8','neutral7'):
        initial=O/f'initial_{engine}_dev8' if suite=='dev8' else N/f'final_{engine}_neutral7'
        final=O/f'final_{engine}_{suite}'
        ip=O/f'initial_{engine}_dev8' if suite=='dev8' else N/f'final_{engine}'
        fp=O/f'final_{engine}_{suite}'
        results[f'{engine}_{suite}']=pair(initial,final,read(Path(str(ip)+'_readout.json')),read(Path(str(fp)+'_readout.json')),read(Path(str(ip)+'_support.json')),read(Path(str(fp)+'_support.json')))
mid=read(O/'checkpoint250_mujoco_dev8_readout.json')
mid_support=read(O/'checkpoint250_mujoco_dev8_support.json')
assert mid['suite_sha256']==results['mujoco_dev8']['suite_sha256']
for row,episode,geometry in zip(results['mujoco_dev8']['cases'],mid['episodes'],mid_support['cases']):
    assert row['case_id']==episode['case_id']==geometry['case_id']
    row['checkpoint250']=value(episode)
    row['checkpoint250_support']=support(geometry)
result={'checkpoint250_scope':'251 completed new training iterations; MuJoCo dev8 only, no intermediate PhysX evaluation.', 'runs':results,'evidence_limit':'Provisional B-route train-derived development and neutral retention. Full requested windows versus common executed fragments remain separate. Nonfoot net force is not a contact-pair identity in PhysX. No full WBC, world-fixed EE-only or hardware acceptance; trained=false.'}
with (O/'paired_results.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
for key,r in results.items():
    print(key,r['before_completed'],'->',r['after_completed'])
    for x in r['cases']:
        b=x['before'];a=x['after'];m=x['common_windows']['pose_hold']
        print(x['case_id'],b['actual_duration_s'],'->',a['actual_duration_s'],'common_hold',None if m['before'] is None else [m[v]['ee_position_m']['rmse'] for v in ('before','after')])
