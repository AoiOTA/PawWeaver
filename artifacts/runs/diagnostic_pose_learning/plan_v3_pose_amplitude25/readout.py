"""Reuse exact E3 full-window metrics and full-FK support for one owned evaluation."""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
E=O.parent/'plan_v3_commanded_pose'
N=O.parent/'plan_v3_neutral_learning'
def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,E/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value
p=argparse.ArgumentParser()
p.add_argument('phase');a=p.parse_args();folder=O/a.phase
report=json.loads((folder/'report.json').read_text())
suite_root=O
bundle=N/'train500/bundle' if a.phase.startswith('initial_') else O/('checkpoint250_bundle' if a.phase.startswith('checkpoint250_') else 'train500/bundle')
support=module('e3_support','support_readout.py');support.ROOT=suite_root
s=support.build(folder,bundle)
r=module('e3_summary','summarize.py').summarize(folder)
for row in r['episodes']:
    with np.load(folder/row['case_id']/'trace.npz',allow_pickle=False) as t:
        for key in ('steady_command','pose_hold'):
            w=row['windows'][key]
            mask=np.arange(len(t['times']))*report['evaluation']['control_dt']>=w['requested_target_time_bounds_s'][0]
            assert int(mask.sum())==w['samples']
            response={'actual_mean_vx_vy_yawdot':np.column_stack((t['base_velocity_yaw'][:,:2],t['base_yaw_rate']))[mask].mean(0).tolist(),'command_mean_vx_vy_yawdot':t['velocity_commands_yaw'][mask].mean(0).tolist()} if mask.any() else None
            w['full_window_response' if w['complete_no_fall_window'] else 'executed_fragment_response']=response
for suffix,value in (('readout',r),('support',s)):
    with (O/f'{a.phase}_{suffix}.json').open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False)
print(json.dumps({'phase':a.phase,'completed':r['completed_episodes'],'cases':len(r['episodes']),'max_fk_error_m':max(c['alignment_checks']['fk_tcp_world_xyz_max_error_m'] for c in s['cases'])}))
