"""Existing E3 summary/support, bounded to two exact trajectories per combination."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;A=O.parent/'plan_v3_pose_amplitude25';E=O.parent/'plan_v3_commanded_pose'
def module(n,f):
 s=importlib.util.spec_from_file_location(n,E/f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
support=module('support','support_readout.py');support.ROOT=O
summary=module('summary','summarize.py')
def write(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result={'evidence_limit':'Four CPU episodes; provisional B route base_xy_yaw_ground_z EE target, preset velocity commands, single18 Actor. No training or unique root-cause attribution.','runs':{}}
for name in ('oldW_oldN','newW_newN','newW_oldN','oldW_newN'):
 if name in ('oldW_oldN','newW_newN'):
  phase='initial_mujoco_dev11' if name=='oldW_oldN' else 'final_mujoco_dev11'
  folder=A/phase;r=json.loads((A/(phase+'_readout.json')).read_text());s=json.loads((A/(phase+'_support.json')).read_text())
  r['reused_source_readout_sha256']=sha(A/(phase+'_readout.json'));s['reused_source_support_sha256']=sha(A/(phase+'_support.json'))
  r['episodes']=[v for v in r['episodes'] if v['case_id'] in ('neutral_forward','neutral_backward')];s['cases']=[v for v in s['cases'] if v['case_id'] in ('neutral_forward','neutral_backward')]
  r['completed_episodes']=sum(v['completed'] for v in r['episodes'])
 else:
  folder=O/name;r=summary.summarize(folder);s=support.build(folder,O/(name+'_bundle'))
  for row in r['episodes']:
   with np.load(folder/row['case_id']/'trace.npz',allow_pickle=False) as t:
    for key in ('steady_command','pose_hold'):
     w=row['windows'][key];mask=np.arange(len(t['times']))*.02>=w['requested_target_time_bounds_s'][0]
     assert int(mask.sum())==w['samples']
     response={'actual_mean_vx_vy_yawdot':np.column_stack((t['base_velocity_yaw'][:,:2],t['base_yaw_rate']))[mask].mean(0).tolist(),'command_mean_vx_vy_yawdot':t['velocity_commands_yaw'][mask].mean(0).tolist()} if mask.any() else None
     w['full_window_response' if w['complete_no_fall_window'] else 'executed_fragment_response']=response
 write(O/(name+'_readout.json'),r);write(O/(name+'_support.json'),s)
 result['runs'][name]={'readout':r,'support':s}
write(O/'comparison.json',result)
print(json.dumps({k:[{'case':v['case_id'],'duration':v['actual_duration_s'],'fallen':v['fallen'],'response':v['windows']['steady_command'].get('full_window_response',v['windows']['steady_command'].get('executed_fragment_response')),'hold':v['windows']['pose_hold']['full_window_metrics']} for v in r['readout']['episodes']] for k,r in result['runs'].items()}))
