"""Reuse E3 full-window error and support consumers for neutral7 learning readout."""
from pathlib import Path
import argparse,importlib.util,json,hashlib
import numpy as np
O=Path(__file__).resolve().parent
E=O.parent/'plan_v3_commanded_pose'
s=importlib.util.spec_from_file_location('e3_summary',E/'summarize.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=m.summarize(a.run)
assert r['status']=='reported'
suite=json.loads((O/'neutral7/manifest.json').read_text())
assert r['suite_sha256']==hashlib.sha256((O/'neutral7/manifest.json').read_bytes()).hexdigest()
assert [v['case_id'] for v in r['episodes']]==[v['case_id'] for v in suite['cases']]
for row in r['episodes']:
    w=row['windows']['steady_command']
    with np.load(a.run/row['case_id']/'trace.npz',allow_pickle=False) as t:
        mask=np.arange(len(t['times']))*.02>=w['requested_target_time_bounds_s'][0]
        response=dict(actual_mean_vx_vy_yawdot=np.column_stack((t['base_velocity_yaw'][:,:2],t['base_yaw_rate']))[mask].mean(0).tolist(),command_mean_vx_vy_yawdot=t['velocity_commands_yaw'][mask].mean(0).tolist()) if mask.any() else None
    assert int(mask.sum())==w['samples']
    w['full_window_response']=response if w['complete_no_fall_window'] else None
    w['executed_fragment_response']=response if not w['complete_no_fall_window'] else None
r['scope']='Exact neutral train-derived cases, provisional hardware, trained=false; full windows separate from fallen fragments; no full WBC or world-fixed EE-only acceptance'
a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({'completed':r['completed_episodes'],'cases':len(r['episodes']),'output':str(a.output)}))
