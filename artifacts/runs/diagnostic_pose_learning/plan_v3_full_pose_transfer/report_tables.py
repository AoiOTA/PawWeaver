"""Render already computed before/mid/final evidence as compact review tables."""
import json
from pathlib import Path
O=Path(__file__).resolve().parent
r=json.loads((O/'paired_results.json').read_text())
text=['# Full28 transfer: actual paired evidence','',
'Each requested episode is60s. A dash in hold or command response means the full requested window was unavailable; executed fragments remain in paired_results.json. EEhold uses target20..59.98s; command response uses target8..59.98s. Support uses poststate20..60s. Checkpoint250 contains251 new training iterations and was evaluated only in MuJoCo. No intermediate PhysX values are inferred.','',
'Full duration is not task success. Original task thresholds remain unchanged; orientation acceptance is unspecified. All results use provisional hardware and trained=false.','']
def fmt(value):return '—' if value is None else f'{value:.5f}'
def pose(row):
    m=row['full_pose_hold']
    return '—' if m is None else '/'.join(fmt(m[k]['rmse']) for k in ('ee_position_m','ee_orientation_rad'))
def response(row):
    m=row['full_command_response']
    return '—' if m is None else '/'.join(fmt(x) for x in m['actual_mean_vx_vy_yawdot'])
for key,run in r['runs'].items():
    text += [f'## {key}','','| Case | Stage | Actual seconds | EE hold m/rad | Actual vx/vy/yawdot |','|---|---|---:|---|---|']
    for x in run['cases']:
        for stage,label in (('before','initial'),('checkpoint250','250'),('after','500')):
            if stage not in x:continue
            row=x[stage]
            text.append(f"| {x['case_id']} | {label} | {row['actual_duration_s']:.2f}{' fall' if row['fallen'] else ''} | {pose(row)} | {response(row)} |")
    text += ['','| Case | Stage | Foot-force counts >1N: samples | Nonfoot net>5N fraction | Mean baseZ m | Max tilt rad | Max foot-center above-thigh fraction |','|---|---|---|---:|---:|---:|---:|']
    for x in run['cases']:
        for stage,label in (('before','initial'),('checkpoint250','250'),('after','500')):
            name=stage+'_support'
            if name not in x:continue
            s=x[name];w=s['windows']['post20'];m=s['movement']['post20'];h=s['hip_height']['post20']
            counts='—' if w is None else ', '.join(f'{k}:{v}' for k,v in w['foot_net_force_gt1N_count_samples'].items() if v)
            nonfoot=None if w is None else w['robot_nonfoot_net_force_gt5N_any_fraction'];height=None if m is None else m['base_height_above_ground_m']['mean'];tilt=None if m is None else m['base_tilt_from_world_up_rad']['max'];above=None if h is None else max(v['center_above_thigh_fraction'] for v in h['feet'].values())
            text.append(f"| {x['case_id']} | {label} | {counts} | {fmt(nonfoot)} | {fmt(height)} | {fmt(tilt)} | {fmt(above)} |")
    text += ['']
text += ['Net force magnitude is not vertical load or a contact-pair identity. MuJoCo ground-pair evidence, foot heights, continuous durations and same-side thigh geometry are retained in paired_results.json and the full support files. Zero above-thigh fraction does not imply suitable support or successful tracking. The initial neutral7 reports are reused from neutral500; final PhysX neutral7 keeps its1-environment sequential layout.','']
with (O/'results.md').open('x') as f:f.write('\n'.join(text))
print('Wrote results.md from paired_results.json; no new physics.')
