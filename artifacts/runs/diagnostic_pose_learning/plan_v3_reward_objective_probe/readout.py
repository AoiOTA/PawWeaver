"""Read captured actual rewards and reuse existing task/support consumers."""
from pathlib import Path
import importlib.util,json
import numpy as np
O=Path(__file__).resolve().parent
E=O.parent/'plan_v3_commanded_pose'
def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
support=load_module('existing_support',E/'support_readout.py');support.ROOT=O
summary=load_module('existing_summary',E/'summarize.py')
provenance=json.loads((O/'provenance.json').read_text())
cases=[c['case_id'] for c in json.loads((O/'suite/manifest.json').read_text())['cases']]
traces={};out={'task_frame':'base_xy_yaw_ground_z','command_source':'preset trajectory','control_dt_s':.02,'policies':{},'common_windows':[]}
def window(t,start,end):
    target_time=np.arange(len(t['times']))*.02
    mask=(target_time>=start)&(target_time<end)
    n=int(mask.sum()); expected=round((end-start)/.02)
    if not n:return {'samples':0,'expected_samples':expected,'complete_no_fall_window':False}
    fallen=t['fall_height']|t['fall_tilt'];term=t['terminal_reward'][mask]
    terms={k.removeprefix('reward_term_'):v[mask] for k,v in t.items() if k.startswith('reward_term_')}
    return {'requested_target_time_bounds_s':[start,end],'samples':n,'expected_samples':expected,
        'complete_no_fall_window':n==expected and not bool(fallen[mask].any()),
        'executed_target_time_bounds_s':[float(target_time[mask][0]),float(target_time[mask][-1])],
        'actual_reward_total':float(t['actual_reward'][mask].sum(dtype=np.float64)),
        'nonterminal_reward_total_dt_scaled':float(t['nonterminal_postclip'][mask].sum(dtype=np.float64)*.02),
        'nonterminal_mean_rate_all_recorded_steps':float(t['nonterminal_postclip'][mask].mean(dtype=np.float64)),
        'terminal_reward_total':float(term.sum()),'falls':int(fallen[mask].sum()),
        'weighted_terms_mean_rate':{k:float(v.mean(dtype=np.float64)) for k,v in terms.items()},
        'weighted_terms_total_dt_scaled':{k:float(v.sum(dtype=np.float64)*.02) for k,v in terms.items()},
        'ee_position_rmse_m':float(np.sqrt(np.mean(t['errors'][mask]**2))),
        'ee_orientation_rmse_rad':float(np.sqrt(np.mean(t['orientation_errors_rad'][mask]**2))),
        'actual_mean_vx_vy_yawdot':np.column_stack((t['base_velocity_yaw'][:,:2],t['base_yaw_rate']))[mask].mean(0).tolist(),
        'command_mean_vx_vy_yawdot':t['velocity_commands_yaw'][mask].mean(0).tolist()}
for label in ('old','new'):
    folder=O/label;bundle=Path(provenance['policies'][label]['bundle'])
    report=json.loads((folder/'report.json').read_text())
    assert report['policy_sha256']==provenance['policies'][label]['policy_sha256']
    r=summary.summarize(folder);s=support.build(folder,bundle)
    (O/f'{label}_readout.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    (O/f'{label}_support.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    rows=[];traces[label]={}
    for case in cases:
        with np.load(folder/case/'trace.npz',allow_pickle=False) as f:t={k:f[k] for k in f.files}
        traces[label][case]=t
        terms=[v for k,v in t.items() if k.startswith('reward_term_')]
        assert np.allclose(sum(terms),t['nonterminal_preclip'],rtol=1e-5,atol=1e-5)
        assert np.array_equal(t['nonterminal_preclip'],t['nonterminal_postclip'])
        assert np.allclose(t['actual_reward'],.02*t['nonterminal_postclip']+t['terminal_reward'],rtol=1e-5,atol=1e-6)
        assert np.allclose(t['position_sigma_m2'],.005) and np.allclose(t['orientation_sigma_rad'],.5)
        fallen=t['fall_height']|t['fall_tilt']; assert not fallen[:-1].any()
        rows.append({'case_id':case,'actual_steps':len(t['times']),'fallen':bool(fallen.any()),
            'fall_time_s':float(t['times'][-1]) if fallen.any() else None,
            'windows':{name:window(t,start,60) for name,start in [('full_episode',0),('command_8_60',8),('hold_20_60',20)]}})
    out['policies'][label]={'cases':rows,'max_fk_error_m':max(c['alignment_checks']['fk_tcp_world_xyz_max_error_m'] for c in s['cases']),
        'max_reward_reconstruction_error':max(float(np.abs(traces[label][c]['actual_reward']-(.02*traces[label][c]['nonterminal_postclip']+traces[label][c]['terminal_reward'])).max()) for c in cases)}
for case in cases:
    end=min(len(traces[label][case]['times']) for label in ('old','new'))*.02
    row={'case_id':case,'common_end_s':end,'windows':{}}
    for name,start in [('command',8),('hold',20)]:
        if end<=start:continue
        w={label:window(traces[label][case],start,end) for label in ('old','new')}
        w['new_minus_old_mean_rate']=w['new']['nonterminal_mean_rate_all_recorded_steps']-w['old']['nonterminal_mean_rate_all_recorded_steps']
        w['new_minus_old_term_mean_rate']={k:w['new']['weighted_terms_mean_rate'][k]-v for k,v in w['old']['weighted_terms_mean_rate'].items()}
        row['windows'][name]=w
    out['common_windows'].append(row)
(O/'reward_readout.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'policies':{k:{'max_fk_error_m':v['max_fk_error_m'],'cases':[(r['case_id'],r['actual_steps'],r['fallen']) for r in v['cases']]} for k,v in out['policies'].items()}},indent=2))
