"""Pair existing E3 before/after reports without running or changing evaluations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
SOURCE=OUT/'summarize.py'
SPEC=importlib.util.spec_from_file_location('e3_existing_summary',SOURCE)
existing=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(existing)
ENGINES={'physx':'PhysX','mujoco':'MuJoCo'}
SUITES=('dev8','test8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(folder,engine,suite_path):
    result=existing.summarize(folder)
    if result['status']!='reported':
        return result
    suite=json.loads((suite_path/'manifest.json').read_text())
    expected=[entry['case_id'] for entry in suite['cases']]
    if len(expected)!=8 or len(set(expected))!=8:
        raise ValueError(f'Expected eight distinct specified cases: {suite_path}')
    if result['engine']!=engine or result['suite_sha256']!=digest(suite_path/'manifest.json'):
        raise ValueError(f'Engine or frozen suite identity differs: {folder}')
    if [row['case_id'] for row in result['episodes']]!=expected:
        raise ValueError(f'Report must retain all eight specified cases in suite order: {folder}')
    report=json.loads((folder/'report.json').read_text())
    dt=report['evaluation']['control_dt']
    if dt!=.02:
        raise ValueError(f'Expected the actual50Hz E3 evaluation: {folder}')
    for row,original in zip(result['episodes'],report['episodes']):
        if original['policy_sha256']!=result['policy_sha256'] or original['engine']!=engine:
            raise ValueError(f'Episode policy/engine identity differs: {folder}/{row["case_id"]}')
        if original['requested_steps']!=3000 or not 0<row['actual_steps']<=3000:
            raise ValueError(f'Expected a requested60s E3 case: {folder}/{row["case_id"]}')
        if bool(original['completed'])!=(row['actual_steps']==3000):
            raise ValueError(f'Completion disagrees with actual steps: {folder}/{row["case_id"]}')
        metadata=original['trajectory']
        if metadata['pose_hold_start_s']!=20. or metadata['command_ramp_end_s']!=8.:
            raise ValueError(f'Unexpected E3 pose/command schedule: {folder}/{row["case_id"]}')
        row['source_id']=metadata['source_id']
        row['split']=metadata['split']
        row['windows'].pop('post_transient')
        for window in row['windows'].values():
            for key in ('selection_time','boundary_convention','requested_poststate_time_bounds_s'):
                window.pop(key)
        # Reuse summarize's target-k boundaries, not poststate timestamps, for response means.
        steady=row['windows']['steady_command']
        first_tick=int(np.ceil(steady['requested_target_time_bounds_s'][0]/dt-1e-9))
        mask=np.arange(row['actual_steps'])>=first_tick
        with np.load(folder/row['case_id']/'trace.npz',allow_pickle=False) as trace:
            actual=np.column_stack((trace['base_velocity_yaw'][:,:2],trace['base_yaw_rate']))
            command=trace['velocity_commands_yaw']
            if actual.shape!=(row['actual_steps'],3) or command.shape!=actual.shape or not np.isfinite(actual).all() or not np.isfinite(command).all():
                raise ValueError(f'Invalid velocity response samples: {folder}/{row["case_id"]}')
            response=dict(actual_mean_vx_vy_yawdot=actual[mask].mean(0).tolist(),
                          command_mean_vx_vy_yawdot=command[mask].mean(0).tolist()) if mask.any() else None
        if int(mask.sum())!=steady['samples']:
            raise ValueError('Response and error metrics must use the same steady-command window')
        steady['full_window_response']=response if steady['complete_no_fall_window'] else None
        steady['executed_fragment_response']=response if not steady['complete_no_fall_window'] else None
    return result


def build(root=OUT,before_prefix='initial',after_prefix='final'):
    runs={}
    policies={'before':set(),'after':set()}
    for stage,prefix in (('before',before_prefix),('after',after_prefix)):
        for key,engine in ENGINES.items():
            for suite in SUITES:
                run=read_run(root/f'{prefix}_{key}_{suite}',engine,root/suite)
                runs[stage,key,suite]=run
                if run['status']=='reported':
                    policies[stage].add(run['policy_sha256'])
    if any(len(values)>1 for values in policies.values()):
        raise ValueError('Within each stage, both engines and suites must use the same policy hash')
    pairs={}
    for key,engine in ENGINES.items():
        for suite in SUITES:
            before=runs['before',key,suite];after=runs['after',key,suite]
            reported=[run for run in (before,after) if run['status']=='reported']
            if len(reported)==2 and before['suite_sha256']!=after['suite_sha256']:
                raise ValueError(f'Before/after frozen suite differs: {engine}/{suite}')
            manifest=json.loads((root/suite/'manifest.json').read_text())
            ids=[entry['case_id'] for entry in manifest['cases']]
            if len(ids)!=8 or len(set(ids))!=8:
                raise ValueError(f'Expected eight distinct specified cases: {suite}')
            pair=dict(engine=engine,suite=suite,expected_episodes=8,
                status='paired' if len(reported)==2 else 'not_fully_run',
                suite_sha256=digest(root/suite/'manifest.json'),runs={},cases=[])
            for stage,run in (('before',before),('after',after)):
                identity={k:v for k,v in run.items() if k!='episodes'}
                if run['status']=='reported':
                    identity.update(completed_no_fall_episodes=sum(r['completed'] and not r['fallen'] for r in run['episodes']),
                        fallen_episodes=sum(r['fallen'] for r in run['episodes']),expected_denominator=8)
                pair['runs'][stage]=identity
            for i,case in enumerate(ids):
                pair['cases'].append(dict(case_id=case,
                    before=before['episodes'][i] if before['status']=='reported' else {'status':before['status']},
                    after=after['episodes'][i] if after['status']=='reported' else {'status':after['status']}))
            pairs[f'{key}_{suite}']=pair
    return dict(task_kind='velocity_ee_pose',ee_target_frame='base_xy_yaw_ground_z',
        command_source='preset_trajectory',deployment_commands='Future operator; not implemented here',
        before_prefix=before_prefix,after_prefix=after_prefix,
        policies={stage:next(iter(values)) if values else None for stage,values in policies.items()},
        method_source=str(SOURCE),method_source_sha256=digest(SOURCE),
        timing='Control target k at k*0.02 is scored against poststate(k+1)*0.02. Pose hold includes target20..59.98s; steady command includes target8..59.98s. Mean response and command use precisely the latter window.',
        response_order=['vx_yaw_mps','vy_yaw_mps','yawdot_radps'],pairs=pairs,
        evidence_limit='Provisional B-task recipe comparison only. dev8 uses training sources; test8 has independent original test sources. Failed fragments remain separate, with no mixed full-window error aggregate. No world-fixed EE-only, hardware, formal acceptance, or three-extra-input causal claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=OUT)
    parser.add_argument('--before-prefix',default='initial')
    parser.add_argument('--after-prefix',default='final')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f'Refusing overwrite: {args.output}')
    result=build(args.root,args.before_prefix,args.after_prefix)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        print(text,end='')


if __name__=='__main__':
    main()
