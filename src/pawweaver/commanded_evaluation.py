"""Metrics for the explicitly commanded yaw-frame EE comparison task."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluation import completion_status
from .task import episode_metrics


def commanded_metrics(rows, requested_steps, fallen):
    result=episode_metrics(rows['times'],rows['errors'],rows['base'],rows['torques'],
        rows['velocities'],fallen,orientation_errors=rows['orientation_errors_rad'])
    # The A route's static world-reaching success rule is not a B-task criterion.
    result.pop('reached');result.pop('reach_time_s')
    result.update(completion_status(requested_steps,len(rows['times']),fallen))
    times=np.asarray(rows['times'])
    commands=np.asarray(rows['velocity_commands_yaw'])
    actual=np.asarray(rows['base_velocity_yaw'])
    rate=np.asarray(rows['base_yaw_rate'])
    if commands.shape!=(len(times),3) or actual.shape!=(len(times),3) or rate.shape!=(len(times),):
        raise ValueError('Command metrics require one XYZ velocity and yaw-rate sample per control tick')
    if not all(np.isfinite(value).all() for value in (commands,actual,rate)):
        raise ValueError('Nonfinite commanded velocity evidence')
    mask=times>=2.
    for prefix,error in (
        ('base_linear_command',np.linalg.norm(actual[:,:2]-commands[:,:2],axis=1)),
        ('base_yaw_rate_command',np.abs(rate-commands[:,2]))):
        units='mps' if prefix=='base_linear_command' else 'radps'
        selected=error[mask]
        result[f'{prefix}_rmse_{units}']=float(np.sqrt(np.mean(selected**2))) if len(selected) else None
        result[f'{prefix}_p95_{units}']=float(np.quantile(selected,.95)) if len(selected) else None
    result.update(task_kind='velocity_ee_pose',elapsed_seconds=float(times[-1]),
        metric_scope='Post2 EE errors use the moving yaw-only task frame; command errors use yaw-frame XY velocity and wrapped finite-difference yaw rate.',
        pose_acceptance_passed=None,command_acceptance_passed=None)
    return result


def save_command_run(output,suite,asset_manifest,bundle,seed,engine,results,*,evaluation):
    from .commanded_pose import COMMAND_TASK_FRAME,COMMAND_UNITS
    suite=Path(suite);output=Path(output)
    manifest=json.loads((suite/'manifest.json').read_text())
    expected=[entry['case_id'] for entry in manifest['cases']]
    if [row['trajectory']['case_id'] for row in results]!=expected:
        raise ValueError('Command evaluation must retain every specified case in suite order')
    if any(row['task_kind']!='velocity_ee_pose' for row in results):
        raise ValueError('Command report cannot contain world-only task results')
    completed=[r for r in results if r['completed'] and not r['fallen']]
    def mean(key):
        return float(np.mean([r[key] for r in completed])) if completed and all(r[key] is not None for r in completed) else None
    summary=dict(episodes=len(results),completed_episodes=sum(r['completed'] for r in results),
        completion_rate=float(np.mean([r['completed'] for r in results])) if results else None,
        no_fall_rate=float(np.mean([not r['fallen'] for r in results])) if results else None,
        completed_no_fall_episodes=len(completed),
        completed_mean_metrics={key:mean(key) for key in ('rmse_m','p95_m','orientation_rmse_rad','orientation_p95_rad',
            'base_linear_command_rmse_mps','base_linear_command_p95_mps','base_yaw_rate_command_rmse_radps','base_yaw_rate_command_p95_radps')},
        acceptance_passed=None)
    report=dict(task_kind='velocity_ee_pose',ee_target_frame=COMMAND_TASK_FRAME,units=COMMAND_UNITS,
        command_source='preset_trajectory',trajectory_command_sources=sorted({r['trajectory']['command_source'] for r in results}),
        deployment_command_source='Future operator commands; not implemented by this simulation evaluator.',
        engine=engine,seed=seed,asset_hash=asset_manifest['asset_hash'],policy_sha256=bundle['policy_sha256'],
        suite_sha256=hashlib.sha256((suite/'manifest.json').read_bytes()).hexdigest(),case_ids=expected,
        diagnostic=bool(bundle.get('training_metadata',{}).get('diagnostic',False)),
        summary=summary,episodes=results,evaluation=evaluation,
        evidence_limit='Velocity plus moving yaw-frame EE recipe comparison; not world-fixed EE-only success, hardware validity, or causal attribution to three extra inputs alone. No new acceptance thresholds.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
