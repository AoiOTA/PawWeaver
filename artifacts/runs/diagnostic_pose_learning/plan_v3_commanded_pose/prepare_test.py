"""Freeze eight B-task test inputs from the existing held-out geometry pool, CPU only."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from pawweaver.assets.model import RobotTree
from pawweaver.contracts import JOINT_NAMES
from pawweaver.commanded_pose import (CommandedPoseTrajectory,load_command_suite,
    COMMAND_TASK_KIND,COMMAND_TASK_FRAME,COMMAND_UNITS)

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
RUNS=OUT.parent
THRESHOLDS=dict(low_height_upper_quantile_m=.34798258688333417,
    high_height_lower_quantile_m=1.2865194978877805,lateral_abs_y_lower_quantile_m=.4881018986525837)
RECIPES={'low':[('stand',[0.,0.,0.]),('forward',[.18,0.,0.])],
    'high':[('stand',[0.,0.,0.]),('yaw',[0.,0.,-.25])],
    'lateral':[('left',[0.,.1,0.]),('arc',[.18,0.,.2])],
    'middle':[('backward',[-.12,0.,0.]),('right',[0.,-.1,0.])]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    test=OUT/'test8';summary_path=OUT/'test_generation_summary.json'
    if test.exists() or summary_path.exists():
        raise FileExistsError('Preserve the frozen B-task test inputs; refusing overwrite')
    pool_path=RUNS/'wbc_random_paths/test_geometry_pool.json'
    categories_path=RUNS/'wbc_random_paths/generation_summary.json'
    prepared_path=OUT/'generation_summary.json'
    urdf=REPO/'assets/generated/diagnostic/robot.urdf'
    categories=json.loads(categories_path.read_text())['pool_stats']['test']['categories']
    if any(categories[key]!=value for key,value in THRESHOLDS.items()):
        raise ValueError('Original held-out pool category thresholds differ')
    pool=sorted(json.loads(pool_path.read_text()),key=lambda row:row['pool_id'])
    if len({row['pool_id'] for row in pool})!=len(pool) or any(not row['pool_id'].startswith('test_') for row in pool):
        raise ValueError('Expected unique original test-pool identities')
    prepared=json.loads(prepared_path.read_text())
    start=np.asarray(prepared['initial_tcp_position_t'])
    initial_q=np.asarray(prepared['initial_tcp_quat_t'])
    start_r=Rotation.from_quat(initial_q[[1,2,3,0]])
    tree=RobotTree.load(urdf)
    times=np.arange(3001)*.02
    def ramp(end):
        x=np.clip((times-2.)/(end-2.),0.,1.)
        return x**3*(10.-15.*x+6.*x*x)
    pose_s,command_s=ramp(20.),ramp(8.)
    def eligible(row,group):
        _,y,z=row['tcp_xyz_m']
        low=z<=THRESHOLDS['low_height_upper_quantile_m']
        high=z>=THRESHOLDS['high_height_lower_quantile_m']
        lateral=abs(y)>=THRESHOLDS['lateral_abs_y_lower_quantile_m']
        return dict(low=low,high=high,lateral=lateral,middle=not (low or high or lateral))[group]
    used=set();trajectories=[];witnesses=[]
    for group,recipes in RECIPES.items():
        selected=[row for row in pool if row['pool_id'] not in used and eligible(row,group)][:2]
        if len(selected)!=2:
            raise ValueError(f'Insufficient unused test endpoints for {group}')
        for row,(command_name,command) in zip(selected,recipes):
            identity=row['pool_id'];used.add(identity)
            root=np.eye(4);root[:3,:3]=Rotation.from_euler('xyz',row['base_rpy_rad']).as_matrix()
            root[:3,3]=row['base_xyz_m']
            tcp=tree.forward(dict(zip(JOINT_NAMES,row['joint_positions_rad'])),root)['tcp']
            position_error=float(np.linalg.norm(tcp[:3,3]-row['tcp_xyz_m']))
            actual_r=Rotation.from_matrix(tcp[:3,:3])
            expected_r=Rotation.from_quat(np.asarray(row['tcp_quat_wxyz'])[[1,2,3,0]])
            orientation_error=float((actual_r.inv()*expected_r).magnitude())
            if position_error>1e-10 or orientation_error>1e-10:
                raise ValueError(f'Original test witness differs from actual URDF FK: {identity}')
            yaw=Rotation.from_euler('z',row['base_rpy_rad'][2])
            origin=np.array([*row['base_xyz_m'][:2],0.])
            target=yaw.inv().apply(tcp[:3,3]-origin)
            target_r=yaw.inv()*actual_r
            positions=start+(target-start)*pose_s[:,None]
            quats=Slerp([0.,1.],Rotation.concatenate([start_r,target_r]))(pose_s).as_quat()[:,[3,0,1,2]]
            commands=command_s[:,None]*command
            case=f'{group}_{command_name}'
            metadata=dict(split='test',case_id=case,source_id=identity,
                source_pool=str(pool_path.relative_to(REPO)),source_pool_sha256=digest(pool_path),
                test_category=group,command_source='preset_trajectory',task_kind=COMMAND_TASK_KIND,
                task_frame=COMMAND_TASK_FRAME,units=COMMAND_UNITS,hardware_validation=False,
                pose_transition_start_s=2.,pose_hold_start_s=20.,command_ramp_end_s=8.,
                scope='Independent B-task input from original test geometry; not world-fixed EE acceptance. No dynamic or path-feasibility claim.')
            trajectory=CommandedPoseTrajectory(times,positions,quats,commands,metadata)
            np.testing.assert_allclose(trajectory.positions_task[:101],np.broadcast_to(start,(101,3)),rtol=0,atol=1e-12)
            np.testing.assert_allclose(trajectory.positions_task[1000:],np.broadcast_to(target,(2001,3)),rtol=0,atol=1e-12)
            np.testing.assert_allclose(commands[:101],0.,rtol=0,atol=1e-12)
            np.testing.assert_allclose(commands[400:],np.broadcast_to(command,(2601,3)),rtol=0,atol=1e-12)
            trajectories.append(trajectory)
            witnesses.append(dict(case_id=case,category=group,source_id=identity,source_endpoint_witness=row,
                target_position_t=target.tolist(),target_quat_t=target_r.as_quat()[[3,0,1,2]].tolist(),
                velocity_command_yaw=command,fk_position_error_m=position_error,fk_orientation_error_rad=orientation_error,
                max_target_linear_speed_mps=float((np.linalg.norm(np.diff(positions,axis=0),axis=1)/.02).max())))
    entries=[]
    for trajectory in trajectories:
        path=test/(trajectory.metadata['case_id']+'.npz');trajectory.save(path)
        entries.append(dict(case_id=trajectory.metadata['case_id'],path=path.name,
            source_id=trajectory.metadata['source_id'],sha256=digest(path)))
    manifest=dict(schema_version=1,task_kind=COMMAND_TASK_KIND,task_frame=COMMAND_TASK_FRAME,
        units=COMMAND_UNITS,split='test',cases=entries,
        scope='Frozen independent commanded-pose generalization readout; no world-fixed EE-only acceptance.')
    (test/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    loaded=load_command_suite(test)
    for original,actual in zip(trajectories,loaded):
        assert actual.metadata==original.metadata
        for name in ('timestamps','positions_task','velocity_commands_yaw'):
            np.testing.assert_array_equal(getattr(actual,name),getattr(original,name))
        np.testing.assert_allclose(actual.orientations_task_wxyz,original.orientations_task_wxyz,rtol=0,atol=1e-14)
    assert len(loaded)==8 and len({t.metadata['source_id'] for t in loaded})==8
    summary=dict(count=8,split='test',frozen_at_utc=datetime.now(timezone.utc).isoformat(),duration_s=60.,control_dt_s=.02,
        task_frame=COMMAND_TASK_FRAME,ground_z=0.,thresholds=THRESHOLDS,
        selection_rule='In low/high/lateral/middle order, sort original test pool_id and take the first two not already selected. Middle excludes low, high and lateral eligibility. No training results used.',
        schedule=dict(initial_hold_end_s=2.,pose_arrival_s=20.,command_arrival_s=8.),
        initial_tcp_position_t=start.tolist(),initial_tcp_quat_t=initial_q.tolist(),endpoint_witnesses=witnesses,
        source_hashes={str(path.relative_to(REPO)):digest(path) for path in [pool_path,categories_path,prepared_path,urdf,Path(__file__)]},
        consumer_check=dict(loader='load_command_suite',cases=8,unique_original_test_sources=8,
            saved_arrays_and_metadata_match=True,initial_hold_and_final_pose_command_windows_verified=True),
        evidence_limit='CPU FK and real suite-loader validation only. Witness baseXY/yaw removed; no world far-target translation inherited. Preset velocity/EE settings differ from training; no simulation, hardware validity, formal acceptance or extra-input causal claim.')
    summary_path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(test8=str(test),selected_sources=[t.metadata['source_id'] for t in loaded],consumer_check=summary['consumer_check'])))


if __name__=='__main__':
    main()
