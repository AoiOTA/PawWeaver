"""Command-route CPU contracts with synthetic states; no simulator is started."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pawweaver.commanded_evaluation import commanded_metrics,save_command_run
from pawweaver.commanded_pose import (CommandedPoseTrajectory,COMMAND_TASK_FRAME,COMMAND_UNITS,
                                     task_pose_to_world,yaw_linear_velocity,yaw_rate)
from pawweaver.contracts import RobotState
from pawweaver.math import rpy_quat
from pawweaver.mujoco_runtime import MujocoRunner,CommandedMujocoRunner
from pawweaver.observations import CommandObservationBuilder

SPEC=importlib.util.spec_from_file_location('command_isaac',Path(__file__).resolve().parents[1]/'scripts/evaluate_commanded_isaac.py')
isaac=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(isaac)


def trajectory():
    return CommandedPoseTrajectory(np.array([0.,.02,.04]),np.array([[1.,0.,.6],[2.,0.,.6],[3.,0.,.6]]),
        np.tile([1.,0.,0.,0.],(3,1)),np.array([[.1,0.,.2],[.2,0.,.4],[.3,0.,.6]]),
        dict(task_kind='velocity_ee_pose',task_frame=COMMAND_TASK_FRAME,units=COMMAND_UNITS,
             split='test',case_id='case',source_id='synthetic_contract',command_source='preset_trajectory'))


def state(x=0.,yaw=0.,height=.5):
    return RobotState(joint_pos=torch.zeros(1,18),joint_vel=torch.zeros(1,18),
        base_pos_w=torch.tensor([[x,0.,height]]),base_quat_w=torch.tensor(rpy_quat([0.,0.,yaw]),dtype=torch.float32)[None],
        base_ang_vel_b=torch.zeros(1,3),tcp_pos_w=torch.tensor([[x+1.,0.,.6]]),
        tcp_quat_w=torch.tensor([[1.,0.,0.,0.]]),base_lin_vel_b=torch.tensor([[.1,0.,0.]]))


def metric_rows(times):
    n=len(times)
    return dict(times=times,errors=np.full(n,.01),base=np.zeros((n,3)),torques=np.zeros((n,18)),
        velocities=np.zeros((n,18)),orientation_errors_rad=np.zeros(n),velocity_commands_yaw=np.zeros((n,3)),
        base_velocity_yaw=np.tile([3.,4.,50.],(n,1)),base_yaw_rate=np.full(n,.5))


def test_command_metrics_separate_xy_and_yaw_and_keep_post2_empty():
    rows=metric_rows([.02,1.,2.,2.02])
    result=commanded_metrics(rows,4,False)
    assert result['base_linear_command_rmse_mps']==5.
    assert result['base_yaw_rate_command_p95_radps']==.5
    assert result['completed'] and result['task_kind']=='velocity_ee_pose'
    assert 'reached' not in result and result['pose_acceptance_passed'] is None
    short=commanded_metrics(metric_rows([.02,.04]),4,False)
    assert not short['completed'] and short['termination_reason']=='incomplete'
    assert short['rmse_m'] is None and short['base_linear_command_rmse_mps'] is None
    assert short['executed_fragment_rmse_m']==.01
    fallen=commanded_metrics(rows,4,True)
    assert fallen['termination_reason']=='fall' and fallen['fallen']
    rows['base_yaw_rate'][0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        commanded_metrics(rows,4,False)


def test_command_report_does_not_apply_world_success_or_drop_failure(tmp_path):
    suite=tmp_path/'suite';suite.mkdir()
    (suite/'manifest.json').write_text(json.dumps({'cases':[{'case_id':'complete'},{'case_id':'failed'}]}))
    complete=commanded_metrics(metric_rows([2.,3.]),2,False)
    failed=commanded_metrics(metric_rows([.02]),2,True)
    for row,name in [(complete,'complete'),(failed,'failed')]:
        row['trajectory']=dict(trajectory().metadata,case_id=name)
    report=save_command_run(tmp_path/'report',suite,{'asset_hash':'asset'},{'policy_sha256':'policy'},0,'MuJoCo',[complete,failed],evaluation={})
    assert report['summary']['completion_rate']==.5
    assert report['summary']['no_fall_rate']==.5
    assert report['summary']['completed_mean_metrics']['base_linear_command_rmse_mps']==5.
    assert report['summary']['acceptance_passed'] is None
    assert 'tracking_pass_rate' not in report['summary']
    assert report['ee_target_frame']==COMMAND_TASK_FRAME
    with pytest.raises(ValueError,match='every specified case'):
        save_command_run(tmp_path/'missing',suite,{}, {},0,'MuJoCo',[complete],evaluation={})
    assert not (tmp_path/'missing').exists()


def test_command_runtime_selects279_before_model_loading_and_legacy_stays276(monkeypatch):
    import pawweaver.mujoco_runtime as module
    monkeypatch.setattr(module,'verify_asset',lambda _:dict(robot='synthetic_software_fixture',asset_hash='asset'))
    seen=[]
    def stop_before_physics(*args,observation_spec,**kwargs):
        seen.append(observation_spec.size)
        raise ValueError('stop before model loading')
    monkeypatch.setattr(module,'load_bundle',stop_before_physics)
    for cls in [MujocoRunner,CommandedMujocoRunner]:
        with pytest.raises(ValueError,match='stop before model'):
            cls(Path('unused'),Path('unused'),software_fixture=True)
    assert seen==[276,279]


def test_commanded_step_history_keeps_task_frame_while_base_moves():
    runner=CommandedMujocoRunner.__new__(CommandedMujocoRunner)
    runner.action=torch.zeros(1,18);runner.ground_z=torch.zeros(1);runner.data=SimpleNamespace(time=0.)
    runner.observations=CommandObservationBuilder(1,torch.zeros(1,18))
    current=[state()];runner.state=lambda:current[0];runner._reset_physics=lambda:current[0]
    runner.reset([1.,0.,.6],goal_quat_t=[1.,0.,0.,0.])
    runner.policy=lambda obs:torch.zeros(1,18)
    def advance():
        current[0]=state(x=5.,yaw=np.pi/2,height=.2);runner.data.time+=.02
    runner._apply_action=advance
    first,_=runner.step([1.,0.,.6],goal_quat_t=[1.,0.,0.,0.],velocity_command=[.1,.2,.3])
    second,_=runner.step([1.,0.,.6],goal_quat_t=[1.,0.,0.,0.],velocity_command=[.4,.5,.6])
    assert first.shape==second.shape==(279,)
    np.testing.assert_allclose(second[-3:],[.4,.5,.6])
    torch.testing.assert_close(runner.observations.goals_t,torch.tensor([[[1.,0.,.6]]]).expand(1,4,3))
    # Changing base height affects body-relative EE goal height; T's world height stays .6.
    world,_=task_pose_to_world(runner.observations.goals_t,runner.observations.goal_quats_t,
        current[0].base_pos_w,current[0].base_quat_w,runner.ground_z)
    torch.testing.assert_close(world[0,0],torch.tensor([5.,1.,.6]))


def test_mujoco_evaluator_scores_same_k_target_with_post_base_without_physics(tmp_path):
    runner=CommandedMujocoRunner.__new__(CommandedMujocoRunner)
    runner.ground_z=torch.zeros(1);runner.torque=np.zeros(18);runner.diagnostic=True
    runner.bundle={'policy_sha256':'p','trained':False};runner.termination={'minimum_base_height_m':.2,'minimum_base_up_z':.35}
    runner.model=SimpleNamespace(nbody=1,body=lambda _:SimpleNamespace(name='base_link'))
    runner.data=SimpleNamespace(body=lambda _:SimpleNamespace(xmat=np.eye(3).ravel()))
    current=[state()];calls=[]
    runner.state=lambda:current[0]
    runner.reset=lambda *args,**kwargs:None
    runner._contact_snapshot=lambda:(np.zeros(1),0)
    def tick(goal_t,*,goal_quat_t,velocity_command):
        calls.append(np.array(goal_t))
        current[0]=state(x=float(len(calls)),height=.5)
        return np.zeros(279),np.zeros(18)
    runner.step=tick
    result=runner.evaluate(trajectory(),tmp_path)
    assert result['actual_steps']==2 and result['completed']
    with np.load(tmp_path/'trace.npz') as trace:
        np.testing.assert_allclose(trace['goal_task'][:,0],[1.,2.])  # k0/k1, not k1/k2.
        np.testing.assert_allclose(trace['goal'][:,0],[2.,4.])  # New base X plus same k target.
        np.testing.assert_allclose(trace['errors'],[0.,1.])
        assert trace['observations'].shape==(2,279)
    with pytest.raises(ValueError,match='explicit commanded trajectory'):
        runner.evaluate(object(),tmp_path)


def test_physx_batch_keeps_task_samples_and_stops_each_case():
    positions,quats,commands=isaac.batch_reference([trajectory()],3)
    np.testing.assert_allclose(positions[0,:,0],[1.,2.,3.])
    np.testing.assert_allclose(commands[0,:,2],[.2,.4,.6])
    rows=[dict(times=[],goal_task=[]) for _ in range(2)];fallen=np.zeros(2,dtype=bool)
    assert not isaac.append_step(rows,[2,3],fallen,1,{'goal_task':np.array([[1,0,0],[2,0,0]])},[True,False])
    assert not isaac.append_step(rows,[2,3],fallen,2,{'goal_task':np.zeros((2,3))},[False,False])
    assert isaac.append_step(rows,[2,3],fallen,3,{'goal_task':np.zeros((2,3))},[False,False])
    assert [len(r['times']) for r in rows]==[1,3]
    expected={'commanded_position_t':torch.tensor([[1.,0.,.6]])}
    isaac.verify_scored_step(expected,expected,1)
    with pytest.raises(ValueError,match='commanded_position_t'):
        isaac.verify_scored_step({'commanded_position_t':torch.tensor([[2.,0.,.6]])},expected,1)
