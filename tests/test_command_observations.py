"""Task-frame causal history contracts; no simulator or learned-behavior claim."""
import numpy as np
import pytest
import torch

from pawweaver.contracts import RobotState
from pawweaver.math import rpy_matrix,rpy_quat
from pawweaver.observations import (ObservationSpec,CommandObservationSpec,
                                    CommandObservationBuilder,observation_spec)


def state():
    return RobotState(joint_pos=torch.zeros(2,18),joint_vel=torch.zeros(2,18),
        base_pos_w=torch.tensor([[1.,2.,.3],[3.,-2.,5.4]]),
        base_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1),
        base_ang_vel_b=torch.zeros(2,3),base_lin_vel_b=torch.zeros(2,3),
        tcp_pos_w=torch.tensor([[1.2,2.,.6],[3.2,-2.,5.6]]),
        tcp_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1))


def test_world_contract_unchanged_and_command_contract_explicit():
    expected={'schema_version':2,'history_frames':5,'goal_frames':4,'proprio_dim':42,
        'action_dim':18,'joint_velocity_scale':.05,'angular_velocity_scale':.25,
        'max_goal_age':.5,'control_dt':.02,'orientation_encoding':'rotation_columns_0_xyz_then_1_xyz'}
    assert ObservationSpec().to_dict()==expected
    assert observation_spec({}).to_dict()==observation_spec({'task_mode':'world_ee_pose'}).to_dict()==expected
    command=observation_spec({'task_mode':'velocity_ee_pose'})
    assert isinstance(command,CommandObservationSpec) and command.size==279
    assert command.to_dict()['task_mode']=='velocity_ee_pose'
    assert 'fixed_ground_z' in command.target_frame
    assert 'yaw_rate_rad_s' in command.velocity_command_order
    with pytest.raises(ValueError,match='task_mode'):
        observation_spec({'task_mode':'unknown'})
    with pytest.raises(ValueError,match='279'):
        CommandObservationBuilder(2,torch.zeros(18),ObservationSpec())


def test_history_remains_in_task_frame_under_current_xy_yaw_and_full_base_conversion():
    from scipy.spatial.transform import Rotation
    s=state();builder=CommandObservationBuilder(2,torch.zeros(18))
    initial=torch.tensor([[.4,.2,.6],[.3,-.1,.8]])
    q=torch.tensor(rpy_quat([.2,.1,.4]),dtype=torch.float32).repeat(2,1)
    builder.reset(torch.arange(2),s,initial,torch.zeros(2),goal_quat_t=q)
    for step in (1,2):
        s.base_pos_w[:,:2]+=step
        s.base_quat_w[:]=torch.tensor(rpy_quat([.1*step,-.2*step,.3*step]),dtype=torch.float32)
        builder.push_state(s)
        builder.push_goal(initial+step*.1,torch.full((2,),step*.02),torch.ones(2,dtype=torch.bool),
                          torch.ones(2),orientation_t=q)
    expected_history=torch.stack((initial,initial,initial+.1,initial+.2),dim=1)
    assert torch.equal(builder.goals_t,expected_history)
    stored=builder.goals_t.clone();stored_quats=builder.goal_quats_t.clone()
    ground=torch.tensor([0.,5.]);command=torch.tensor([[.2,-.1,.3],[0.,.1,-.4]])
    previous=torch.zeros(2,18)
    def build_and_reconstruct(rpy):
        s.base_quat_w[:]=torch.tensor(rpy_quat(rpy),dtype=torch.float32)
        obs=builder.build(s,previous,torch.full((2,),.04),velocity_command=command,ground_z=ground)
        assert obs.shape==(2,279)
        assert torch.equal(obs[:,-3:],command)
        goal_b=obs[:,231:243].reshape(2,4,3).numpy()
        rotation=rpy_matrix(rpy)
        recovered=np.einsum('ij,nhj->nhi',rotation,goal_b)+s.base_pos_w.numpy()[:,None,:]
        yaw=rpy_matrix([0.,0.,rpy[2]])
        origin=s.base_pos_w.numpy().copy();origin[:,2]=ground.numpy()
        expected=np.einsum('ij,nhj->nhi',yaw,stored.numpy())+origin[:,None,:]
        np.testing.assert_allclose(recovered,expected,atol=8e-7)
        for env in range(2):
            matrices=rotation.T@yaw@Rotation.from_quat(stored_quats[env].numpy()[:,[1,2,3,0]]).as_matrix()
            six=np.concatenate((matrices[:,:,0],matrices[:,:,1]),axis=-1)
            np.testing.assert_allclose(obs[env,252:276].reshape(4,6),six,atol=5e-7)
        assert torch.equal(builder.goals_t,stored)
        assert torch.equal(builder.goal_quats_t,stored_quats)
        return obs,recovered
    before,world=build_and_reconstruct([.2,-.4,.6])
    s.base_pos_w[:,2]+=.4
    after,world_after=build_and_reconstruct([-.3,.25,.6])
    # Body height/roll/pitch must affect the base-frame errors, not world target.
    np.testing.assert_allclose(world_after,world,atol=8e-7)
    assert not torch.equal(before[:,231:243],after[:,231:243])
    s.base_pos_w[:,:2]+=torch.tensor([2.,-1.])
    _,moved=build_and_reconstruct([-.3,.25,1.1])
    assert not np.allclose(moved,world)


def test_command_history_keeps_atomic_quality_and_partial_reset_semantics():
    s=state();builder=CommandObservationBuilder(2,torch.zeros(18));goal=torch.ones(2,3)
    q=s.tcp_quat_w
    builder.reset(torch.arange(2),s,goal,torch.zeros(2),goal_quat_t=q)
    bad=q.clone();bad[0]=0.
    builder.push_goal(goal+1,torch.full((2,),.02),torch.ones(2,dtype=torch.bool),torch.ones(2),orientation_t=bad)
    assert torch.equal(builder.goals_t[0,-1],goal[0])
    assert torch.equal(builder.goals_t[1,-1],goal[1]+1)
    assert builder.valid.tolist()==[False,True]
    stored=builder.goals_t.clone()
    builder.push_goal(goal+2,torch.zeros(2),torch.ones(2,dtype=torch.bool),torch.ones(2),orientation_t=q)
    assert torch.equal(builder.goals_t,stored)
    other=builder.goals_t[1].clone()
    builder.reset(torch.tensor([0]),s,goal,torch.full((2,),.04),goal_quat_t=q)
    assert torch.equal(builder.goals_t[1],other)
    obs=builder.build(s,torch.zeros(2,18),torch.ones(2),velocity_command=torch.zeros(2,3),ground_z=torch.tensor([0.,5.]))
    assert obs[:,243:245].eq(0).all()
    with pytest.raises(ValueError,match='Velocity command'):
        builder.build(s,torch.zeros(2,18),torch.ones(2),velocity_command=torch.full((2,3),float('nan')),ground_z=torch.zeros(2))
