import dataclasses
import math
import numpy as np
import pytest
import torch

from pawweaver.contracts import GoalSample,RobotState,SCHEMA_VERSION
from pawweaver.math import (normalize_quat,quat_mul,quat_conjugate,quat_angle_error,
                            quat_to_rotation_6d,quat_apply,rpy_quat,rpy_matrix)
from pawweaver.observations import ObservationBuilder,ObservationSpec


def quaternion(rpy):
    return torch.tensor(rpy_quat(rpy),dtype=torch.float64)


def state(n=2):
    return RobotState(joint_pos=torch.zeros(n,18),joint_vel=torch.zeros(n,18),
        base_pos_w=torch.zeros(n,3),base_quat_w=torch.tensor([[1.,0,0,0]]).repeat(n,1),
        base_ang_vel_b=torch.zeros(n,3),tcp_pos_w=torch.tensor([[.2,0,.6]]).repeat(n,1),
        tcp_quat_w=torch.tensor([[1.,0,0,0]]).repeat(n,1),base_lin_vel_b=torch.zeros(n,3))


def test_goal_requires_and_normalizes_orientation_without_asset_schema_change():
    assert SCHEMA_VERSION==1
    with pytest.raises(TypeError): GoalSample(0.,(0.,0.,0.))
    for q in ((0,0,0,0),(float('nan'),0,0,1),(1,0,0),(float('inf'),0,0,0)):
        with pytest.raises(ValueError,match='orientation'): GoalSample(0.,(0.,0.,0.),q)
    sample=GoalSample(0.,(0.,0.,0.),(2.,0,0,0))
    assert sample.orientation_wxyz==(1.,0.,0.,0.)
    assert GoalSample(0.,(0.,0.,0.),(-2.,0,0,0)).orientation_wxyz==(-1.,0.,0.,0.)
    fields={field.name for field in dataclasses.fields(RobotState)}
    assert 'tcp_quat_w' in fields


def test_quaternion_composition_broadcast_and_column_order():
    a=quaternion((.4,-.2,.7));b=quaternion((-.3,.6,-.1))
    v=torch.tensor([[1.,2.,3.],[-2.,.3,1.]],dtype=torch.float64)
    torch.testing.assert_close(quat_apply(quat_mul(a,b),v),quat_apply(a,quat_apply(b,v)))
    product=quat_mul(a.reshape(1,1,4).expand(2,1,4),b.reshape(1,1,4).expand(1,3,4))
    assert product.shape==(2,3,4)
    torch.testing.assert_close(quat_mul(a,quat_conjugate(a)),torch.tensor([1.,0,0,0],dtype=torch.float64))
    matrix=torch.tensor(rpy_matrix((.4,-.2,.7)))
    expected=torch.cat((matrix[:,0],matrix[:,1]))
    torch.testing.assert_close(quat_to_rotation_6d(a),expected)
    assert torch.equal(quat_to_rotation_6d(a),quat_to_rotation_6d(-a))
    torch.testing.assert_close(quat_to_rotation_6d(a*1e200),expected)
    for bad in (torch.zeros(4),torch.tensor([float('inf'),0,0,0])):
        with pytest.raises(ValueError): normalize_quat(bad)
    scripted=torch.jit.script(quat_to_rotation_6d)
    torch.testing.assert_close(scripted(a),expected)


def test_shortest_angle_handles_sign_zero_and_pi():
    identity=torch.tensor([1.,0,0,0],dtype=torch.float64)
    q=quaternion((0,0,.4))
    torch.testing.assert_close(quat_angle_error(identity,q),torch.tensor(.4,dtype=torch.float64))
    assert quat_angle_error(q,-q).item()<1e-14
    assert abs(quat_angle_error(identity,quaternion((math.pi,0,0))).item()-math.pi)<1e-14
    assert quat_angle_error(identity,quaternion((1e-9,0,0))).item()==pytest.approx(1e-9)


def test_pose_update_is_atomic_and_shares_acceptance_stamp():
    s=state();b=ObservationBuilder(2,torch.zeros(18));ids=torch.arange(2)
    goal=torch.ones(2,3);b.reset(ids,s,goal,torch.zeros(2),goal_quat_w=s.tcp_quat_w)
    original_pos=b.goals_w.clone();original_quat=b.goal_quats_w.clone()
    invalid=s.tcp_quat_w.clone();invalid[0]=0
    valid_pos=goal+3
    b.push_goal(valid_pos,torch.ones(2),torch.ones(2,dtype=torch.bool),torch.ones(2),orientation_wxyz=invalid)
    assert torch.equal(b.goals_w[0],original_pos[0])
    assert torch.equal(b.goal_quats_w[0],original_quat[0])
    assert b.stamp.tolist()==[0.,1.] and b.valid.tolist()==[False,True]
    assert torch.equal(b.goals_w[1,-1],valid_pos[1])
    # Invalid position cannot update a valid new orientation either.
    pos=goal.clone();pos[1]=float('nan')
    changed=quaternion((0,0,.7)).float().repeat(2,1)
    previous=b.goal_quats_w.clone()
    b.push_goal(pos,torch.full((2,),2.),torch.ones(2,dtype=torch.bool),torch.ones(2),orientation_wxyz=changed)
    assert torch.equal(b.goal_quats_w[1],previous[1])
    assert b.stamp.tolist()==[2.,1.]
    # Out-of-order sample changes neither half nor validity of a fresher sample.
    previous_pos=b.goals_w.clone();previous_quat=b.goal_quats_w.clone()
    b.push_goal(goal*10,torch.zeros(2),torch.ones(2,dtype=torch.bool),torch.ones(2),orientation_wxyz=s.tcp_quat_w)
    assert torch.equal(b.goals_w,previous_pos) and torch.equal(b.goal_quats_w,previous_quat)
    assert b.valid.tolist()==[True,False]


def test_orientation_history_uses_current_base_and_preserves_position_prefix():
    s=state();b=ObservationBuilder(2,torch.zeros(18));ids=torch.arange(2);goal=torch.ones(2,3)
    b.reset(ids,s,goal,torch.zeros(2),goal_quat_w=s.tcp_quat_w)
    original=b.build(s,torch.zeros(2,18),torch.zeros(2))
    assert original.shape==(2,276) and ObservationSpec().schema_version==2
    expected_identity=torch.tensor([1.,0,0,0,1.,0])
    torch.testing.assert_close(original[:,246:252],expected_identity.repeat(2,1))
    torch.testing.assert_close(original[:,252:],expected_identity.repeat(2,4))
    rotation=quaternion((.1,.2,.3)).float().repeat(2,1)
    b.push_state(s)
    b.push_goal(goal,torch.ones(2)*.02,torch.ones(2,dtype=torch.bool),torch.ones(2),orientation_wxyz=rotation)
    next_obs=b.build(s,torch.zeros(2,18),torch.ones(2)*.02)
    assert torch.equal(next_obs[:,:246],original[:,:246])
    torch.testing.assert_close(next_obs[:,270:276],quat_to_rotation_6d(rotation))
    stored=b.goal_quats_w.clone()
    s.base_quat_w=rotation
    moved=b.build(s,torch.zeros(2,18),torch.ones(2)*.02)
    assert torch.equal(b.goal_quats_w,stored)
    torch.testing.assert_close(moved[:,270:276],expected_identity.repeat(2,1),atol=2e-7,rtol=1e-6)
    old_other=b.goal_quats_w[1].clone()
    b.reset(torch.tensor([0]),s,goal,torch.ones(2),goal_quat_w=rotation)
    assert torch.equal(b.goal_quats_w[1],old_other)
    # Missing orientation is never silently filled.
    with pytest.raises(TypeError): b.reset(ids,s,goal,torch.zeros(2))
    with pytest.raises(TypeError): b.push_goal(goal,torch.ones(2),torch.ones(2,dtype=torch.bool),torch.ones(2))
