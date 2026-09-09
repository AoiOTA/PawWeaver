import json

import numpy as np
import pytest

from pawweaver.math import rpy_quat
from pawweaver.trajectories import Trajectory, synthetic


def equivalent(actual, expected):
    np.testing.assert_allclose(np.abs(np.sum(np.asarray(actual)*expected,axis=-1)),1.,atol=1e-12)


def test_shortest_slerp_sign_equivalence_and_endpoint_hold():
    q=np.array([[1.,0.,0.,0.],[np.sqrt(.5),0.,0.,np.sqrt(.5)]])
    trajectory=Trajectory([0,1],[[0,0,0],[2,0,0]],q,{})
    equivalent(trajectory.sample_orientation(.5),[np.cos(np.pi/8),0,0,np.sin(np.pi/8)])
    np.testing.assert_allclose(trajectory.sample(.5),[1,0,0])
    equivalent(trajectory.sample_orientation([-5,5]),q)
    sign_flipped=Trajectory([0,1],trajectory.positions,-q,{})
    equivalent(sign_flipped.sample_orientation([0,.2,.7,1]),trajectory.sample_orientation([0,.2,.7,1]))
    # Opposite representations of one orientation must not rotate through zero.
    constant=Trajectory([0,1],trajectory.positions,[q[1],-q[1]],{})
    equivalent(constant.sample_orientation(np.linspace(0,1,11)),q[1])
    crossing=Trajectory([0,1],trajectory.positions,
                        [rpy_quat([0,0,np.deg2rad(170)]),rpy_quat([0,0,np.deg2rad(-170)])],{})
    equivalent(crossing.sample_orientation(.5),[0,0,0,1])


def test_align_applies_one_rigid_transform_to_positions_and_orientations():
    positions=np.array([[.5,.6,.7],[1.5,.6,.7],[.5,1.6,.7]])
    trajectory=Trajectory([0,1,2],positions,[[1,0,0,0]]*3,{"source":"test"})
    start=np.array([4.,5.,6.])
    aligned=trajectory.align(start,yaw=np.pi/2,start_orientation_wxyz=rpy_quat([np.pi/2,0,0]))
    np.testing.assert_allclose(aligned.positions,[start,start+[0,1,0],start+[0,0,1]],atol=1e-12)
    equivalent(aligned.orientations_wxyz,rpy_quat([np.pi/2,0,np.pi/2]))
    assert aligned.metadata["alignment"]=="episode_start_only"
    np.testing.assert_array_equal(trajectory.positions,positions)
    # Aligning an already oriented first pose with zero yaw meets the requested pose.
    rotated=Trajectory([0,1,2],positions,[rpy_quat([0,0,.7])]*3,{})
    equivalent(rotated.align(start,start_orientation_wxyz=rpy_quat([.3,.4,0])).orientations_wxyz,
               rpy_quat([.3,.4,0]))


def test_pose_roundtrip_normalizes_and_rejects_position_only_files(tmp_path):
    trajectory=Trajectory([0,1],[[0,0,0],[1,2,3]],[[1e-250,0,0,0],[0,1e250,0,0]],{"source":"explicit"})
    path=tmp_path/"pose.npz"
    trajectory.save(path)
    with np.load(path,allow_pickle=False) as saved:
        assert saved["schema_version"].item()==2
        assert "orientations_wxyz" in saved
    restored=Trajectory.load(path)
    np.testing.assert_array_equal(restored.positions,trajectory.positions)
    equivalent(restored.orientations_wxyz,trajectory.orientations_wxyz)
    assert restored.metadata==trajectory.metadata
    np.savez(tmp_path/"old.npz",timestamps=[0,1],positions=trajectory.positions,metadata=json.dumps({}))
    with pytest.raises(ValueError,match="Position-only.*missing"):
        Trajectory.load(tmp_path/"old.npz")
    np.savez(tmp_path/"wrong_version.npz",schema_version=1,timestamps=[0,1],positions=trajectory.positions,
             orientations_wxyz=trajectory.orientations_wxyz,metadata=json.dumps({}))
    with pytest.raises(ValueError,match="schema_version=2"):
        Trajectory.load(tmp_path/"wrong_version.npz")


@pytest.mark.parametrize("orientations",[[[0,0,0,0]]*2,[[np.nan,0,0,1]]*2,[[np.inf,0,0,1]]*2,[[1,0,0]]*2])
def test_invalid_pose_orientation_fails(orientations):
    with pytest.raises(ValueError,match="orientations|nonzero"):
        Trajectory([0,1],[[0,0,0],[1,0,0]],orientations,{})


def test_synthetic_orientation_is_explicit_and_not_a_reachability_claim():
    with pytest.raises(TypeError,match="orientation_wxyz"):
        synthetic("line",1)
    trajectory=synthetic("line",1,orientation_wxyz=[0,0,0,2])
    equivalent(trajectory.orientations_wxyz,[0,0,0,1])
    assert trajectory.metadata["orientation_generation"]=="explicit_constant_wxyz"
    assert trajectory.metadata["reachability"]=="not robot-validated"
