import numpy as np
import pytest
import torch

from pawweaver.task import GoalBank, episode_metrics, reward_terms
from pawweaver.trajectories import Trajectory


def test_demonstration_preserves_rotation_instead_of_replacing_with_reset_pose(tmp_path):
    demonstration=Trajectory([0,1],[[2,3,4],[3,3,4]],
                             [[1,0,0,0],[np.sqrt(.5),0,0,np.sqrt(.5)]],{"split":"train"})
    path=tmp_path/"demo.npz"
    demonstration.save(path)
    bank=GoalBank(1,10,"cpu",stage=1,demonstrations=[path])
    class PickLast:
        def integers(self,high):
            return high-1
    bank.rng=PickLast()
    bank.reset(torch.tensor([0]),torch.tensor([[8.,9.,10.]]),torch.tensor([[0.,1.,0.,0.]]))
    torch.testing.assert_close(bank.positions[0,0],torch.tensor([8.,9.,10.]))
    torch.testing.assert_close(bank.current_orientation(torch.tensor([0])),torch.tensor([[1.,0.,0.,0.]]))
    q=bank.current_orientation(torch.tensor([10]))[0].numpy()
    np.testing.assert_allclose(q,[np.cos(.2*np.pi/4),0,0,np.sin(.2*np.pi/4)],atol=1e-7)


def test_bad_reset_pose_fails_before_changing_other_goals():
    bank=GoalBank(2,10,"cpu",static_goal_offsets_m=[[0,0,0]])
    starts=torch.zeros(2,3)
    bank.reset(torch.arange(2),starts,torch.tensor([[1.,0.,0.,0.]]*2))
    previous=bank.orientations_wxyz.clone()
    with pytest.raises(ValueError,match="nonzero"):
        bank.reset(torch.tensor([1]),starts[[1]],torch.zeros(1,4))
    assert torch.equal(previous,bank.orientations_wxyz)


def test_orientation_reward_is_separate_from_position_and_uses_radians():
    z=torch.zeros(3)
    vectors=torch.zeros(3,3)
    joints=torch.zeros(3,18)
    kwargs=dict(error=vectors,previous_error=z,tcp_velocity=vectors,goal_velocity=vectors,
                action=joints,previous_action=joints,torque=joints,effort=torch.ones_like(joints),
                q=joints,qd=joints,previous_qd=joints,lower=-torch.ones_like(joints),upper=torch.ones_like(joints),
                gravity_b=vectors,foot_velocity=torch.zeros(3,4,3),foot_contact=torch.zeros(3,4),
                collision=torch.zeros(3,dtype=torch.bool),fallen=torch.zeros(3,dtype=torch.bool))
    terms=reward_terms(orientation_error=torch.tensor([0.,.5,np.pi]),**kwargs)
    assert terms["orientation_tracking"][0]==1
    assert 1>terms["orientation_tracking"][1]>terms["orientation_tracking"][2]>0
    torch.testing.assert_close(terms["tracking"],torch.ones(3))
    with pytest.raises(TypeError,match="orientation_error"):
        reward_terms(**kwargs)
    with pytest.raises(ValueError,match="width"):
        reward_terms(orientation_error=z,orientation_tracking_width_rad=0,**kwargs)


def test_orientation_metrics_report_without_changing_position_reach_criterion():
    times=np.array([0.,1.,2.,3.])
    args=(times,np.array([.04,.04,.03,.04]),np.zeros((4,3)),np.zeros((4,18)),np.zeros((4,18)),False)
    result=episode_metrics(*args,orientation_errors=[0,0,.3,.4])
    assert result["reached"] is True
    assert result["reach_time_s"]==0
    assert result["orientation_rmse_rad"]==pytest.approx(np.sqrt((.3**2+.4**2)/2))
    assert result["orientation_p95_rad"]==pytest.approx(.395)
    assert not any("orientation" in key and ("pass" in key or "reach" in key) for key in result)
    with pytest.raises(TypeError,match="orientation_errors"):
        episode_metrics(*args)
    for invalid in ([0],[-1,0,0,0],[np.nan]*4,[4]*4):
        with pytest.raises(ValueError,match="shortest orientation"):
            episode_metrics(*args,orientation_errors=invalid)
