import numpy as np
import pytest
import torch

from pawweaver.task import GoalBank, episode_metrics, reward_terms, termination_config, fall_causes
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


@pytest.mark.parametrize('stage', [0, 2])
def test_demonstrations_only_samples_uniform_files_and_preserves_loaded_pose(tmp_path, stage):
    paths=[]
    for index in range(3):
        q=[np.cos((index+1)*.1),0,0,np.sin((index+1)*.1)]
        path=tmp_path/f'{index}.npz'
        Trajectory([0,1],[[2,3,4],[3+index,3,4]],[[1,0,0,0],q],{'split':'train'}).save(path)
        paths.append(path)
    count=90
    bank=GoalBank(count,10,'cpu',seed=17,stage=stage,demonstrations=paths,demonstrations_only=True)
    starts=torch.tensor([[8.,9.,10.]]).repeat(count,1)
    bank.reset(torch.arange(count),starts,torch.tensor([[0.,1.,0.,0.]]).repeat(count,1))
    # Each reset makes exactly one uniform file draw, without a family draw.
    selected=np.random.default_rng(17).integers(3,size=count)
    np.testing.assert_array_equal(bank.demonstration_index,selected)
    assert set(selected)=={0,1,2}
    for index,file_index in enumerate(selected):
        trajectory=Trajectory.load(paths[file_index]).align(starts[index].numpy())
        np.testing.assert_allclose(bank.positions[index],trajectory.sample(np.arange(15)*.02),atol=1e-6)
        np.testing.assert_allclose(bank.orientations_wxyz[index],trajectory.sample_orientation(np.arange(15)*.02),atol=1e-7)


def test_demonstrations_only_rejects_empty_or_nontraining_files(tmp_path):
    with pytest.raises(ValueError,match='nonempty'):
        GoalBank(1,10,'cpu',demonstrations_only=True)
    path=tmp_path/'heldout.npz'
    Trajectory([0,1],[[0,0,0],[1,0,0]],[[1,0,0,0]]*2,{'split':'test'}).save(path)
    with pytest.raises(ValueError,match='pre-split training'):
        GoalBank(1,10,'cpu',demonstrations=[path],demonstrations_only=True)


def test_explicit_false_keeps_old_family_sampling(tmp_path):
    path=tmp_path/'demo.npz'
    Trajectory([0,1],[[0,0,0],[1,0,0]],[[1,0,0,0]]*2,{'split':'train'}).save(path)
    args=dict(batch=24,steps=10,device='cpu',seed=23,stage=2,demonstrations=[path])
    old=GoalBank(**args)
    explicit=GoalBank(**args,demonstrations_only=False)
    for bank in (old,explicit):
        bank.reset(torch.arange(24),torch.zeros(24,3),torch.tensor([[1.,0.,0.,0.]]).repeat(24,1))
    torch.testing.assert_close(old.positions,explicit.positions,rtol=0,atol=0)
    torch.testing.assert_close(old.orientations_wxyz,explicit.orientations_wxyz,rtol=0,atol=0)
    np.testing.assert_array_equal(old.family,explicit.family)
    assert len(set(old.family))>1
    np.testing.assert_array_equal(old.demonstration_index,np.where(old.family==len(old.sampler.families)-1,0,-1))


def test_group_weights_select_group_mass_then_uniform_file_and_keep_pose(tmp_path):
    paths=[]
    for index,group in enumerate(['local','local','far','excluded']):
        path=tmp_path/f'group{index}.npz'
        Trajectory([0,1],[[2,3,4],[3+index,3,4]],
                   [[1,0,0,0],[np.cos(.2),0,0,np.sin(.2)]],
                   {'split':'train','training_group':group}).save(path)
        paths.append(path)
    bank=GoalBank(80,10,'cpu',seed=42,demonstrations=paths,demonstrations_only=True,
                  demonstration_group_weights={'local':1.,'far':3.,'absent_zero':0.})
    ids=torch.arange(80)
    starts=torch.tensor([[8.,9.,10.]]).repeat(80,1)
    bank.reset(ids,starts,torch.tensor([[0.,1.,0.,0.]]).repeat(80,1))
    rng=np.random.default_rng(42)
    expected=[]
    for _ in ids:
        group=[[0,1],[2]][rng.choice(2,p=[.25,.75])]
        expected.append(group[rng.integers(len(group))])
    np.testing.assert_array_equal(bank.demonstration_index,expected)
    assert set(expected)=={0,1,2}
    for index,selected in enumerate(expected):
        trajectory=Trajectory.load(paths[selected]).align(starts[index].numpy())
        np.testing.assert_allclose(bank.positions[index],trajectory.sample(np.arange(15)*.02),atol=1e-6)
        np.testing.assert_allclose(bank.orientations_wxyz[index],trajectory.sample_orientation(np.arange(15)*.02),atol=1e-7)
    # A partial reset updates only its selected environments, and synthetic
    # references must clear a previous imported-file index.
    previous=bank.demonstration_index.copy()
    bank.demonstrations_only=False
    bank.static_goal_offsets_m=np.zeros((1,3))
    bank.reset(torch.tensor([3]),starts[[3]],torch.tensor([[1.,0.,0.,0.]]))
    previous[3]=-1
    np.testing.assert_array_equal(bank.demonstration_index,previous)


@pytest.mark.parametrize('weights',[{},[],{'a':-1},{'a':float('nan')},{'a':float('inf')},
                                     {'a':0},{'missing':1},{'':1},{'a':True},{'a':'1'}])
def test_invalid_demonstration_group_weights_rejected(tmp_path,weights):
    path=tmp_path/'group.npz'
    Trajectory([0,1],[[0,0,0],[1,0,0]],[[1,0,0,0]]*2,
               {'split':'train','training_group':'a'}).save(path)
    with pytest.raises(ValueError,match='group|weight'):
        GoalBank(1,10,'cpu',demonstrations=[path],demonstration_group_weights=weights)


@pytest.mark.parametrize('group',[None,'',3])
def test_group_metadata_required_only_when_group_sampling_enabled(tmp_path,group):
    path=tmp_path/'group.npz'
    Trajectory([0,1],[[0,0,0],[1,0,0]],[[1,0,0,0]]*2,
               {'split':'train','training_group':group}).save(path)
    GoalBank(1,10,'cpu',demonstrations=[path])
    with pytest.raises(ValueError,match='training_group'):
        GoalBank(1,10,'cpu',demonstrations=[path],demonstration_group_weights={'a':1})


@pytest.mark.parametrize('minimum', [.2,.15])
def test_fall_rule_keeps_strict_scalar_numpy_and_torch_predicates(minimum):
    rule=termination_config({'minimum_base_height_m':minimum})
    assert termination_config({})=={'minimum_base_height_m':.2,'minimum_base_up_z':.35}
    heights=[.18,.14,.3,minimum,.3,.14]
    up=[1.,1.,.2,.35,.35,.2]
    expected_height=[minimum==.2,True,False,False,False,True]
    expected_tilt=[False,False,True,False,False,True]
    for dtype in (torch.float32,torch.float64):
        height,tilt=fall_causes(torch.tensor(heights,dtype=dtype),torch.tensor(up,dtype=dtype),rule)
        assert height.tolist()==expected_height and tilt.tolist()==expected_tilt
    height,tilt=fall_causes(np.array(heights),np.array(up),rule)
    assert height.tolist()==expected_height and tilt.tolist()==expected_tilt
    assert [fall_causes(h,u,rule) for h,u in zip(heights,up)]==list(zip(expected_height,expected_tilt))


@pytest.mark.parametrize('height', [0,-.1,float('nan'),float('inf')])
def test_invalid_minimum_base_height_fails(height):
    with pytest.raises(ValueError,match='finite and positive'):
        termination_config({'minimum_base_height_m':height})


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
