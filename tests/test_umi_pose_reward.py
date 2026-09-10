"""CPU contract checks; fixed states do not establish physics or learning."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pawweaver.task import UmiPoseReward,termination_config


@pytest.mark.parametrize("attribute,schedule",[
    ("position",UmiPoseReward.position_schedule),
    ("orientation",UmiPoseReward.orientation_schedule)])
def test_strict_thresholds_and_rewidening(attribute,schedule):
    course=UmiPoseReward()
    ema=getattr(course,attribute+("_ema_m" if attribute=="position" else "_ema_rad"))
    sigma_name=attribute+("_sigma_m2" if attribute=="position" else "_sigma_rad")
    for i,(threshold,sigma) in enumerate(schedule):
        ema.fill_(threshold)
        course.on_reset()
        assert getattr(course,sigma_name)==schedule[max(0,i-1)][1]
        ema.fill_(threshold-1e-8)
        course.on_reset()
        assert getattr(course,sigma_name)==sigma
    ema.fill_(1.)
    course.on_reset()
    assert getattr(course,sigma_name)==schedule[0][1]


def test_global_ema_and_reset_persistence():
    course=UmiPoseReward()
    course.on_reset()
    assert (course.position_sigma_m2,course.orientation_sigma_rad)==(2,8)
    course.update(torch.tensor([0.,2.,4.]),torch.tensor([0.,0.,0.]))
    assert course.position_ema_m.item()==pytest.approx(1.00125)
    assert course.orientation_ema_rad.item()==pytest.approx(.99875)
    before=course.state_dict()
    course.on_reset()
    assert course.position_ema_m.item()==before["position_ema_m"]
    assert course.orientation_ema_rad.item()==before["orientation_ema_rad"]
    assert course.orientation_sigma_rad==4


def test_kernel_and_only_pose_terms_replaced():
    course=UmiPoseReward()
    terms={"tracking":torch.zeros(2),"orientation_tracking":torch.zeros(2),
           "collision":torch.tensor([0.,1.]),"termination":torch.ones(2)}
    weights={"tracking":3.,"orientation_tracking":1.,"collision":-10.,"termination":-5.}
    result=course.nonterminal_reward(terms,weights,torch.tensor([0.,1.]),torch.tensor([0.,2.]))
    torch.testing.assert_close(result,torch.tensor([4.,4*torch.exp(torch.tensor(-.75))-10]))


@pytest.fixture
def fixed_env():
    pytest.importorskip("tensordict")
    from pawweaver.isaac_env import WholeBodyEnv
    env=WholeBodyEnv.__new__(WholeBodyEnv)
    joints=torch.zeros(2,18)
    vectors=torch.zeros(2,3)
    quat=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1)
    state=SimpleNamespace(base_pos_w=torch.tensor([[0.,0.,.3],[0.,0.,.1]]),
        base_quat_w=quat,tcp_pos_w=vectors,tcp_quat_w=quat,joint_pos=joints,joint_vel=joints)
    env.config={"domain_randomization":False,"umi_pose_reward":True,"reward_weights":{
        "tracking":3.,"orientation_tracking":1.,"progress":0.,"velocity_tracking":0.,
        "action_rate":0.,"normalized_torque":0.,"joint_acceleration":0.,"joint_limit":0.,
        "saturation":0.,"foot_slip":0.,"body_tilt":0.,"collision":-10.,"termination":-5.}}
    env.umi_pose_reward=UmiPoseReward()
    env.termination=termination_config(env.config)
    env.spec=SimpleNamespace(decimation=0)
    env.num_envs=2
    env.device="cpu"
    env.diagnostic=True
    env.state=lambda:state
    env.episode_length_buf=torch.zeros(2,dtype=torch.long)
    env.max_episode_length=1000
    env.reference=SimpleNamespace(current=lambda _:torch.tensor([[1.,0.,0.],[1.,0.,0.]]),
        current_orientation=lambda _:quat,sampler=SimpleNamespace(families=["hold"],update=Mock()),
        family=[0,0])
    env.observations=Mock()
    env.scene=SimpleNamespace(sensors={},env_origins=vectors)
    env.robot=SimpleNamespace(body_names=["foot","base"],data=SimpleNamespace(
        body_link_lin_vel_w=SimpleNamespace(torch=torch.zeros(2,2,3))))
    env.foot_ids=[0]
    env.contacts=torch.tensor([[0.,6.],[0.,6.]])
    env.pd=SimpleNamespace(command=lambda _:None,last_action=joints,effort=torch.ones(18),
        lower=-torch.ones(18),upper=torch.ones(18))
    env.previous_error=torch.zeros(2)
    env.episode_error=torch.zeros(2)
    env.previous_tcp=vectors.clone()
    env.previous_action=env.torque=env.previous_qd=joints.clone()
    env.get_observations=lambda:None
    return env


def test_actual_step_clips_nonterminal_but_preserves_fall_cost(fixed_env):
    env=fixed_env
    _,reward,done,extras=env.step(torch.zeros(2,18),auto_reset=False)
    torch.testing.assert_close(reward,torch.tensor([0.,-5.]))
    assert done.tolist()==[False,True]
    assert extras["umi_nonterminal_clipped_fraction"]==1
    assert extras["umi_nonterminal_preclip_mean"]==pytest.approx(4*torch.exp(torch.tensor(-.5)).item()-10)
    assert extras["falls"]==1 and extras["resets"]==0
    assert extras["collision_control_samples"]==2
    assert "reward_diagnostics" not in extras
    assert "demonstration_group_stats" not in extras


@pytest.mark.parametrize("second_index",[1,-1])
def test_group_statistics_capture_pre_reset_identity_and_exclude_unimported(fixed_env,second_index):
    env=fixed_env
    env.config["demonstration_group_weights"]={"near":1.,"body":1.,"empty":0.}
    env.reference.demonstration_index=torch.tensor([0,second_index]).numpy()
    env.reference.demonstrations=[SimpleNamespace(metadata={"training_group":g}) for g in ["near","body"]]
    env.reference.current=lambda _:torch.tensor([[1.,0.,0.],[2.,0.,0.]])
    env.episode_length_buf[:]=torch.tensor([99,9])
    def reset(ids):
        assert ids.tolist()==[1]
        env.reference.demonstration_index[ids.numpy()]=0
        env.episode_length_buf[ids]=0
    env.reset=reset
    _,_,_,extras=env.step(torch.zeros(2,18))
    groups=extras["demonstration_group_stats"]
    assert groups["near"]["transitions"]==1
    assert groups["near"]["position_error_sum_m"]==1.
    assert groups["near"]["ended_episodes"]==0
    assert groups["empty"]["transitions"]==0
    if second_index>=0:
        body=groups["body"]
        assert body["transitions"]==body["falls"]==body["resets"]==body["ended_episodes"]==1
        assert body["timeouts"]==0
        assert body["ended_episode_seconds_sum"]==pytest.approx(.2)
        assert body["position_error_sum_m"]==2.
        assert body["nonterminal_preclip_sum"]<0
        assert body["nonterminal_postclip_sum"]==0.
    else:
        assert all(value==0 for value in groups["body"].values())


def test_group_statistics_timeout_and_fall_are_disjoint(fixed_env):
    env=fixed_env
    env.config["demonstration_group_weights"]={"near":1.}
    env.reference.demonstration_index=torch.tensor([0,0]).numpy()
    env.reference.demonstrations=[SimpleNamespace(metadata={"training_group":"near"})]
    env.episode_length_buf[:]=999
    _,_,_,extras=env.step(torch.zeros(2,18),auto_reset=False)
    group=extras["demonstration_group_stats"]["near"]
    assert group["transitions"]==group["ended_episodes"]==2
    assert group["falls"]==group["timeouts"]==1
    assert group["resets"]==0
    assert group["ended_episode_seconds_sum"]==40.


@pytest.mark.parametrize("clip",[True,False])
def test_umi_diagnostics_match_actual_reward_and_fall_semantics(fixed_env,clip):
    env=fixed_env
    env.config.update(reward_diagnostics=True,umi_clip_nonnegative=clip)
    _,reward,done,extras=env.step(torch.zeros(2,18),auto_reset=False)
    diagnostics=extras["reward_diagnostics"]
    weighted=diagnostics["weighted_nonterminal_terms"]
    assert not {"tracking","orientation_tracking","termination"}&weighted.keys()
    torch.testing.assert_close(weighted["umi_pose"],torch.full((2,),4*torch.exp(torch.tensor(-.5))))
    torch.testing.assert_close(weighted["collision"],torch.full((2,),-10.))
    torch.testing.assert_close(sum(weighted.values()),diagnostics["nonterminal_preclip"])
    preclip=diagnostics["nonterminal_preclip"]
    expected=preclip.clamp_min(0) if clip else preclip
    torch.testing.assert_close(diagnostics["nonterminal_postclip"],expected)
    torch.testing.assert_close(reward,expected*.02+torch.tensor([0.,-5.]))
    assert done.tolist()==[False,True]
    assert extras["umi_nonterminal_clipped_fraction"]==(1. if clip else 0.)
    assert (preclip<0).all()
    if not clip:
        assert (reward<0).all()


def test_umi_weighted_terms_preserve_default_formula():
    course=UmiPoseReward()
    position=torch.tensor([0.,.2,1.])
    orientation=torch.tensor([0.,.8,2.])
    terms={"tracking":torch.ones(3),"orientation_tracking":torch.ones(3),
           "termination":torch.ones(3),"collision":torch.tensor([0.,1.,0.]),
           "action_rate":torch.tensor([.1,.2,.3])}
    weights={"tracking":99.,"orientation_tracking":88.,"termination":-5.,
             "collision":-10.,"action_rate":-.02}
    legacy=4*torch.exp(-position.square()/2.)*torch.exp(-orientation/8.)+sum(
        weights[name]*value for name,value in terms.items()
        if name not in ("tracking","orientation_tracking","termination"))
    actual,weighted=course.nonterminal_reward(terms,weights,position,orientation,return_terms=True)
    assert torch.equal(actual,legacy)
    assert torch.equal(course.nonterminal_reward(terms,weights,position,orientation),legacy)
    torch.testing.assert_close(sum(weighted.values()),legacy)


def test_actual_step_support_costs_use_world_fz_and_world_xy_with_dt(fixed_env):
    from pawweaver.contracts import FOOT_NAMES
    env=fixed_env
    env.contacts.zero_()
    env.state().base_pos_w[:,2]=.3
    _,baseline,_,_=env.step(torch.zeros(2,18),auto_reset=False)
    names=list(FOOT_NAMES)+[name.replace('_foot','_thigh') for name in FOOT_NAMES]
    env.robot.body_names=names
    env.foot_ids=[0,1,2,3];env.thigh_ids=[4,5,6,7]
    pose=torch.zeros(2,8,7);pose[:,:,6]=1.
    pose[1,0,0]=.5
    pose[:,:,2]=torch.arange(8)  # Vertical differences must not affect planar preference.
    env.robot.data.body_link_pose_w=SimpleNamespace(torch=pose)
    env.robot.data.body_link_lin_vel_w=SimpleNamespace(torch=torch.zeros(2,8,3))
    env.contacts=torch.zeros(2,8)
    class Scene(dict):
        pass
    scene=Scene();scene.sensors={};scene.env_origins=torch.zeros(2,3)
    for i,name in enumerate(FOOT_NAMES):
        forces=torch.tensor([[[1000.,0.,10.]],[[1000.,0.,40. if i==0 else 0.]]])
        sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=SimpleNamespace(torch=forces)))
        scene['contact_'+name]=sensor;scene.sensors['contact_'+name]=sensor
    env.scene=scene
    env.config['reward_weights'].update(even_mass_distribution=-1.,feet_under_hips=-1.)
    _,reward,_,_=env.step(torch.zeros(2,18),auto_reset=False)
    torch.testing.assert_close(baseline-reward,torch.tensor([0.,(.25+1-torch.exp(torch.tensor(-1.)))*.02]),atol=1e-7,rtol=0)
    # The legacy contacts field still contains full vector norms, not Fz.
    assert (env.contacts[:,:4]>=1000).all()




@pytest.mark.parametrize('height,tilt,fallen', [(.18,False,False),(.14,False,True),(.3,True,True),(.14,True,True)])
def test_actual_physx_step_preserves_configured_fall_causes_before_reset(fixed_env,height,tilt,fallen):
    env=fixed_env
    env.config['minimum_base_height_m']=.15
    env.termination=termination_config(env.config)
    state=env.state()
    # Nonzero environment origins must not affect the resolved ground height.
    env.scene.env_origins=env.scene.env_origins.clone()
    env.scene.env_origins[:,2]=2.
    state.base_pos_w[:,2]=height+2.
    if tilt:
        state.base_quat_w[:]=torch.tensor([2**-.5,2**-.5,0.,0.])
    env.reset=lambda ids: state.base_pos_w.fill_(99.)
    _,_,done,extras=env.step(torch.zeros(2,18),auto_reset=True)
    assert done.tolist()==[fallen]*2
    assert extras['fall_height'].tolist()==[height<.15]*2
    assert extras['fall_tilt'].tolist()==[tilt]*2
    torch.testing.assert_close(extras['base_up_z'],torch.full((2,),0. if tilt else 1.),atol=1e-6,rtol=0)


def test_default_actual_step_keeps_negative_nonterminal(fixed_env):
    env=fixed_env
    env.config["umi_pose_reward"]=False
    _,reward,_,extras=env.step(torch.zeros(2,18),auto_reset=False)
    expected=(3*torch.exp(torch.tensor(-1/.15**2))+1-10)*.02
    torch.testing.assert_close(reward,torch.tensor([expected,expected-5]))
    assert "umi_nonterminal_clipped_fraction" not in extras


def test_empty_reset_leaves_curriculum_unchanged(fixed_env):
    env=fixed_env
    env.umi_pose_reward.position_ema_m.fill_(.05)
    before=env.umi_pose_reward.state_dict()
    env.reset(torch.empty(0,dtype=torch.long))
    assert env.umi_pose_reward.state_dict()==before


def test_actual_step_uses_old_sigma_until_reset(fixed_env):
    env=fixed_env
    env.contacts.zero_()
    env.umi_pose_reward.position_ema_m.fill_(.05)
    # Run the actual reset curriculum tail while replacing simulator-only work.
    source=Path(__file__).resolve().parents[1]/"src/pawweaver/isaac_env.py"
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="WholeBodyEnv")
    reset=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="reset")
    code=compile(ast.Module(body=[reset.body[-1]],type_ignores=[]),str(source),"exec")
    env.reset=lambda ids:exec(code,{"self":env}) if len(ids) else None
    _,reward,_,_=env.step(torch.zeros(2,18))
    expected=4*torch.exp(torch.tensor(-.5))*.02
    torch.testing.assert_close(reward,torch.tensor([expected,expected-5]))
    assert env.umi_pose_reward.position_sigma_m2==.005


def test_checkpoint_roundtrip_and_resume_reselects(tmp_path):
    course=UmiPoseReward()
    course.update(torch.zeros(4),torch.zeros(4))
    path=tmp_path/"checkpoint.pt"
    torch.save({"umi_pose_reward":course.state_dict()},path)
    checkpoint=torch.load(path,weights_only=False)
    restored=UmiPoseReward()
    restored.load_state_dict(checkpoint["umi_pose_reward"])
    assert restored.state_dict()==course.state_dict()
    restored.on_reset()
    assert (restored.position_sigma_m2,restored.orientation_sigma_rad)==(1,4)
    # A transferred policy leaves the new environment's curriculum fresh.
    assert UmiPoseReward().state_dict()["position_ema_m"]==1


@pytest.mark.parametrize("state",[None,{}, {"position_ema_m":.1},
    dict(position_ema_m=float("nan"),orientation_ema_rad=1,position_sigma_m2=2,orientation_sigma_rad=8)])
def test_invalid_resume_state_fails_without_mutation(state):
    course=UmiPoseReward()
    before=course.state_dict()
    with pytest.raises(ValueError,match="UMI pose reward"):
        course.load_state_dict(state)
    assert course.state_dict()==before


def test_actual_trainer_checkpoint_branches():
    source=Path(__file__).resolve().parents[1]/"scripts/train.py"
    tree=ast.parse(source.read_text())
    preflight=next(n for n in tree.body if isinstance(n,ast.If)
                   and ast.unparse(n.test).startswith("args.resume and config.get('umi_pose_reward'"))
    profile_branches=[n for n in ast.walk(tree) if isinstance(n,ast.If)
                      and ast.unparse(n.test)=="env.umi_pose_reward is not None"]
    save=next(n for n in profile_branches if any(isinstance(c,ast.Assign)
              and "checkpoint['umi_pose_reward']" in ast.unparse(c) for c in n.body))
    restore=next(n for n in profile_branches if "load_state_dict" in ast.unparse(n))
    def execute(node,namespace):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
    namespace={"env":SimpleNamespace(umi_pose_reward=None),"checkpoint":{},
               "args":SimpleNamespace(resume=True),"config":{},"UmiPoseReward":UmiPoseReward}
    execute(preflight,namespace)  # Old no-profile checkpoint remains valid.
    execute(save,namespace)
    execute(restore,namespace)
    assert namespace["checkpoint"]=={}
    namespace["config"]={"umi_pose_reward":True}
    with pytest.raises(ValueError,match="requires saved EMA"):
        execute(preflight,namespace)
    namespace["args"].resume=False  # initialize-from needs no saved course.
    execute(preflight,namespace)
    namespace["env"].umi_pose_reward=UmiPoseReward()
    namespace["env"].umi_pose_reward.update(torch.zeros(2),torch.zeros(2))
    execute(save,namespace)
    saved=namespace["checkpoint"]["umi_pose_reward"].copy()
    namespace["env"].umi_pose_reward=UmiPoseReward()
    execute(restore,namespace)
    restored=namespace["env"].umi_pose_reward.state_dict()
    assert restored["position_ema_m"]==saved["position_ema_m"]
    assert restored["orientation_ema_rad"]==saved["orientation_ema_rad"]
    assert (restored["position_sigma_m2"],restored["orientation_sigma_rad"])==(1,4)
