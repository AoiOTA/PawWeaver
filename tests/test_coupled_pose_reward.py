import pytest
import torch

from pawweaver.task import reward_terms,sum_reward_terms


def test_joint_pose_reward_requires_both_errors_small():
    # Exact existing widths: good pose, poor position, poor orientation.
    position=torch.exp(-torch.tensor([0.,.6,0.]).square()/.15**2)
    orientation=torch.exp(-torch.tensor([0.,0.,2.]).square()/.5**2)
    terms={"tracking":position,"orientation_tracking":orientation}
    weights={"tracking":2.,"orientation_tracking":1.}
    old=sum_reward_terms(terms,weights)
    coupled=sum_reward_terms(terms,weights,coupled_pose=True)
    assert coupled[0]==old[0]==3
    assert torch.all(coupled[1:]<1e-6)
    assert old[1]>=1 and old[2]>=2


def test_default_sum_preserves_other_rewards_and_separate_termination():
    terms={"tracking":torch.tensor([.7,.3]),"orientation_tracking":torch.tensor([.2,.8]),
           "foot_slip":torch.tensor([.05,.1]),"termination":torch.ones(2)}
    weights={"tracking":2.,"orientation_tracking":1.,"foot_slip":-.1,"termination":-5.}
    previous=sum(weights[name]*value for name,value in terms.items() if name!="termination")
    assert torch.equal(sum_reward_terms(terms,weights),previous)
    assert torch.equal(sum_reward_terms(terms,weights,coupled_pose=False),previous)
    expected=3*terms["tracking"]*terms["orientation_tracking"]-.1*terms["foot_slip"]
    torch.testing.assert_close(sum_reward_terms(terms,weights,coupled_pose=True),expected)


@pytest.mark.parametrize("coupled",[False,True])
@pytest.mark.parametrize("missing",["tracking","orientation_tracking","foot_slip"])
def test_missing_required_reward_weight_still_fails(coupled,missing):
    terms={name:torch.ones(1) for name in ("tracking","orientation_tracking","foot_slip")}
    weights={"tracking":2.,"orientation_tracking":1.,"foot_slip":-.1}
    del weights[missing]
    with pytest.raises(KeyError,match=missing):
        sum_reward_terms(terms,weights,coupled_pose=coupled)


@pytest.fixture
def reward_inputs():
    vectors=torch.zeros(3,3)
    joints=torch.zeros(3,18)
    return dict(error=torch.tensor([[0.,0.,0.],[.15,0.,0.],[.45,0.,0.]]),
        orientation_error=torch.tensor([0.,.2,.5]),previous_error=torch.zeros(3),
        tcp_velocity=vectors,goal_velocity=vectors,action=joints,previous_action=joints,
        torque=joints,effort=torch.ones_like(joints),q=joints,qd=joints,previous_qd=joints,
        lower=-torch.ones_like(joints),upper=torch.ones_like(joints),gravity_b=vectors,
        foot_velocity=torch.zeros(3,4,3),foot_contact=torch.zeros(3,4),
        collision=torch.zeros(3,dtype=torch.bool),fallen=torch.zeros(3,dtype=torch.bool))


def test_position_width_preserves_default_and_changes_only_position_kernel(reward_inputs):
    default=reward_terms(**reward_inputs)
    explicit=reward_terms(**reward_inputs,tracking_width=.15)
    wider=reward_terms(**reward_inputs,tracking_width=.45)
    for name in default:
        assert torch.equal(default[name],explicit[name])
        if name!="tracking":
            assert torch.equal(default[name],wider[name])
    torch.testing.assert_close(default["tracking"],torch.exp(-torch.tensor([0.,1.,9.])))
    torch.testing.assert_close(wider["tracking"],torch.exp(-torch.tensor([0.,1/9,1.])))
    assert torch.all(wider["tracking"][1:]>default["tracking"][1:])


def test_optional_joint_margin_is_range_normalized_and_has_restoring_gradient(reward_inputs):
    q=reward_inputs['q'].clone()
    q[:,:6]=torch.tensor([-1.2,-1.,-.8,0.,.8,1.])
    q.requires_grad_()
    inputs=dict(reward_inputs,q=q)
    default=reward_terms(**inputs)
    explicit=reward_terms(**inputs,joint_limit_margin_fraction=None,foot_clearance=None)
    assert default.keys()==explicit.keys()
    for name in default:
        assert torch.equal(default[name],explicit[name])
    assert torch.equal(default['joint_limit'],
        ((inputs['lower']+.02-q).clamp_min(0)+(q-inputs['upper']+.02).clamp_min(0)).sum(-1))
    result=reward_terms(**inputs,joint_limit_margin_fraction=.1)
    # At each hard bound the corresponding soft-margin penalty is one.
    torch.testing.assert_close(result['joint_limit'],torch.full((3,),6.))
    result['joint_limit'].sum().backward()
    torch.testing.assert_close(q.grad[:,:6],torch.tensor([-20.,-10.,0.,0.,0.,10.]).expand(3,-1),atol=1e-5,rtol=0)
    assert torch.count_nonzero(q.grad[:,6:])==0
    changed=reward_terms(**dict(inputs,q=q.detach()*3+.7,
        lower=inputs['lower']*3+.7,upper=inputs['upper']*3+.7),joint_limit_margin_fraction=.1)
    torch.testing.assert_close(changed['joint_limit'],result['joint_limit'])
    for name in default:
        if name!='joint_limit':
            assert torch.equal(default[name],result[name])


@pytest.mark.parametrize('margin',[0.,-.1,.5,1.,float('nan'),float('inf')])
def test_invalid_optional_joint_margin_fails(reward_inputs,margin):
    with pytest.raises(ValueError,match='Joint limit margin fraction'):
        reward_terms(**reward_inputs,joint_limit_margin_fraction=margin)


def test_unloaded_foot_height_uses_clearance_not_contact_count(reward_inputs):
    clearance=torch.tensor([[-.1,0.,.2,.3]]).repeat(3,1).requires_grad_()
    contact=torch.tensor([[False,False,False,True]]).repeat(3,1)
    terms=reward_terms(**dict(reward_inputs,foot_contact=contact),foot_clearance=clearance)
    torch.testing.assert_close(terms['unloaded_foot_height'],torch.full((3,),.04))
    terms['unloaded_foot_height'].sum().backward()
    torch.testing.assert_close(clearance.grad,torch.tensor([[0.,0.,.4,0.]]).expand(3,-1))
    # Old weight dictionaries still work because the optional term is absent by default.
    old=reward_terms(**reward_inputs)
    assert 'unloaded_foot_height' not in old
    weights={name:1. for name in old}
    assert torch.isfinite(sum_reward_terms(old,weights)).all()


@pytest.mark.parametrize("width",[0.,-.1,float("nan"),float("inf"),-float("inf")])
def test_position_width_rejects_nonpositive_or_nonfinite(reward_inputs,width):
    with pytest.raises(ValueError,match="Position reward width must be finite and positive"):
        reward_terms(**reward_inputs,tracking_width=width)


@pytest.mark.parametrize("config,expected_width",[({},.15),({"tracking_width_m":.15},.15),
                                                ({"tracking_width_m":.45},.45)])
def test_isaac_step_passes_position_width_to_actual_reward(monkeypatch,config,expected_width):
    from types import SimpleNamespace
    from unittest.mock import Mock
    pytest.importorskip("tensordict")
    from pawweaver import isaac_env
    env=isaac_env.WholeBodyEnv.__new__(isaac_env.WholeBodyEnv)
    joints=torch.zeros(1,18)
    vector=torch.zeros(1,3)
    quat=torch.tensor([[1.,0.,0.,0.]])
    state=SimpleNamespace(base_pos_w=torch.tensor([[0.,0.,.3]]),base_quat_w=quat,
        tcp_pos_w=vector,tcp_quat_w=quat,joint_pos=joints,joint_vel=joints)
    env.config=dict(config,domain_randomization=False)
    env.termination=isaac_env.termination_config(env.config)
    # Exercise the actual step->reward call on fixed CPU state, without physics.
    env.spec=SimpleNamespace(decimation=0)
    env.num_envs=1
    env.device="cpu"
    env.diagnostic=False
    env.state=lambda:state
    env.episode_length_buf=torch.zeros(1,dtype=torch.long)
    env.reference=SimpleNamespace(current=lambda _:torch.tensor([[.45,0.,0.]]),
                                  current_orientation=lambda _:quat)
    env.observations=Mock()
    env.scene=SimpleNamespace(sensors={},env_origins=vector)
    env.robot=SimpleNamespace(body_names=["foot"],data=SimpleNamespace(
        body_link_lin_vel_w=SimpleNamespace(torch=torch.zeros(1,1,3))))
    env.foot_ids=[0]
    env.contacts=torch.zeros(1,1)
    env.pd=SimpleNamespace(command=lambda _:None,last_action=joints,effort=torch.ones(18),
        lower=-torch.ones(18),upper=torch.ones(18))
    env.previous_error=torch.zeros(1)
    env.previous_tcp=vector
    env.previous_action=env.torque=env.previous_qd=joints
    class RewardReached(Exception):
        pass
    def capture(**kwargs):
        assert kwargs["tracking_width"]==expected_width
        actual=reward_terms(**kwargs)["tracking"]
        torch.testing.assert_close(actual,torch.exp(-torch.tensor([.45**2/expected_width**2])))
        raise RewardReached
    monkeypatch.setattr(isaac_env,"reward_terms",capture)
    with pytest.raises(RewardReached):
        env.step(joints,auto_reset=False)
