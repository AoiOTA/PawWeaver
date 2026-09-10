import pytest
import torch
pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from pawweaver.learning import WholeBodyActor, AuxiliaryPPO

def observations(n=8):
    # Actual combined-robot critic adds base velocity, 30 contacts, domain
    # parameters and 18-joint actuator randomization to the 276 policy values.
    return TensorDict({"policy":torch.randn(n,276),"critic":torch.randn(n,386),
        "velocity_label":torch.randn(n,3),"future_label":torch.randn(n,12),
        "future_valid":torch.ones(n,12)},batch_size=[n])

def actor_for(obs):
    return WholeBodyActor(obs,{"actor":["policy"]},"actor",18,prediction=True,
        hidden_dims=[64,32],obs_normalization=True,
        distribution_cfg={"class_name":"rsl_rl.modules.distribution:GaussianDistribution","init_std":.5})

def test_causality_export_and_auxiliary_ppo():
    torch.manual_seed(71)
    torch.set_num_threads(2)
    obs = observations()
    actor = actor_for(obs)
    actor.update_normalization(obs)
    original = actor(obs).detach()
    poisoned = obs.clone()
    for key in ("critic","future_label","velocity_label"):
        poisoned[key].fill_(1e5)
    assert torch.equal(original,actor(poisoned).detach())
    exported = torch.jit.script(actor.as_jit().eval())
    assert (original-exported(obs["policy"])).abs().max()<1e-5
    critic = MLPModel(obs,{"critic":["critic"]},"critic",1,hidden_dims=[32])
    storage = RolloutStorage("rl",8,8,obs,[18],"cpu")
    algorithm = AuxiliaryPPO(actor,critic,storage,num_learning_epochs=2,num_mini_batches=2)
    before = actor.features.future_net[0].weight.detach().clone()
    with torch.inference_mode():
        for _ in range(8):
            algorithm.act(obs)
            obs = observations()
            algorithm.process_env_step(obs,torch.randn(8),torch.zeros(8),{})
        algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert not torch.equal(before,actor.features.future_net[0].weight)
    assert storage.step==0


def ppo_for_update(algorithm_type=AuxiliaryPPO,leg_mean_transform="identity",**kwargs):
    torch.manual_seed(511)
    obs=observations()
    actor=WholeBodyActor(obs,{"actor":["policy"]},"actor",18,prediction=False,velocity=False,
        hidden_dims=[16],obs_normalization=True,leg_mean_transform=leg_mean_transform,
        distribution_cfg={"class_name":"rsl_rl.modules.distribution:GaussianDistribution","init_std":.5})
    critic=MLPModel(obs,{"critic":["critic"]},"critic",1,hidden_dims=[16])
    algorithm=algorithm_type(actor,critic,RolloutStorage("rl",8,4,obs,[18],"cpu"),
        num_learning_epochs=2,num_mini_batches=2,learning_rate=1e-5,**kwargs)
    with torch.no_grad():
        for _ in range(4):
            algorithm.act(obs)
            algorithm.process_env_step(obs,torch.randn(8),torch.zeros(8,dtype=torch.bool),{})
        algorithm.compute_returns(obs)
    return algorithm


def test_zero_mean_bound_skips_mean_access_and_preserves_update(monkeypatch):
    default=ppo_for_update()
    zero=ppo_for_update(leg_mean_bound_coef=0.)
    def forbidden_mean(_):
        raise AssertionError("Zero coefficient must not access the mean-bound calculation")
    monkeypatch.setattr(WholeBodyActor,"output_mean",property(forbidden_mean))
    torch.manual_seed(612)
    default_losses=default.update()
    default_rng=torch.get_rng_state()
    torch.manual_seed(612)
    zero_losses=zero.update()
    assert torch.equal(default_rng,torch.get_rng_state())
    assert default_losses==zero_losses
    assert zero_losses["leg_mean_bound"]==0
    for left,right in ((default.actor,zero.actor),(default.critic,zero.critic)):
        for name,value in left.state_dict().items():
            assert torch.equal(value,right.state_dict()[name])
    assert default.storage.step==zero.storage.step==0


@pytest.mark.parametrize("coefficient",[-.001,float("nan"),float("inf"),-float("inf")])
def test_mean_bound_rejects_invalid_coefficient(coefficient):
    with pytest.raises(ValueError,match="finite and nonnegative"):
        AuxiliaryPPO(leg_mean_bound_coef=coefficient)


@pytest.mark.parametrize('effective_bounds',[False,True])
def test_mean_bound_uses_live_means_in_actual_ppo_update(monkeypatch,effective_bounds):
    torch.manual_seed(613)
    obs=observations()
    actor=WholeBodyActor(obs,{"actor":["policy"]},"actor",18,prediction=False,velocity=False,
        hidden_dims=[16],obs_normalization=True,
        distribution_cfg={"class_name":"rsl_rl.modules.distribution:GaussianDistribution","init_std":.5})
    bias=torch.tensor([-2.,-.5,0.,.5,2.,-1.,1.,-3.,.8,-.8,0.,0.]+[4.]*6)
    critic=MLPModel(obs,{"critic":["critic"]},"critic",1,hidden_dims=[16])
    with torch.no_grad():
        actor.mlp[-1].weight.zero_()
        actor.mlp[-1].bias.copy_(bias)
        for parameter in critic.parameters():
            parameter.zero_()
    lower=torch.full((12,),-.25);upper=torch.full((12,),.4)
    algorithm=AuxiliaryPPO(actor,critic,RolloutStorage("rl",8,1,obs,[18],"cpu"),
        num_learning_epochs=1,num_mini_batches=1,learning_rate=1e-5,
        entropy_coef=0.,value_loss_coef=0.,leg_mean_bound_coef=.001,
        leg_mean_bounds=(lower,upper) if effective_bounds else None)
    with torch.no_grad():
        algorithm.act(obs)
        algorithm.process_env_step(obs,torch.zeros(8),torch.zeros(8,dtype=torch.bool),{})
        algorithm.compute_returns(obs)
    means=[]
    forward=actor.forward
    def record(*args,**kwargs):
        result=forward(*args,**kwargs)
        actor.output_mean.retain_grad()
        means.append(actor.output_mean)
        return result
    monkeypatch.setattr(actor,"forward",record)
    losses=algorithm.update()
    if effective_bounds:
        below=(lower-bias[:12]).clamp_min(0);above=(bias[:12]-upper).clamp_min(0)
        assert losses['leg_mean_bound']==pytest.approx(.001*(below.square()+above.square()).mean().item())
        expected=torch.cat((2*.001*(above-below)/(8*12),torch.zeros(6)))
        # Means already inside [-1,1] still get a restoring gradient at the tighter actual PD boundary.
        assert expected[1]<0 and expected[3]>0
    else:
        assert losses["leg_mean_bound"]==pytest.approx(.001*6/12)
        expected=2*.001*(bias.abs()-1).clamp_min(0)*bias.sign()/(8*12)
        expected[12:]=0
    torch.testing.assert_close(means[0].grad,expected[None].expand(8,-1))
    assert torch.count_nonzero(actor.distribution.std_param.grad)==0
    with torch.no_grad():
        actual=forward(obs)
        assert torch.isfinite(actual).all()
        assert torch.all(actor.mlp[-1].bias[[0,4,7]].abs()<bias[[0,4,7]].abs())
        assert torch.count_nonzero(actor.mlp[-1].weight)>0
    assert algorithm.storage.step==0


def test_effective_mean_bounds_match_actual_pd_target_interval():
    import json
    from pathlib import Path
    from pawweaver.contracts import ActuatorSpec
    from pawweaver.control import JointPD
    from pawweaver.learning import effective_pd_leg_mean_bounds,validate_leg_mean_config
    path=Path(__file__).resolve().parents[1]/'artifacts/runs/diagnostic_pose_learning/wbc_random_training/spec.json'
    pd=JointPD(ActuatorSpec.from_dict(json.loads(path.read_text())['actuators']),1,'cpu')
    lower,upper=effective_pd_leg_mean_bounds(pd)
    assert lower.shape==upper.shape==(12,)
    assert ((lower>-1)|(upper<1)).any()
    for bounds in (lower,upper):
        action=torch.zeros(1,18);action[0,:12]=bounds
        pd.command(action)
        torch.testing.assert_close(pd.target[0,:12],pd.default_pos[:12]+pd.action_scale[:12]*bounds)
        assert ((pd.target[0,:12]>=pd.lower[:12]-1e-6)&(pd.target[0,:12]<=pd.upper[:12]+1e-6)).all()
    assert validate_leg_mean_config({'leg_mean_bound_effective_pd':True})=='identity'
    with pytest.raises(ValueError,match='identity Gaussian'):
        validate_leg_mean_config({'leg_mean_bound_effective_pd':True,'leg_mean_transform':'softsign'})


def test_softsign_mean_distribution_and_derivative():
    from pawweaver.learning import LegSoftsignGaussianDistribution
    from pawweaver.policy import LegSoftsignMean
    logits=torch.tensor([[-100.,-2.,-1.,-.5,0.,.5,1.,2.,100.,-.1,.1,3.]+[2.]*6],requires_grad=True)
    mapped=LegSoftsignMean()(logits)
    assert torch.equal(mapped[:,12:],logits[:,12:])
    assert (mapped[:,:12].abs()<1).all()
    mapped.sum().backward()
    torch.testing.assert_close(logits.grad[:,:12],1/(1+logits.detach()[:,:12].abs()).square())
    assert torch.equal(logits.grad[:,12:],torch.ones(1,6))
    distribution=LegSoftsignGaussianDistribution(18,init_std=.5)
    distribution.update(logits)
    sample=distribution.sample()
    expected=torch.distributions.Normal(mapped,torch.full_like(mapped,.5))
    torch.testing.assert_close(distribution.log_prob(sample),expected.log_prob(sample).sum(-1))
    torch.testing.assert_close(distribution.entropy,expected.entropy().sum(-1))
    torch.testing.assert_close(distribution.params[0],mapped)
    torch.testing.assert_close(distribution.deterministic_output(logits),mapped)


def test_softsign_real_ppo_update_and_identity_default():
    import copy
    original=ppo_for_update()
    identity=ppo_for_update(leg_mean_transform="identity")
    assert all(torch.equal(v,identity.actor.state_dict()[k]) for k,v in original.actor.state_dict().items())
    torch.manual_seed(72);left=original.update()
    torch.manual_seed(72);right=identity.update()
    assert left==right
    assert all(torch.equal(v,identity.actor.state_dict()[k]) for k,v in original.actor.state_dict().items())
    algorithm=ppo_for_update(leg_mean_transform="softsign")
    algorithm.num_learning_epochs=algorithm.num_mini_batches=1
    before=algorithm.actor.mlp[0].weight.detach().clone()
    steps=[]
    algorithm.optimizer.register_step_post_hook(lambda *unused:steps.append(1))
    losses=algorithm.update()
    assert len(steps)==1 and all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert not torch.equal(before,algorithm.actor.mlp[0].weight)
    restored=ppo_for_update(leg_mean_transform="softsign")
    restored.load(copy.deepcopy(algorithm.save()),None,True)
    restored.learning_rate=restored.optimizer.param_groups[0]['lr']
    assert restored.learning_rate==algorithm.learning_rate
    assert all(torch.equal(v,restored.actor.state_dict()[k]) for k,v in algorithm.actor.state_dict().items())
    for key,state in algorithm.optimizer.state_dict()['state'].items():
        for name,value in state.items():
            actual=restored.optimizer.state_dict()['state'][key][name]
            assert torch.equal(value,actual) if isinstance(value,torch.Tensor) else value==actual


def test_mean_config_initialization_and_resume():
    from pawweaver.learning import validate_leg_mean_config
    raw={"actor_hidden_dims":[16]}
    soft=dict(raw,leg_mean_transform="softsign")
    assert validate_leg_mean_config(raw)=="identity"
    assert validate_leg_mean_config(soft,raw)=="softsign"
    assert validate_leg_mean_config(soft,soft,resume=True)=="softsign"
    with pytest.raises(ValueError,match="Resume config differs"):
        validate_leg_mean_config(soft,raw,resume=True)
    with pytest.raises(ValueError,match="softsign-to-identity"):
        validate_leg_mean_config(raw,soft)
    with pytest.raises(ValueError,match="Unsupported leg_mean_transform"):
        validate_leg_mean_config({"leg_mean_transform":"tanh"})
    # Transform adds no state keys: existing learned tensors load strictly.
    raw_ppo=ppo_for_update()
    soft_ppo=ppo_for_update(leg_mean_transform="softsign")
    soft_ppo.actor.load_state_dict(raw_ppo.actor.state_dict(),strict=True)
    broken=dict(raw_ppo.actor.state_dict());broken.pop("mlp.0.weight")
    with pytest.raises(RuntimeError,match="Missing key"):
        soft_ppo.actor.load_state_dict(broken,strict=True)
