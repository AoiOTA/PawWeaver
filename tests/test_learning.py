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


def ppo_for_update(algorithm_type=AuxiliaryPPO,**kwargs):
    torch.manual_seed(511)
    obs=observations()
    actor=WholeBodyActor(obs,{"actor":["policy"]},"actor",18,prediction=False,velocity=False,
        hidden_dims=[16],obs_normalization=True,
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


def test_mean_bound_uses_live_means_in_actual_ppo_update(monkeypatch):
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
    algorithm=AuxiliaryPPO(actor,critic,RolloutStorage("rl",8,1,obs,[18],"cpu"),
        num_learning_epochs=1,num_mini_batches=1,learning_rate=1e-5,
        entropy_coef=0.,value_loss_coef=0.,leg_mean_bound_coef=.001)
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
