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
