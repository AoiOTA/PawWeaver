"""Both explicit policy contracts, bundle rejection and initialization units."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
pytest.importorskip('rsl_rl')
from tensordict import TensorDict
from pawweaver.learning import WholeBodyActor,AuxiliaryPPO,initialize_fresh_actor
from pawweaver.observations import CommandObservationSpec,ObservationSpec
from pawweaver.bundle import export_bundle,load_bundle
from pawweaver.contracts import ActuatorSpec,JOINT_NAMES,canonical_hash


def actor_for(dim=279,velocity=False,normalization=True):
    torch.manual_seed(91)
    obs=TensorDict({'policy':torch.randn(4,dim)},[4])
    actor=WholeBodyActor(obs,{'actor':['policy']},'actor',18,observation_dim=dim,
        prediction=False,velocity=velocity,hidden_dims=[16],obs_normalization=normalization,
        distribution_cfg={'class_name':'rsl_rl.modules.distribution:GaussianDistribution','init_std':.5})
    return actor,obs


def actuator_spec():
    return ActuatorSpec(JOINT_NAMES,(0.,)*18,(.2,)*12+(2.,)*6,(10.,)*18,(1.,)*18,
        (-2.,)*18,(2.,)*18,(5.,)*18,(4.,)*18,(0.,)*18,(0.,)*18,(0.,)*18,(0,)*18)


def test_command_actor_eager_jit_and_direct_checkpoint_contract():
    actor,obs=actor_for()
    actor.update_normalization(obs)
    assert actor.obs_normalizer.mean.shape==(279,)
    assert actor.mlp[0].in_features==279 and actor(obs).shape==(4,18)
    changed=obs.clone();changed['policy'][:,-3:]+=.3
    assert not torch.equal(actor(obs),actor(changed))
    jit=torch.jit.script(actor.as_jit().eval())
    torch.testing.assert_close(jit(obs['policy']),actor(obs))
    with pytest.raises((ValueError,torch.jit.Error),match='279'):
        jit(torch.zeros(4,276))
    restored,_=actor_for();restored.load_state_dict(actor.state_dict())
    torch.testing.assert_close(restored(obs),actor(obs))
    broken=copy.deepcopy(actor.state_dict());broken['obs_normalizer._mean']=torch.zeros(1,276)
    with pytest.raises(ValueError,match='normalizer'):
        restored.load_state_dict(broken)
    # Same first-layer width cannot conceal the old raw input contract, even
    # without normalizer buffers to distinguish them.
    old,_=actor_for(276,velocity=True,normalization=False)
    new,_=actor_for(279,normalization=False)
    assert old.mlp[0].in_features==new.mlp[0].in_features==279
    assert 'observation_contract_dim' not in old.state_dict()
    for destination,source in [(old,new),(new,old)]:
        before=copy.deepcopy(destination.state_dict())
        with pytest.raises(ValueError,match='contract'):
            destination.load_state_dict(source.state_dict())
        for name,value in before.items():
            assert torch.equal(value,destination.state_dict()[name])
    with pytest.raises(ValueError,match='prediction=false'):
        actor_for(279,velocity=True)


def test_command_actor_runs_existing_ppo_without_auxiliary_labels():
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    actor,obs=actor_for()
    obs['critic']=torch.randn(4,389)
    critic=MLPModel(obs,{'critic':['critic']},'critic',1,hidden_dims=[16])
    algorithm=AuxiliaryPPO(actor,critic,RolloutStorage('rl',4,2,obs,[18],'cpu'),
                           num_learning_epochs=1,num_mini_batches=1)
    before=actor.mlp[-1].weight.detach().clone()
    with torch.no_grad():
        for _ in range(2):
            assert algorithm.act(obs).shape==(4,18)
            algorithm.process_env_step(obs,torch.randn(4),torch.zeros(4,dtype=torch.bool),{})
        algorithm.compute_returns(obs)
    losses=algorithm.update()
    assert losses['auxiliary']==0.
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.equal(before,actor.mlp[-1].weight)


def test_command_bundle_roundtrip_and_world_command_mutual_rejection(tmp_path):
    actor,obs=actor_for();spec=actuator_spec()
    asset={'asset_hash':'command-fixture','ready_for_training':False}
    config={'task_mode':'velocity_ee_pose','trajectory_prediction':False,'velocity_estimation':False}
    manifest=export_bundle(actor,spec,asset,config,tmp_path/'command',trained=False)
    assert manifest['observation']==CommandObservationSpec().to_dict()
    loaded,_,_=load_bundle(tmp_path/'command',asset['asset_hash'],False,observation_spec=CommandObservationSpec())
    torch.testing.assert_close(loaded(obs['policy']),actor(obs))
    with pytest.raises(ValueError,match='observation contract'):
        load_bundle(tmp_path/'command',asset['asset_hash'],False)
    old,_=actor_for(276)
    export_bundle(old,spec,asset,{},tmp_path/'world',trained=False)
    with pytest.raises(ValueError,match='observation contract'):
        load_bundle(tmp_path/'world',asset['asset_hash'],False,observation_spec=CommandObservationSpec())
    for wrong_actor,wrong_config in [(old,config),(actor,{})]:
        with pytest.raises(ValueError,match='observation contract'):
            export_bundle(wrong_actor,spec,asset,wrong_config,tmp_path/'wrong',trained=False)
    assert not (tmp_path/'wrong').exists()
    # Config and explicitly requested contract must also agree on disk.
    manifest.pop('bundle_hash');manifest['training_config']={}
    manifest['bundle_hash']=canonical_hash(manifest)
    (tmp_path/'command/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='config and observation'):
        load_bundle(tmp_path/'command',asset['asset_hash'],False,observation_spec=CommandObservationSpec())


def test_fresh_joint_std_radians_zero_mean_and_checkpoint_skip():
    actor,obs=actor_for();scale=actuator_spec().action_scale
    before=copy.deepcopy(actor.state_dict())
    initialize_fresh_actor(actor,{},scale)
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
    config={'initial_joint_std_rad':[.05]*12+[.01]*6,'zero_initial_actor_mean':True}
    source=Path(__file__).resolve().parents[1]/'scripts/train.py'
    tree=ast.parse(source.read_text())
    branch=next(node for node in ast.walk(tree) if isinstance(node,ast.If) and ast.unparse(node.test)=='checkpoint is None')
    code=compile(ast.Module(body=[branch],type_ignores=[]),str(source),'exec')
    namespace={'actor':actor,'config':config,'env':SimpleNamespace(spec=actuator_spec()),
               'checkpoint':None,'initialize_fresh_actor':initialize_fresh_actor}
    exec(code,namespace)
    torch.testing.assert_close(actor.distribution.std_param,torch.tensor([.25]*12+[.005]*6))
    assert actor(obs).eq(0).all()
    # A restored policy remains learned even if its config retains fresh knobs.
    with torch.no_grad():
        actor.mlp[-1].bias.fill_(.1);actor.distribution.std_param.fill_(.3)
    namespace['checkpoint']={'algorithm':{}}
    exec(code,namespace)
    assert actor.mlp[-1].bias.eq(.1).all() and actor.distribution.std_param.eq(.3).all()


@pytest.mark.parametrize('config',[{'initial_joint_std_rad':[.1]*17},
    {'initial_joint_std_rad':[.1]*17+[0.]},{'initial_joint_std_rad':[.1]*17+[float('nan')]},
    {'initial_joint_std_rad':[.1]*18,'zero_initial_actor_mean':'yes'}])
def test_invalid_fresh_configuration_fails_before_mutating_actor(config):
    actor,_=actor_for();before=copy.deepcopy(actor.state_dict())
    with pytest.raises(ValueError):
        initialize_fresh_actor(actor,config,actuator_spec().action_scale)
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())


def test_training_command_contract_allows_transfer_but_rejects_auxiliary_heads():
    source=Path(__file__).resolve().parents[1]/'scripts/train.py'
    tree=ast.parse(source.read_text())
    branch=next(node for node in tree.body if isinstance(node,ast.If) and ast.unparse(node.test)=='obs_spec.size == 279')
    code=compile(ast.Module(body=[branch],type_ignores=[]),str(source),'exec')
    namespace={'obs_spec':CommandObservationSpec(),'config':{},'args':SimpleNamespace(initialize_from=None)}
    exec(code,namespace)
    namespace['args'].initialize_from='old.pt'
    exec(code,namespace)
    namespace['args'].initialize_from=None;namespace['config']['velocity_estimation']=True
    with pytest.raises(ValueError,match='disabled'):
        exec(code,namespace)
