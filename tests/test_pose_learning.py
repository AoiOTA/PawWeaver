"""CPU pose-policy learning/export compatibility, independent of simulator startup."""
import copy
import json
import pytest
import torch
pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from pawweaver.learning import WholeBodyActor,AuxiliaryPPO
from pawweaver.bundle import export_bundle,load_bundle
from pawweaver.contracts import ActuatorSpec,JOINT_NAMES,canonical_hash
from pawweaver.observations import ObservationSpec


def observations():
    return TensorDict({'policy':torch.randn(8,276),'critic':torch.randn(8,386),
        'velocity_label':torch.randn(8,3),'future_label':torch.randn(8,12),'future_valid':torch.ones(8,12)},[8])


def actor_for(obs):
    return WholeBodyActor(obs,{'actor':['policy']},'actor',18,hidden_dims=[32,16],obs_normalization=True,
        prediction=True,velocity=True,distribution_cfg={
            'class_name':'rsl_rl.modules.distribution:GaussianDistribution','init_std':.1})


def test_pose_actor_causal_features_jit_and_original_ppo():
    torch.manual_seed(132);torch.set_num_threads(2)
    obs=observations();actor=actor_for(obs);actor.update_normalization(obs)
    assert actor.obs_normalizer.mean.shape==(276,)
    assert actor.features.velocity_net[0].in_features==210
    assert actor.features.future_net[0].in_features==12
    assert actor.mlp[0].in_features==276+3+12
    output=actor(obs).detach()
    poisoned=obs.clone()
    for key in ('critic','velocity_label','future_label'): poisoned[key].fill_(10000)
    assert torch.equal(output,actor(poisoned).detach())
    jit=torch.jit.script(actor.as_jit().eval())
    torch.testing.assert_close(jit(obs['policy']),output,atol=1e-6,rtol=1e-6)
    with pytest.raises((torch.jit.Error,ValueError),match='276'):
        jit(torch.zeros(8,246))
    legacy=copy.deepcopy(actor.state_dict());legacy['mlp.0.weight']=legacy['mlp.0.weight'][:,:-30]
    with pytest.raises(ValueError,match='Position-only'):
        actor.load_state_dict(legacy,strict=True)
    bad=obs.clone();bad['policy']=torch.zeros(8,246)
    with pytest.raises(ValueError,match='276'): actor_for(bad)
    critic=MLPModel(obs,{'critic':['critic']},'critic',1,hidden_dims=[32])
    algorithm=AuxiliaryPPO(actor,critic,RolloutStorage('rl',8,4,obs,[18],'cpu'),
                           num_learning_epochs=2,num_mini_batches=2)
    before=actor.mlp[0].weight.detach().clone()
    with torch.no_grad():
        for _ in range(4):
            algorithm.act(obs);algorithm.process_env_step(obs,torch.randn(8),torch.zeros(8,dtype=torch.bool),{})
        algorithm.compute_returns(obs)
    losses=algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.equal(before,actor.mlp[0].weight)


def test_pose_bundle_version_is_separate_from_asset_version(tmp_path):
    actor=actor_for(observations())
    spec=ActuatorSpec(JOINT_NAMES,(0.,)*18,(.2,)*18,(10.,)*18,(1.,)*18,
        (-2.,)*18,(2.,)*18,(5.,)*18,(4.,)*18,(0.,)*18,(0.,)*18,(0.,)*18,(0,)*18)
    asset={'schema_version':1,'asset_hash':'software-pose-fixture','ready_for_training':False}
    metadata={'seed':17,'diagnostic':True,'provisional_spec':{'source':'software fixture'},
              'asset_hash':asset['asset_hash'],'large_diagnostic_field':['not runtime state']*1000}
    manifest=export_bundle(actor,spec,asset,{},tmp_path,trained=False,metadata=metadata)
    assert manifest['schema_version']==2 and manifest['observation']==ObservationSpec().to_dict()
    assert manifest['training_metadata']=={key:metadata[key] for key in ('seed','diagnostic','provisional_spec')}
    assert 'large_diagnostic_field' in metadata  # Export does not mutate the producer's full record.
    loaded,loaded_spec,loaded_manifest=load_bundle(tmp_path,asset['asset_hash'],require_trained=False)
    assert loaded_manifest['training_metadata']==manifest['training_metadata']
    assert loaded_spec==spec
    raw=torch.randn(2,276)
    torch.testing.assert_close(loaded(raw),actor.as_jit()(raw),atol=1e-6,rtol=1e-6)
    manifest.pop('bundle_hash');manifest['schema_version']=1
    manifest['bundle_hash']=canonical_hash(manifest)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='position-only'):
        load_bundle(tmp_path,asset['asset_hash'],require_trained=False)
