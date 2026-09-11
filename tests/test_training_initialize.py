"""Execute trainer checkpoint preflight and transfer with actual CPU RSL-RL models."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from pawweaver.learning import AuxiliaryPPO, WholeBodyActor, validate_leg_mean_config
from pawweaver.observations import observation_spec
from pawweaver.training_inputs import check_training_identity

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/train.py'


def trainer_code():
    tree = ast.parse(SOURCE.read_text())
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'checkpoint')
    stop = next(i for i, node in enumerate(tree.body)
                if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'launcher')
    body = next(node.body for node in tree.body if isinstance(node, ast.Try))
    transfer = next(node for node in body if isinstance(node, ast.If)
                    and ast.unparse(node.test) == 'args.initialize_from')
    return [compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec')
            for nodes in (tree.body[start:stop], [transfer])]


def algorithm_for(config):
    dim = observation_spec(config).size
    obs = TensorDict({'policy': torch.randn(4, dim), 'critic': torch.randn(4, dim+110)}, [4])
    actor = WholeBodyActor(obs, {'actor': ['policy']}, 'actor', 18, observation_dim=dim,
        prediction=False, velocity=False, hidden_dims=config['actor_hidden_dims'],
        leg_mean_transform=config.get('leg_mean_transform', 'identity'), obs_normalization=True,
        distribution_cfg={'class_name': 'rsl_rl.modules.distribution:GaussianDistribution', 'init_std': .5})
    critic = MLPModel(obs, {'critic': ['critic']}, 'critic', 1, hidden_dims=[16], obs_normalization=True)
    return AuxiliaryPPO(actor, critic, RolloutStorage('rl', 4, 2, obs, [18], 'cpu'),
                        learning_rate=config['learning_rate']), obs


@pytest.fixture
def transfer_case(tmp_path):
    config = {'task_mode': 'velocity_ee_pose', 'trajectory_prediction': False,
              'velocity_estimation': False, 'actor_hidden_dims': [16],
              'learning_rate': .001, 'demonstrations': [f'neutral{i}.npz' for i in range(7)]}
    metadata = {'config': config, 'observation': observation_spec(config).to_dict(),
                'asset_hash': 'same-asset', 'diagnostic': True,
                'provisional_spec': {'actuators': {'kp': [30.], 'default_pos': [.1]}}}
    prior, obs = algorithm_for(config)
    prior.actor.update_normalization(obs)
    prior.critic.update_normalization(obs)
    # Populate real Adam state and change weights/std so restore is observable.
    sum(parameter.square().sum() for parameter in prior.actor.parameters()).backward()
    prior.optimizer.step()
    checkpoint = {'metadata': metadata, 'algorithm': copy.deepcopy(prior.save())}
    assert checkpoint['algorithm']['optimizer_state_dict']['state']
    config = dict(config, learning_rate=.0001, demonstrations=[f'full{i}.npz' for i in range(28)])
    current = dict(copy.deepcopy(metadata), config=config)
    return tmp_path/'source.pt', checkpoint, config, current


def initialize(case, *, resume=False):
    path, checkpoint, config, metadata = case
    torch.save(checkpoint, path)
    namespace = {'torch': torch, 'config': config, 'metadata': metadata,
        'args': SimpleNamespace(resume=path if resume else None, initialize_from=None if resume else path),
        'observation_spec': observation_spec, 'validate_leg_mean_config': validate_leg_mean_config,
        'check_training_identity': check_training_identity}
    preflight, transfer = trainer_code()
    exec(preflight, namespace)
    algorithm, obs = algorithm_for(config)
    namespace['algorithm'] = algorithm
    exec(transfer, namespace)
    return algorithm, obs


def test_same_279_transfer_restores_models_std_normalizers_with_fresh_optimizer(transfer_case):
    algorithm, _ = initialize(transfer_case)
    checkpoint = transfer_case[1]
    for model, key in ((algorithm.actor, 'actor_state_dict'), (algorithm.critic, 'critic_state_dict')):
        for name, value in model.state_dict().items():
            assert torch.equal(value, checkpoint['algorithm'][key][name]), name
    assert not algorithm.optimizer.state
    assert algorithm.optimizer.param_groups[0]['lr'] == .0001
    assert algorithm.learning_rate == .0001
    with pytest.raises(ValueError, match='Resume config differs'):
        initialize(transfer_case, resume=True)


@pytest.mark.parametrize('field', ['target_frame', 'goal_history_frame', 'velocity_command_order', 'control_dt', 'task_mode'])
def test_transfer_rejects_different_observation_contract(transfer_case, field):
    transfer_case[1]['metadata']['observation'][field] = 'incompatible'
    with pytest.raises(ValueError, match='Checkpoint observation contract differs'):
        initialize(transfer_case)


@pytest.mark.parametrize('field', ['asset_hash', 'diagnostic', 'provisional_spec'])
def test_transfer_preserves_hardware_and_mode_identity(transfer_case, field):
    transfer_case[1]['metadata'][field] = None
    with pytest.raises(ValueError, match='Checkpoint'):
        initialize(transfer_case)


@pytest.mark.parametrize('field', ['actor_hidden_dims', 'trajectory_prediction', 'velocity_estimation'])
def test_transfer_preserves_architecture(transfer_case, field):
    transfer_case[1]['metadata']['config'][field] = [32] if field == 'actor_hidden_dims' else True
    with pytest.raises(ValueError, match='architecture differs'):
        initialize(transfer_case)


def test_279_transfer_rejects_mean_transform_change(transfer_case):
    transfer_case[2]['leg_mean_transform'] = 'softsign'
    with pytest.raises(ValueError, match='same leg_mean_transform'):
        initialize(transfer_case)


@pytest.mark.parametrize('damage', ['missing_weight', 'missing_marker'])
def test_transfer_strict_state_and_actor_contract_checks(transfer_case, damage):
    state = transfer_case[1]['algorithm']['actor_state_dict']
    state.pop('mlp.0.weight' if damage == 'missing_weight' else 'observation_contract_dim')
    with pytest.raises((RuntimeError, ValueError), match='Missing key|observation contract differs'):
        initialize(transfer_case)
