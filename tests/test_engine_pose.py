"""Portable runtime TCP pose against independent canonical kinematics (CPU only)."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from pawweaver.assets.model import RobotTree
from pawweaver.contracts import JOINT_NAMES, FOOT_NAMES
from pawweaver.math import rpy_matrix,rpy_quat
from pawweaver.mujoco_runtime import MujocoRunner


ASSET = Path(__file__).resolve().parents[1] / "assets/generated/diagnostic"


@pytest.mark.skipif(not (ASSET / "robot.xml").exists(), reason="Build diagnostic asset first")
def test_runtime_tcp_world_orientation_matches_canonical_fk():
    # Exercise state(), not the bundle constructor: this checks the state contract
    # on the diagnostic geometry without granting it formal runtime acceptance.
    runner = MujocoRunner.__new__(MujocoRunner)
    runner.model = mujoco.MjModel.from_xml_path(str(ASSET / "robot.xml"))
    runner.data = mujoco.MjData(runner.model)
    runner.q_indices = [runner.model.joint(name).qposadr[0] for name in JOINT_NAMES]
    runner.v_indices = [runner.model.joint(name).dofadr[0] for name in JOINT_NAMES]
    tree = RobotTree.load(ASSET / "robot.urdf")
    rng = np.random.default_rng(223)
    for _ in range(12):
        root = np.eye(4)
        root[:3, :3] = rpy_matrix(rng.uniform(-1, 1, 3))
        root[:3, 3] = rng.uniform(-1, 1, 3)
        runner.data.qpos[:3] = root[:3, 3]
        mujoco.mju_mat2Quat(runner.data.qpos[3:7], root[:3, :3].ravel())
        joints = {}
        for name, index in zip(JOINT_NAMES, runner.q_indices):
            joint = runner.model.joint(name)
            joints[name] = rng.uniform(*runner.model.jnt_range[joint.id])
            runner.data.qpos[index] = joints[name]
        mujoco.mj_forward(runner.model, runner.data)
        state = runner.state()
        actual = np.empty(9)
        mujoco.mju_quat2Mat(actual, state.tcp_quat_w[0].numpy().astype(np.float64))
        expected = tree.forward(joints, root)["tcp"]
        assert np.max(np.abs(actual.reshape(3, 3) - expected[:3, :3])) < 3e-7
        assert np.max(np.abs(state.tcp_pos_w[0].numpy() - expected[:3, 3])) < 2e-7


def test_runtime_rejects_position_without_orientation_before_stepping():
    runner = MujocoRunner.__new__(MujocoRunner)
    for call in (runner.reset, runner.step):
        with pytest.raises(ValueError, match="both position and world-frame orientation"):
            call([0., 0., .5])


@pytest.mark.parametrize('swapped', [False, True])
def test_contact_snapshot_sums_world_vectors_and_counts_only_nonfoot_floor_pairs(monkeypatch, swapped):
    runner = MujocoRunner.__new__(MujocoRunner)
    # Deliberately different body/geom orders; two foot contacts cancel while
    # an internal arm/base collision contributes force but no ground count.
    names = ['world', 'arm_link', FOOT_NAMES[2], 'base_link']
    runner.model = SimpleNamespace(nbody=4, geom_bodyid=np.array([1, 0, 3, 2]),
        body=lambda index: SimpleNamespace(name=names[index]),
        geom=lambda name: SimpleNamespace(id={'floor': 1}[name]))
    frame = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    pairs = [(1, 3), (1, 3), (0, 2), (1, 2)]
    world_forces = np.array([[3., 4., 0.], [-3., -4., 0.], [0., 0., 5.], [1., 2., 2.]])
    if swapped:
        pairs = [(b, a) for a, b in pairs]
        world_forces *= -1
    runner.data = SimpleNamespace(ncon=len(pairs), contact=[
        SimpleNamespace(geom1=a, geom2=b, frame=frame.ravel()) for a, b in pairs])
    calls = []
    def contact_force(model, data, index, wrench):
        assert model is runner.model and data is runner.data
        calls.append(index)
        wrench[:] = np.r_[frame @ world_forces[index], [100., 200., 300.]]
    monkeypatch.setattr(mujoco, 'mj_contactForce', contact_force)
    forces, count = runner._contact_snapshot()
    np.testing.assert_allclose(forces, [3., 5., 0., np.sqrt(54.)])
    assert count == 1 and calls == list(range(4))
    runner.data.ncon = 0
    forces, count = runner._contact_snapshot()
    np.testing.assert_array_equal(forces, np.zeros(4))
    assert count == 0


def test_isaac_state_composes_xyzw_gripper_with_nonidentity_tcp_mount():
    pytest.importorskip("tensordict")
    from pawweaver.isaac_env import WholeBodyEnv
    env = WholeBodyEnv.__new__(WholeBodyEnv)
    env.num_envs, env.gripper_id, env.joint_ids = 1, 0, list(range(18))
    gripper_rpy, mount_rpy = [.3, -.5, .7], [-.2, .4, -.6]
    gripper_q = rpy_quat(gripper_rpy)
    gripper_xyz = np.array([.1, -.2, .3])
    env.tcp_offset = torch.tensor([.03, -.02, .14])
    env.tcp_rotation = torch.tensor(rpy_quat(mount_rpy), dtype=torch.float32)
    pose = torch.tensor(np.r_[gripper_xyz, gripper_q[[1,2,3,0]]], dtype=torch.float32)[None]
    fields = {"root_link_pose_w":pose, "body_link_pose_w":pose[:,None,:],
              "joint_pos":torch.zeros(1,18), "joint_vel":torch.zeros(1,18),
              "root_link_ang_vel_b":torch.zeros(1,3), "root_link_lin_vel_b":torch.zeros(1,3)}
    env.robot = SimpleNamespace(data=SimpleNamespace(**{k:SimpleNamespace(torch=v) for k,v in fields.items()}))
    state = env.state()
    actual_rotation = np.empty(9)
    mujoco.mju_quat2Mat(actual_rotation, state.tcp_quat_w[0].numpy().astype(np.float64))
    expected_rotation = rpy_matrix(gripper_rpy) @ rpy_matrix(mount_rpy)
    expected_position = gripper_xyz + rpy_matrix(gripper_rpy) @ env.tcp_offset.numpy()
    assert np.max(np.abs(actual_rotation.reshape(3,3)-expected_rotation)) < 3e-7
    assert np.max(np.abs(state.tcp_pos_w[0].numpy()-expected_position)) < 1e-7


@pytest.fixture
def diagnostic_bundle(tmp_path):
    from pawweaver.bundle import export_bundle
    from pawweaver.training_inputs import training_inputs
    spec_path = ASSET.parents[2] / 'configs/diagnostic_actuators.json'
    manifest, spec, provisional = training_inputs(ASSET, diagnostic=True, provisional_spec=spec_path)
    policy = torch.nn.Linear(276, 18)
    torch.nn.init.zeros_(policy.weight)
    torch.nn.init.zeros_(policy.bias)
    actor = SimpleNamespace(obs_dim=276, as_jit=lambda: policy)
    bundle = tmp_path / 'bundle'
    export_bundle(actor, spec, manifest, {'initial_base_height_m': .31762150697067526}, bundle,
                  trained=False, metadata={'diagnostic': True, 'provisional_spec': provisional, 'seed': 0})
    return bundle, spec_path


def test_diagnostic_runtime_compiles_passive_constants_and_resets_pose(diagnostic_bundle):
    import xml.etree.ElementTree as ET
    bundle, spec_path = diagnostic_bundle
    runner = MujocoRunner(ASSET, bundle, diagnostic=True, provisional_spec=spec_path)
    # Independent construction from the existing corrected hold path catches a
    # post-compilation armature patch even when dof_armature itself looks right.
    root = ET.parse(ASSET / 'robot.xml').getroot()
    root.find('compiler').set('meshdir', str(ASSET / 'meshes'))
    root.find(".//body[@name='base_link']").set('pos', '0 0 .31762150697067526')
    for i, name in enumerate(JOINT_NAMES):
        for field in ('armature', 'damping', 'frictionloss'):
            root.find(f".//joint[@name='{name}']").set(field, str(getattr(runner.spec, field)[i]))
    expected = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    np.testing.assert_array_equal(runner.model.dof_invweight0, expected.dof_invweight0)
    for field in ('armature', 'damping', 'frictionloss'):
        np.testing.assert_array_equal(getattr(runner.model, 'dof_' + field)[runner.v_indices], getattr(runner.spec, field))
    np.testing.assert_array_equal(runner.data.qpos[runner.q_indices], runner.spec.default_pos)
    assert runner.data.qpos[2] == .31762150697067526
    runner.data.qpos[2] = 4.
    runner.reset()
    assert runner.data.qpos[2] == .31762150697067526
    assert runner.bundle['trained'] is False and runner.diagnostic
    assert runner.policy(torch.zeros(1, 276)).shape == (1, 18)


@pytest.mark.parametrize('kwargs,match', [
    ({}, 'Diagnostic/source-only'),
    ({'diagnostic': True}, 'requires --provisional-spec'),
    ({'provisional_spec': Path('unused')}, 'requires --provisional-spec'),
    ({'software_fixture': True, 'diagnostic': True, 'provisional_spec': Path('unused')}, 'mutually exclusive'),
])
def test_diagnostic_runtime_keeps_explicit_mode_gates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MujocoRunner(ASSET, Path('unused'), **kwargs)


@pytest.mark.parametrize('change,match', [('provenance', 'provenance differs'), ('actuators', 'actuator spec differs'),
                                         ('mode', 'mode differs'), ('height', 'height must be finite')])
def test_diagnostic_runtime_rejects_incompatible_bundle(diagnostic_bundle, change, match):
    import json
    from pawweaver.contracts import canonical_hash
    bundle, spec_path = diagnostic_bundle
    path = bundle / 'manifest.json'
    manifest = json.loads(path.read_text())
    if change == 'provenance':
        manifest['training_metadata']['provisional_spec']['sources']['extra'] = 'changed'
    elif change == 'actuators':
        manifest['actuators']['kp'][0] += 1
    elif change == 'mode':
        manifest['training_metadata']['diagnostic'] = False
    else:
        manifest['training_config']['initial_base_height_m'] = -1.
    manifest.pop('bundle_hash')
    manifest['bundle_hash'] = canonical_hash(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        MujocoRunner(ASSET, bundle, diagnostic=True, provisional_spec=spec_path)


@pytest.mark.parametrize('fall', [False, True])
def test_diagnostic_evaluation_archives_elapsed_pose_and_early_fall(diagnostic_bundle, tmp_path, monkeypatch, fall):
    import json
    from pawweaver.trajectories import Trajectory
    bundle, spec_path = diagnostic_bundle
    runner = MujocoRunner(ASSET, bundle, diagnostic=True, provisional_spec=spec_path)
    # Interface test only: no policy-driven dynamics or training-suite evaluation.
    def step(*args, **kwargs):
        runner.data.time += .02
        if fall:
            runner.data.qpos[2] = .1
        mujoco.mj_forward(runner.model, runner.data)
        return np.zeros(276), np.zeros(18)
    monkeypatch.setattr(runner, 'step', step)
    snapshots = []
    def contact_snapshot():
        # Persist a distinguishable value for every actual runtime body column
        # at each control tick, including the early-fall tick.
        values = np.arange(runner.model.nbody) + runner.data.time
        snapshots.append(values)
        return values, len(snapshots)
    monkeypatch.setattr(runner, '_contact_snapshot', contact_snapshot)
    state = runner.state()
    trajectory = Trajectory(np.array([5., 5.06]), np.repeat(state.tcp_pos_w.numpy(), 2, axis=0),
                            np.repeat(state.tcp_quat_w.numpy(), 2, axis=0), {'family': 'reach', 'case_id': 'test'})
    result = runner.evaluate(trajectory, tmp_path / 'evaluation')
    expected = .02 if fall else .06
    assert result['elapsed_seconds'] == pytest.approx(expected)
    assert result['fallen'] == fall and result['diagnostic'] and not result['trained']
    assert json.loads((tmp_path / 'evaluation/metrics.json').read_text()) == result
    with np.load(tmp_path / 'evaluation/trace.npz', allow_pickle=False) as trace:
        assert trace['times'][-1] == pytest.approx(expected)
        assert trace['tcp_quat_w'].shape == (1 if fall else 3, 4)
        np.testing.assert_array_equal(trace['contact_body_names'],
            [runner.model.body(index).name for index in range(runner.model.nbody)])
        np.testing.assert_array_equal(trace['contacts'], snapshots)
        np.testing.assert_array_equal(trace['nonfoot_ground_contact_count'], np.arange(1, len(snapshots)+1))
        for name in FOOT_NAMES:
            index = runner.model.body(name).id
            assert trace['contact_body_names'][index] == name
            np.testing.assert_allclose(trace['contacts'][:, index], index + trace['times'])
        from pawweaver.task import episode_metrics
        unchanged = episode_metrics(trace['times'], trace['errors'], trace['base'], trace['torques'],
            trace['velocities'], fall, orientation_errors=trace['orientation_errors_rad'])
        unchanged.update(engine='MuJoCo', trajectory=trajectory.metadata,
            policy_sha256=runner.bundle['policy_sha256'], diagnostic=True, trained=False,
            elapsed_seconds=float(runner.data.time),fall_height=fall,fall_tilt=False)
        unchanged.update(requested_steps=3,actual_steps=1 if fall else 3,completed=not fall,
                         termination_reason='fall' if fall else 'completed')
        assert result == unchanged
        assert trace['fall_height'][-1]==fall and not trace['fall_tilt'].any()
        np.testing.assert_allclose(trace['base_up_z'],1.)


def test_mujoco_suite_cli_preserves_engineering_report(diagnostic_bundle, tmp_path, monkeypatch):
    import hashlib
    import json
    import runpy
    import sys
    from pawweaver.trajectories import Trajectory
    bundle, spec_path = diagnostic_bundle
    suite = tmp_path / 'suite'
    suite.mkdir()
    case = suite / 'case.npz'
    Trajectory(np.array([0., .02]), np.array([[0., 0., .4], [0., 0., .4]]),
               np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
               {'family': 'reach', 'case_id': 'case'}).save(case)
    (suite / 'manifest.json').write_text(json.dumps({'task_kind': 'world_tcp_pose', 'reachability_screened': False,
        'cases': [{'file': case.name, 'sha256': hashlib.sha256(case.read_bytes()).hexdigest()}]}))
    def step(self, *args, **kwargs):
        self.data.time += .02
        return np.zeros(276), np.zeros(18)
    monkeypatch.setattr(MujocoRunner, 'step', step)
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['evaluate_mujoco.py', '--asset', str(ASSET), '--bundle', str(bundle),
        '--suite', str(suite), '--output', str(output), '--seed', '0', '--diagnostic', '--provisional-spec', str(spec_path)])
    runpy.run_path(str(ASSET.parents[2] / 'scripts/evaluate_mujoco.py'), run_name='__main__')
    report = json.loads((output / 'report.json').read_text())
    assert report['diagnostic'] and not report['trained']
    assert report['episodes'][0]['elapsed_seconds'] == .02
    assert report['evaluation']['provisional_spec'] == json.loads(spec_path.read_text())
    assert 'no hardware validity' in report['evaluation']['evidence_limit']
    assert report['evaluation']['termination']=={'minimum_base_height_m':.2,'minimum_base_up_z':.35}
    assert (output / 'case/trace.npz').exists()


@pytest.mark.parametrize('height,tilt,fallen', [(.18,False,False),(.14,False,True),(.3,True,True),(.14,True,True)])
def test_mujoco_bundle_height_profile_and_actual_terminal_snapshot(diagnostic_bundle,tmp_path,monkeypatch,height,tilt,fallen):
    import json
    from pawweaver.contracts import canonical_hash
    from pawweaver.trajectories import Trajectory
    bundle,spec_path=diagnostic_bundle
    manifest_path=bundle/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['training_config']['minimum_base_height_m']=.15
    manifest.pop('bundle_hash')
    manifest['bundle_hash']=canonical_hash(manifest)
    manifest_path.write_text(json.dumps(manifest))
    runner=MujocoRunner(ASSET,bundle,diagnostic=True,provisional_spec=spec_path)
    assert runner.termination=={'minimum_base_height_m':.15,'minimum_base_up_z':.35}
    def step(*args,**kwargs):
        runner.data.time+=.02
        runner.data.qpos[2]=height
        if tilt:
            runner.data.qpos[3:7]=[np.sqrt(.5),np.sqrt(.5),0,0]
        mujoco.mj_forward(runner.model,runner.data)
        return np.zeros(276),np.zeros(18)
    monkeypatch.setattr(runner,'step',step)
    state=runner.state()
    trajectory=Trajectory([0,.06],np.repeat(state.tcp_pos_w.numpy(),2,axis=0),
                          np.repeat(state.tcp_quat_w.numpy(),2,axis=0),{'family':'reach'})
    result=runner.evaluate(trajectory,tmp_path/'evaluation')
    assert result['fallen']==fallen
    assert result['fall_height']==(height<.15) and result['fall_tilt']==tilt
    with np.load(tmp_path/'evaluation/trace.npz',allow_pickle=False) as trace:
        assert len(trace['times'])==(1 if fallen else 3)
        assert trace['fall_height'][-1]==result['fall_height']
        assert trace['fall_tilt'][-1]==result['fall_tilt']
        assert trace['base'][-1,2]==pytest.approx(height)
        assert trace['base_up_z'][-1]==pytest.approx(0. if tilt else 1.,abs=1e-7)


def test_attach_marker_preserves_diagnostic_model(diagnostic_bundle, tmp_path):
    from pawweaver.visual_runtime import attach_marker
    bundle, spec_path = diagnostic_bundle
    runner = MujocoRunner(ASSET, bundle, diagnostic=True, provisional_spec=spec_path)
    before = runner.model
    before_qpos = runner.data.qpos.copy()
    before_tcp = runner.data.body('tcp').xpos.copy()
    attach_marker(runner, ASSET, {'marker_id': 7, 'marker_size_m': .16}, tmp_path)
    for field in ('dof_armature', 'dof_damping', 'dof_frictionloss', 'dof_invweight0',
                  'actuator_ctrlrange', 'qpos0'):
        np.testing.assert_array_equal(getattr(runner.model, field), getattr(before, field))
    np.testing.assert_array_equal(runner.data.qpos, before_qpos)
    np.testing.assert_array_equal(runner.data.body('tcp').xpos, before_tcp)
    marker = runner.model.body('vision_marker')
    assert marker.mocapid[0] >= 0
    geom_id = marker.geomadr[0]
    assert runner.model.geom_contype[geom_id] == runner.model.geom_conaffinity[geom_id] == 0
    assert runner.model.opt.timestep == before.opt.timestep
    assert runner.data.time == 0
