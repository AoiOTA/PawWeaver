"""Portable runtime TCP pose against independent canonical kinematics (CPU only)."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from pawweaver.assets.model import RobotTree
from pawweaver.contracts import JOINT_NAMES
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
