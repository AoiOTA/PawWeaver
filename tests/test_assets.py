"""Independent URDF FK against compiled MuJoCo; never rely on joint array order."""
from pathlib import Path
import numpy as np
import pytest
import mujoco
from pawweaver.assets.model import RobotTree
from pawweaver.assets.build import verify_asset
from pawweaver.contracts import JOINT_NAMES, FOOT_NAMES
from pawweaver.math import rpy_matrix

ASSET = Path(__file__).resolve().parents[1]/"assets/generated/diagnostic"

@pytest.mark.skipif(not (ASSET/"robot.xml").exists(), reason="Run fetch-assets and build-assets --diagnostic")
def test_independent_random_forward_kinematics():
    tree = RobotTree.load(ASSET/"robot.urdf")
    model = mujoco.MjModel.from_xml_path(str(ASSET/"robot.xml"))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(731)
    for _ in range(50):
        root = np.eye(4)
        root[:3,3] = rng.uniform(-2,2,3)
        root[:3,:3] = rpy_matrix(rng.uniform(-1,1,3))
        data.qpos[:3] = root[:3,3]
        mujoco.mju_mat2Quat(data.qpos[3:7],root[:3,:3].flatten())
        qs = {}
        for name in JOINT_NAMES:
            joint = model.joint(name)
            q = rng.uniform(*model.jnt_range[joint.id])
            data.qpos[joint.qposadr[0]] = q
            qs[name] = q
        mujoco.mj_forward(model,data)
        expected = tree.forward(qs,root)
        for name in (*FOOT_NAMES,"tcp"):
            actual = data.body(name)
            assert np.linalg.norm(actual.xpos-expected[name][:3,3]) < 1e-8
            assert np.max(np.abs(actual.xmat.reshape(3,3)-expected[name][:3,:3])) < 1e-8
    assert abs(model.body_mass.sum()-tree.mass) < 1e-8
    with pytest.raises(ValueError,match="Diagnostic"):
        verify_asset(ASSET)
