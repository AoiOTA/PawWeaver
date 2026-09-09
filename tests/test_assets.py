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

@pytest.mark.skipif(not (ASSET/"robot.xml").exists(), reason="Build diagnostic asset first")
def test_official_sdk_mdh_matches_urdf_and_mount_is_flush():
    import json
    from pawweaver.assets.model import numbers
    tree=RobotTree.load(ASSET/"robot.urdf")
    reference=json.loads((ASSET.parents[2]/"configs/actuator_references.json").read_text())
    mdh=reference["piper_h"]["mdh"]
    rng=np.random.default_rng(729)
    for _ in range(100):
        q=rng.uniform(-1,1,6)
        fk=tree.forward(dict(zip((f"arm_joint{i}" for i in range(1,7)),q)))
        expected=np.eye(4)
        for angle,(d,a,alpha,offset) in zip(q,mdh):
            rx=np.eye(4);rx[:3,:3]=rpy_matrix([alpha,0,0]);rx[0,3]=a
            rz=np.eye(4);rz[:3,:3]=rpy_matrix([0,0,angle+offset]);rz[2,3]=d
            expected=expected@rx@rz
        actual=np.linalg.inv(fk["arm_base_link"])@fk["arm_link6"]
        assert np.linalg.norm(expected[:3,3]-actual[:3,3])<1e-6
        assert np.max(np.abs(expected[:3,:3]-actual[:3,:3]))<1e-6
    mount_z=numbers(tree.joints["arm_mount"].find("origin").get("xyz"))[2]
    # Proposed adapter top is the arm mounting plane; geometry never floats above it.
    visuals=tree.links["base_link"].findall("visual")
    adapter=next(v for v in visuals if v.get("name")=="preview_adapter_plate")
    top=numbers(adapter.find("origin").get("xyz"))[2]+numbers(adapter.find("geometry/box").get("size"))[2]/2
    assert mount_z==pytest.approx(top,abs=1e-9)
