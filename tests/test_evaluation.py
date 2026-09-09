import json
import numpy as np
import pytest
from pawweaver.evaluation import create_suite,load_suite,compare

def test_frozen_suite_covers_far_goals_and_rejects_tampering(tmp_path):
    create_suite(tmp_path)
    trajectories=load_suite(tmp_path)
    assert len(trajectories)==100
    reaches=[t for t in trajectories if t.metadata["family"]=="reach"]
    radii=np.array([np.linalg.norm(t.positions[0,:2]) for t in reaches])
    assert radii.min()==pytest.approx(.3)
    assert radii.max()==pytest.approx(2.)
    for trajectory in trajectories:
        speed=np.linalg.norm(np.diff(trajectory.positions,axis=0),axis=1)/np.diff(trajectory.timestamps)
        assert speed.max()<=.3001
    (tmp_path/"case_0000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError,match="Changed"):
        load_suite(tmp_path)

def test_pairing_rejects_different_policy(tmp_path):
    common=dict(suite_sha256="suite",asset_hash="asset",seed=0,case_ids=["a"])
    a=tmp_path/"a.json";b=tmp_path/"b.json"
    a.write_text(json.dumps(dict(common,engine="PhysX",policy_sha256="policy_a")))
    b.write_text(json.dumps(dict(common,engine="MuJoCo",policy_sha256="policy_b")))
    with pytest.raises(ValueError,match="policy_sha256"):
        compare(a,b)
