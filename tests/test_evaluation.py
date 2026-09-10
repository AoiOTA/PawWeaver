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


@pytest.mark.parametrize('left,right,accepted', [
    (None,{'minimum_base_height_m':.2,'minimum_base_up_z':.35},True),
    ({'minimum_base_height_m':.15,'minimum_base_up_z':.35},
     {'minimum_base_height_m':.15,'minimum_base_up_z':.35},True),
    (None,{'minimum_base_height_m':.15,'minimum_base_up_z':.35},False),
    ({'minimum_base_height_m':.15,'minimum_base_up_z':.35},
     {'minimum_base_height_m':.15,'minimum_base_up_z':.2},False),
])
def test_crossengine_pairing_checks_only_resolved_termination_semantics(tmp_path,left,right,accepted):
    common=dict(suite_sha256='suite',asset_hash='asset',seed=0,case_ids=['a'],policy_sha256='policy',
                reachability_screened=False,summary={'reach_success_rate':1.})
    paths=[tmp_path/'physx.json',tmp_path/'mujoco.json']
    for path,engine,rule,background in zip(paths,['PhysX','MuJoCo'],[left,right],['one','different']):
        evaluation={'other_metadata':background}
        if rule is not None:
            evaluation['termination']=rule
        path.write_text(json.dumps(dict(common,engine=engine,evaluation=evaluation)))
    if accepted:
        assert compare(*paths)['reach_success_drop_percentage_points']==0
    else:
        with pytest.raises(ValueError,match='termination rule'):
            compare(*paths)

def test_pose_suite_is_explicit_and_frozen(tmp_path):
    manifest=create_suite(tmp_path)
    trajectories=load_suite(tmp_path)
    assert manifest["task_kind"]=="world_tcp_pose"
    assert not manifest["reachability_screened"]
    rotating=trajectories[1]
    np.testing.assert_allclose(np.linalg.norm(rotating.orientations_wxyz,axis=-1),1.)
    assert np.ptp(rotating.orientations_wxyz[:,3])>.2
    assert rotating.metadata["orientation_profile"]=="bounded_world_yaw_0.25_rad"
    assert rotating.metadata["orientation_generation"]=="explicit_sinusoidal_world_yaw"
    assert "orientation_wxyz" not in rotating.metadata
    with pytest.raises(FileExistsError,match="Frozen"):
        create_suite(tmp_path)


def test_position_only_suite_is_not_upgraded(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps({"schema_version":1,"cases":[]}))
    with pytest.raises(ValueError,match="position-only"):
        load_suite(tmp_path)


def test_orientation_does_not_invent_acceptance_thresholds():
    from pawweaver.evaluation import summarize
    rows=[dict(trajectory={"family":"line"},rmse_m=.08,p95_m=.15,fallen=False,
               orientation_rmse_rad=2.,orientation_p95_rad=3.)]
    summary=summarize(rows)
    assert summary["tracking_pass_rate"]==1.
    assert summary["mean_orientation_rmse_rad"]==2.
    assert summary["mean_orientation_p95_rad"]==3.
    assert summary["success_rates_scope"]=="position_criteria_only"
    assert summary["pose_acceptance_passed"] is None
    rows[0]["rmse_m"]+=.0001
    assert summarize(rows)["tracking_pass_rate"]==0.


def test_pair_report_keeps_position_transfer_separate(tmp_path):
    common=dict(suite_sha256="suite",asset_hash="asset",seed=0,case_ids=list(range(100)),
                policy_sha256="policy",reachability_screened=True,task_kind="world_tcp_pose")
    a=tmp_path/"a.json";b=tmp_path/"b.json"
    a.write_text(json.dumps(dict(common,engine="PhysX",summary={"reach_success_rate":.95})))
    b.write_text(json.dumps(dict(common,engine="MuJoCo",summary={"reach_success_rate":.90})))
    result=compare(a,b)
    assert result["transfer_drop_within_target"]
    assert result["position_criteria_eligible"]
    assert not result["eligible_for_acceptance"]
    assert result["pose_acceptance_passed"] is None
    common.pop("task_kind")
    a.write_text(json.dumps(dict(common,engine="PhysX",summary={"reach_success_rate":.95})))
    with pytest.raises(ValueError,match="task_kind"):
        compare(a,b)
    b.write_text(json.dumps(dict(common,engine="MuJoCo",summary={"reach_success_rate":.90})))
    assert compare(a,b)["task_kind"]=="position_only"


def test_pose_metrics_flow_into_report_without_full_pose_pass(tmp_path):
    from pawweaver.task import episode_metrics
    from pawweaver.evaluation import save_run
    times=np.array([0.,1.,2.,3.])
    metrics=episode_metrics(times,np.full(4,.04),np.zeros((4,3)),np.ones((4,18)),
                            np.ones((4,18)),False,orientation_errors=np.array([0.,0.,.3,.4]))
    metrics["trajectory"]={"case_id":"one","family":"line"}
    suite=tmp_path/"suite";suite.mkdir()
    (suite/"manifest.json").write_text(json.dumps({"task_kind":"world_tcp_pose","reachability_screened":False}))
    report=save_run(tmp_path/"report",suite,{"asset_hash":"asset"},{"policy_sha256":"policy"},0,"MuJoCo",[metrics])
    assert report["task_kind"]=="world_tcp_pose"
    assert report["summary"]["mean_orientation_rmse_rad"]==pytest.approx(np.sqrt((.3**2+.4**2)/2))
    assert report["summary"]["mean_orientation_p95_rad"]==pytest.approx(.395)
    assert report["summary"]["tracking_pass_rate"]==1.
    assert report["summary"]["pose_acceptance_passed"] is None


def test_pairing_ignores_unconsumed_bundle_metadata_hash(tmp_path):
    common=dict(suite_sha256="suite",asset_hash="asset",seed=0,case_ids=["a"],policy_sha256="same_actor",
                reachability_screened=False,summary={"reach_success_rate":1.})
    a=tmp_path/"a.json";b=tmp_path/"b.json"
    a.write_text(json.dumps(dict(common,engine="PhysX",bundle_hash="config_a")))
    b.write_text(json.dumps(dict(common,engine="MuJoCo",bundle_hash="config_b")))
    assert compare(a,b)["reach_success_drop_percentage_points"]==0.


@pytest.mark.parametrize("reach_rates,reach_count,expected_drop,within",[
    ((None,None),0,None,None),
    ((.9,.7),2,20.,False),
])
def test_pairing_dynamic_only_and_mixed_suites(tmp_path,reach_rates,reach_count,expected_drop,within):
    common=dict(suite_sha256="suite",asset_hash="asset",seed=0,case_ids=list(range(8)),
                policy_sha256="policy",reachability_screened=False,task_kind="world_tcp_pose")
    summaries=[dict(reach_episodes=reach_count,tracking_episodes=8-reach_count,
                    reach_success_rate=rate,mean_rmse_m=.04+index*.01,
                    mean_orientation_rmse_rad=.2+index*.1)
               for index,rate in enumerate(reach_rates)]
    a=tmp_path/"a.json";b=tmp_path/"b.json"
    a.write_text(json.dumps(dict(common,engine="PhysX",summary=summaries[0])))
    b.write_text(json.dumps(dict(common,engine="MuJoCo",summary=summaries[1])))
    result=compare(a,b)
    if expected_drop is None:
        assert result["reach_success_drop_percentage_points"] is None
    else:
        assert result["reach_success_drop_percentage_points"]==pytest.approx(expected_drop)
    assert result["transfer_drop_within_target"] is within
    assert result["physx"]==summaries[0]
    assert result["mujoco"]==summaries[1]
    del summaries[1]["reach_success_rate"]
    b.write_text(json.dumps(dict(common,engine="MuJoCo",summary=summaries[1])))
    with pytest.raises(KeyError,match="reach_success_rate"):
        compare(a,b)
