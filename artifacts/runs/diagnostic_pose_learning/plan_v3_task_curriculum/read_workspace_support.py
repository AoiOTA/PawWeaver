"""Read one completed workspace4 evaluation using the existing E1 geometry method."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/"plan_v3_reward_signal/support_geometry.py"
spec=importlib.util.spec_from_file_location("e1_support_geometry",SOURCE)
geometry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
EXPECTED={"low":1000,"high":1000,"lateral":1000,"far":3000}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(evaluation,bundle_path):
    report=geometry.read(evaluation/"report.json")
    bundle=geometry.read(bundle_path/"manifest.json")
    asset=geometry.REPO/"assets/generated/diagnostic"
    manifest=geometry.read(asset/"manifest.json")
    suite=ROOT.parent/"wbc_workspace/targets/manifest.json"
    if report["suite_sha256"]!=digest(suite) or report["case_ids"]!=list(EXPECTED):
        raise ValueError("Expected the unchanged low/high/lateral20s, far60s workspace4 suite")
    if report["task_kind"]!="world_tcp_pose":
        raise ValueError("Expected world-frame TCP pose evaluation")
    if report["asset_hash"]!=bundle["asset_hash"] or report["asset_hash"]!=manifest["asset_hash"]:
        raise ValueError("Report/bundle/local asset identity differs")
    if report["policy_sha256"]!=bundle["policy_sha256"] or report["policy_sha256"]!=digest(bundle_path/"policy.pt"):
        raise ValueError("Report/bundle/actual policy identity differs")
    if digest(asset/"robot.urdf")!=manifest["files"]["robot.urdf"]:
        raise ValueError("Actual URDF differs from asset manifest")
    engine={"PhysX":"physx","MuJoCo":"mujoco"}[report["engine"]]
    actuators=bundle["actuators"]
    if actuators["joint_names"]!=list(geometry.JOINT_NAMES):
        raise ValueError("Bundle joint order differs from observation convention")
    tree=geometry.RobotTree.load(asset/"robot.urdf")
    if tree.root_name!="base_link":
        raise ValueError("Expected base_link-rooted canonical URDF")
    spheres=geometry.geometry(tree)
    q0=np.array(actuators["default_pos"])
    check=geometry.default_check(tree,q0,spheres)
    episodes=report["episodes"]
    if [episode["trajectory"]["case_id"] for episode in episodes]!=list(EXPECTED):
        raise ValueError("Missing, duplicate or reordered workspace episodes")
    rows=[]
    for episode in episodes:
        case=episode["trajectory"]["case_id"]
        actual=episode["actual_steps"]
        if episode["requested_steps"]!=EXPECTED[case] or not 0<actual<=EXPECTED[case]:
            raise ValueError(f"Unexpected requested/actual duration: {case}")
        if episode["completed"]!=(actual==EXPECTED[case]) or not np.isclose(episode["elapsed_seconds"],actual*.02,rtol=0,atol=1e-8):
            raise ValueError(f"Step count and recorded completion/duration disagree: {case}")
        if episode["policy_sha256"]!=report["policy_sha256"] or episode["engine"]!=report["engine"]:
            raise ValueError(f"Episode/report identity differs: {case}")
        trace=evaluation/case/"trace.npz"
        if actual>=2:
            row=geometry.case_result(trace,episode,engine,tree,q0,spheres,actuators["physics_dt"])
        else:
            # A first-step fall has no next observation to align; retain it.
            with np.load(trace,allow_pickle=False) as data:
                if data["times"].shape!=(1,) or not np.isclose(data["times"][0],.02,rtol=0,atol=1e-8):
                    raise ValueError(f"Single-step trace/report mismatch: {case}")
            row={"case_id":case,"trace":str(trace),"original_episode_metrics":episode,
                 "aligned_samples":0,"excluded_final_poststep_samples":1,"alignment_checks":None,
                 "status":"No next observation; support reconstruction unavailable",
                 "windows":dict(all_aligned=None,post2=None,designed_middle_hold=None)}
        rows.append(row)
    return {"evaluation":str(evaluation),"bundle":str(bundle_path),"engine":report["engine"],
            "policy_sha256":report["policy_sha256"],"asset_hash":report["asset_hash"],
            "suite_sha256":report["suite_sha256"],"default_fk_self_check":check,"cases":rows,
            "method_source":str(SOURCE),"method_source_sha256":digest(SOURCE),
            "scope":"Saved 50Hz states only. Each row preserves its original task errors, duration and failure; absent aligned post2/hold windows remain null. No mixed-duration task mean or gait/acceptance verdict.",
            "height_method":"Canonical foot collision sphere-bottom world Z from projected gravity; next-row pre-step observation aligns with prior post-step base/contact, final row omitted. MuJoCo hinge q backed up one physics_dt*qvel to match saved implicitfast kinematics. World XY/yaw not reconstructed.",
            "contact_scope":"3cm height, foot net force >1N and robot nonfoot net force >5N are descriptive engineering thresholds, not acceptance. Net force is not vertical load or a contact pair; separately recorded MuJoCo nonfoot ground-pair counts remain distinct. 50Hz snapshots do not cover all physics substeps.",
            "orientation_acceptance":None,"evidence_limit":"Provisional model geometry readout, not hardware validity or formal whole-body task completion."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation",type=Path,required=True)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result=build(args.evaluation.resolve(),args.bundle.resolve())
    text=json.dumps(result,indent=2,allow_nan=False)+"\n"
    with args.output.open("x") as stream:
        stream.write(text)
    print(json.dumps({"output":str(args.output),"engine":result["engine"],
        "cases":len(result["cases"]),"aligned_samples":sum(row["aligned_samples"] for row in result["cases"])}))
