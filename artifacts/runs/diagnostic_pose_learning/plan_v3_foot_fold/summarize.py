"""CPU-only fold-cost comparison; run after the operator confirms all processes ended."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
COMPARISONS=(("mujoco","workspace4"),("mujoco","train10"),("mujoco","test8"),("physx","train10"))
NOT_RUN=("control_physx_workspace4","candidate_physx_workspace4","control_physx_test8","candidate_physx_test8")
STOP_REASON=("Candidate not adopted; no budget extension or coefficient sweep. Completed MuJoCo evaluations added lateral4.66s and step_01 13.54s falls; "
             "the three focus cases did not show a material fold reduction across both engines. PhysX body_high's local reduction also accompanied worse precision and a longer longest fold interval. "
             "This observed negative result was sufficient to stop before the four originally planned PhysX workspace4/test8 evaluations.")


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e1=load("fold_e1_compare",ROOT.parent/"plan_v3_reward_signal/compare.py")
e2=load("fold_e2_compare",ROOT.parent/"plan_v3_task_curriculum/summarize_final1000.py")
geometry=load("fold_support_geometry",ROOT.parent/"plan_v3_reward_signal/support_geometry.py")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (REPO/path).resolve()


def training_pair():
    decision=read(ROOT/"decision.json")
    runs={side:read(ROOT/f"{side}100/run.json") for side in ("control","candidate")}
    a,b=runs["control"],runs["candidate"]
    for key in ("seed","num_envs","asset_hash","diagnostic","provisional_spec","observation","source_sha256"):
        if a[key]!=b[key]:
            raise ValueError(f"Training arms differ in {key}")
    initializer=source_path(decision["initializer"])
    if source_path(a["initialize_from"])!=initializer or source_path(b["initialize_from"])!=initializer:
        raise ValueError("Arms did not record the same selected initializer")
    if a["seed"]!=0:
        raise ValueError("Expected the assigned seed0 comparison")
    configs={side:deepcopy(run["config"]) for side,run in runs.items()}
    weights={side:config["reward_weights"].pop("foot_above_thigh",0.) for side,config in configs.items()}
    if weights!={"control":0.,"candidate":-10.} or configs["control"]!=configs["candidate"]:
        raise ValueError("Effective config difference must be only foot_above_thigh 0/-10")
    for side,run in runs.items():
        if run["config"]!=read(ROOT/f"{side}_config.json"):
            raise ValueError(f"Actual {side} config differs from the selected config")
    summaries={side:e1.training(ROOT/f"{side}100") for side in runs}
    budget=decision["arm_budget"]
    for side,run in runs.items():
        if run["training_progress"]["requested_iterations"]!=budget["iterations"] or run["num_envs"]!=budget["num_envs"]:
            raise ValueError(f"Unexpected requested budget: {side}")
        if (run["config"]["rollout_steps"],run["config"]["learning_epochs"],run["config"]["mini_batches"])!=(budget["rollout_steps"],budget["epochs"],budget["minibatches"]):
            raise ValueError(f"Unexpected PPO budget: {side}")
    matched=all(summaries["control"][key]==summaries["candidate"][key] for key in ("recorded_iterations","transitions","optimizer_updates"))
    return runs,summaries,{"recorded_initializer":str(initializer),"initializer_sha256_now":digest(initializer),
        "initializer_evidence":"Both actual run metadata record this same preserved initializer path; hash is of its current file, not a separately captured per-run load hash.",
        "seed":a["seed"],"effective_config_difference":{"reward_weights.foot_above_thigh":weights},
        "same_training_source_hashes":True,"matched_completed_budget":matched,
        "scope":"Matched requested recipe/seed/start; actual completed budgets remain explicit. If completed budgets differ, do not attribute outcome differences to the coefficient alone."}


def support_cases(folder,report,bundle,tree,spheres):
    q0=np.array(bundle["actuators"]["default_pos"])
    rows=[]
    for episode in report["episodes"]:
        case=episode["trajectory"]["case_id"]
        if episode["actual_steps"]>=2:
            row=geometry.case_result(folder/case/"trace.npz",episode,report["engine"].lower(),tree,q0,spheres,
                                     bundle["actuators"]["physics_dt"],include_hip_height=True)
            if episode["trajectory"]["split"]=="train":
                # Follow the established train10 support organization. This
                # workspace-specific middle window is not train10's hold.
                row.pop("designed_middle_hold_planned_s",None)
                row["windows"].pop("designed_middle_hold",None)
                row["hip_height"]["windows"].pop("designed_middle_hold",None)
        else:
            row={"case_id":case,"original_episode_metrics":episode,"aligned_samples":0,
                 "status":"Final post-state has no next observation; geometry unavailable",
                 "windows":{"all_aligned":None,"post2":None},"hip_height":{"windows":{"all_aligned":None,"post2":None}}}
        rows.append(row)
    return rows


def build():
    runs,training,identity=training_pair()
    asset=REPO/"assets/generated/diagnostic"
    manifest=read(asset/"manifest.json")
    if digest(asset/"robot.urdf")!=manifest["files"]["robot.urdf"]:
        raise ValueError("URDF differs from manifest")
    tree=geometry.RobotTree.load(asset/"robot.urdf")
    if tree.root_name!="base_link":
        raise ValueError("Expected base_link-rooted canonical model")
    spheres=geometry.geometry(tree)
    suites={"workspace4":ROOT.parent/"wbc_workspace/targets/manifest.json",
            "train10":ROOT.parent/"plan_v3_task_curriculum/train/manifest.json",
            "test8":ROOT.parent/"references/test/manifest.json"}
    reports={};bundles={};self_checks={}
    # Validate every completed evaluation and actual policy before any FK work.
    for side in ("control","candidate"):
        bundle_path=ROOT/f"{side}100/bundle"
        bundle=read(bundle_path/"manifest.json")
        if bundle["policy_sha256"]!=digest(bundle_path/"policy.pt") or bundle["asset_hash"]!=manifest["asset_hash"] or bundle["asset_hash"]!=runs[side]["asset_hash"]:
            raise ValueError(f"Actual policy/asset mismatch: {side}")
        if bundle["actuators"]["joint_names"]!=list(geometry.JOINT_NAMES):
            raise ValueError("Bundle observation/FK joint order differs")
        bundles[side]=bundle
        self_checks[side]=geometry.default_check(tree,np.array(bundle["actuators"]["default_pos"]),spheres)
        for engine,suite in COMPARISONS:
            suite_path=suites[suite]
            name=f"{side}_{engine}_{suite}"
            report,_=e2.report_and_traces(ROOT/name)
            expected=[entry["case_id"] for entry in read(suite_path)["cases"]]
            if report["case_ids"]!=expected or report["suite_sha256"]!=digest(suite_path):
                raise ValueError(f"Changed or missing frozen cases: {name}")
            if report["policy_sha256"]!=bundle["policy_sha256"] or report["asset_hash"]!=bundle["asset_hash"] or report["seed"]!=runs[side]["seed"]:
                raise ValueError(f"Evaluation policy/asset/seed mismatch: {name}")
            if report["engine"].lower()!=engine or report["task_kind"]!="world_tcp_pose":
                raise ValueError(f"Unexpected evaluation engine/task: {name}")
            reports[name]=report
    comparisons={}
    for engine,suite in COMPARISONS:
        pair=e2.common_comparison(ROOT/f"control_{engine}_{suite}",ROOT/f"candidate_{engine}_{suite}")
        pair.pop("equal_case_common_window_means",None)
        comparisons[f"{engine}_{suite}"]=pair
    support={}
    for side in bundles:
        for engine,suite in COMPARISONS:
            if suite in ("workspace4","train10"):
                name=f"{side}_{engine}_{suite}"
                support[name]=support_cases(ROOT/name,reports[name],bundles[side],tree,spheres)
    return {"training":training,"comparison_identity":identity,"comparisons":comparisons,"support_geometry":support,
            "default_fk_self_checks":self_checks,"focus_training_cases":["body_high","step_00","step_02"],
            "evaluations":{name:{"report_path":str(ROOT/name/"report.json"),"policy_sha256":report["policy_sha256"],"actual_process_exit_code":0,
                "completed_cases":sum(e["completed"] for e in report["episodes"]),"failed_case_ids":[e["trajectory"]["case_id"] for e in report["episodes"] if not e["completed"] or e["fallen"]],
                "episodes":report["episodes"]} for name,report in reports.items()},
            "training_actual_process_exit_codes":{"control":0,"candidate":0},
            "exit_code_source":"Master operator confirmed both training and all eight actually executed evaluation process exits0 before aggregation.",
            "not_run":[{"evaluation":name,"status":"not_run","reason":STOP_REASON} for name in NOT_RUN],
            "stop_decision":{"candidate_adopted":False,"extend_budget":False,"coefficient_sweep":False,"reason":STOP_REASON,"source":"Master decision based on the completed comparisons"},
            "scope":"Fixed actual four comparison pairs. All requested cases and failures within the eight executed evaluations retained; four unstarted evaluations are listed as not_run without invented episodes or outcomes. Post2 and hold deltas use matched ticks. No task-error group mean replaces failed-case reporting. Focus labels do not filter the other tasks.",
            "support_scope":"Reuse existing URDF FK/next-observation alignment and MuJoCo hinge integration correction. Train10 geometry reports all_aligned/post2 only, not a whole-hold claim. Foot above thigh is a geometric zero boundary, not acceptance; 3cm/1N/5N descriptors and MuJoCo floor-pair counts remain distinct. No new gait inference.",
            "orientation_acceptance":None,"evidence_limit":"Provisional-model bounded coefficient comparison; no hardware validity, formal milestone, or full whole-body capability claim."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result=build()
    text=json.dumps(result,indent=2,allow_nan=False)+"\n"
    with args.output.open("x") as stream:
        stream.write(text)
    print(json.dumps({"output":str(args.output),"matched_completed_budget":result["comparison_identity"]["matched_completed_budget"],
        "training":{side:{key:values[key] for key in ("recorded_iterations","transitions","optimizer_updates")} for side,values in result["training"].items()},
        "not_run":[item["evaluation"] for item in result["not_run"]],
        "evaluations":{name:{"completed_cases":value["completed_cases"],"failed_case_ids":value["failed_case_ids"]} for name,value in result["evaluations"].items()}},indent=2))
