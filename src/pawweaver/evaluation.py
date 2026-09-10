"""Frozen test inputs and paired sim-to-sim reports; no training dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .trajectories import Trajectory,synthetic

def create_suite(output:Path,seed=731,count=100):
    if count<100:
        raise ValueError("Candidate suite requires at least 100 episodes per engine")
    if (output/"manifest.json").exists():
        raise FileExistsError("Frozen suite already exists; use a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    entries=[]
    for index in range(count):
        family=("reach","line","arc","eight","smooth")[index%5]
        case=synthetic(family,int(rng.integers(2**31)),duration=12. if family=="reach" else 60.,
                       amplitude=.6,speed=.3,orientation_wxyz=[1.,0.,0.,0.])
        if family=="reach":
            # A frozen radial grid explicitly covers 0.3–2 m, including distant goals.
            radius=.3+1.7*(index//5)/max((count-1)//5,1)
            yaw=rng.uniform(-np.pi,np.pi)
            case.positions[:]=[radius*np.cos(yaw),radius*np.sin(yaw),rng.uniform(.5,.9)]
        # Explicit world yaw targets: frozen bounded rotation, not inferred identity.
        phase=2*np.pi*case.timestamps/case.timestamps[-1]
        yaw=.25*np.sin(phase) if family!="reach" else np.full_like(phase,rng.uniform(-.25,.25))
        orientations=np.stack((np.cos(yaw/2),np.zeros_like(yaw),np.zeros_like(yaw),np.sin(yaw/2)),axis=-1)
        metadata=dict(case.metadata,task_kind="world_tcp_pose",
                      orientation_profile="bounded_world_yaw_0.25_rad",
                      orientation_generation="explicit_constant_world_yaw" if family=="reach" else "explicit_sinusoidal_world_yaw")
        metadata.pop("orientation_wxyz",None)
        case=Trajectory(case.timestamps,case.positions,orientations,metadata)
        case.metadata.update(case_id=f"case_{index:04d}",split="test",suite_seed=seed,
                             reachability_screened=False)
        name=f"case_{index:04d}.npz"
        case.save(output/name)
        entries.append({"file":name,"sha256":hashlib.sha256((output/name).read_bytes()).hexdigest(),
                        "family":family,"case_id":case.metadata["case_id"]})
    manifest={"schema_version":2,"task_kind":"world_tcp_pose","quaternion_convention":"wxyz","seed":seed,"count":count,"reachability_screened":False,
              "screening_note":"Unscreened candidate pose cases; repeated synthetic shape families are not 100 independent shapes. Hardware M0 and reachability review remain required.",
              "cases":entries}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest

def load_suite(root:Path):
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest.get("task_kind")!="world_tcp_pose":
        raise ValueError("Legacy position-only suite cannot evaluate the world TCP pose task; create an explicit pose suite")
    result=[]
    for entry in manifest["cases"]:
        path=root/entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError(f"Changed frozen evaluation input: {path}")
        result.append(Trajectory.load(path))
    return result

def summarize(results):
    reach=[r for r in results if r["trajectory"]["family"]=="reach"]
    tracking=[r for r in results if r["trajectory"]["family"]!="reach"]
    return {"episodes":len(results),"reach_episodes":len(reach),"tracking_episodes":len(tracking),
        "reach_success_rate":float(np.mean([r["reached"] for r in reach])) if reach else None,
        "tracking_pass_rate":float(np.mean([r["rmse_m"]<=.08 and r["p95_m"]<=.15 and not r["fallen"] for r in tracking])) if tracking else None,
        "no_fall_rate":float(np.mean([not r["fallen"] for r in results])),
        "mean_rmse_m":float(np.mean([r["rmse_m"] for r in tracking])) if tracking else None,
        "success_rates_scope":"position_criteria_only",
        "mean_orientation_rmse_rad":float(np.mean([r["orientation_rmse_rad"] for r in results])) if results and all(r.get("orientation_rmse_rad") is not None for r in results) else None,
        "mean_orientation_p95_rad":float(np.mean([r["orientation_p95_rad"] for r in results])) if results and all(r.get("orientation_p95_rad") is not None for r in results) else None,
        "pose_acceptance_passed":None,
        "pose_acceptance_note":"Orientation acceptance thresholds have not been specified; existing success/pass rates describe position criteria only."}

def compare(physx:Path,mujoco:Path):
    a=json.loads(physx.read_text());b=json.loads(mujoco.read_text())
    for key in ("suite_sha256","policy_sha256","asset_hash","seed"):
        if a[key]!=b[key]:
            raise ValueError(f"Paired evaluations differ in {key}")
    if a["case_ids"]!=b["case_ids"]:
        raise ValueError("Paired evaluations contain different episodes")
    if a["engine"]!="PhysX" or b["engine"]!="MuJoCo":
        raise ValueError("Expected PhysX and MuJoCo reports")
    legacy_termination={"minimum_base_height_m":.2,"minimum_base_up_z":.35}
    if a.get("evaluation",{}).get("termination",legacy_termination)!=b.get("evaluation",{}).get("termination",legacy_termination):
        raise ValueError("Paired evaluations differ in termination rule")
    task_a=a.get("task_kind","position_only")
    if task_a!=b.get("task_kind","position_only"):
        raise ValueError("Paired evaluations differ in task_kind")
    reach_a=a["summary"]["reach_success_rate"];reach_b=b["summary"]["reach_success_rate"]
    drop=None if reach_a is None or reach_b is None else reach_a-reach_b
    eligible=a["reachability_screened"] and b["reachability_screened"] and len(a["case_ids"])>=100 and not a.get("diagnostic",False) and not b.get("diagnostic",False)
    return {"seed":a["seed"],"reach_success_drop_percentage_points":None if drop is None else 100*drop,
            "transfer_drop_within_target":None if drop is None else drop<=.1,"position_criteria_eligible":bool(eligible),
            "task_kind":task_a,"eligible_for_acceptance":False,"pose_acceptance_passed":None,
            "acceptance_note":"Position transfer target is unchanged; full pose acceptance is unset pending orientation thresholds.",
            "physx":a["summary"],"mujoco":b["summary"],
            "three_training_seeds_required":True}

def save_run(output,suite,asset_manifest,bundle,seed,engine,results,*,evaluation=None):
    suite_manifest=json.loads((suite/"manifest.json").read_text())
    report={"task_kind":suite_manifest.get("task_kind","position_only"),"engine":engine,"seed":seed,"asset_hash":asset_manifest["asset_hash"],
        "policy_sha256":bundle["policy_sha256"],"suite_sha256":hashlib.sha256((suite/"manifest.json").read_bytes()).hexdigest(),
        "reachability_screened":suite_manifest["reachability_screened"],
        "case_ids":[r["trajectory"]["case_id"] for r in results],"summary":summarize(results),"episodes":results}
    report.update(diagnostic=bool(bundle.get("training_metadata",{}).get("diagnostic",False)),trained=bool(bundle.get("trained",False)))
    if evaluation is not None:
        report["evaluation"]=evaluation
    output.mkdir(parents=True,exist_ok=True)
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    make=sub.add_parser("create");make.add_argument("output",type=Path)
    make.add_argument("--seed",type=int,default=731);make.add_argument("--count",type=int,default=100)
    pair=sub.add_parser("compare");pair.add_argument("physx",type=Path);pair.add_argument("mujoco",type=Path)
    pair.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="create":
        print(json.dumps(create_suite(args.output,args.seed,args.count),indent=2))
    else:
        result=compare(args.physx,args.mujoco)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
