"""Frozen test inputs and paired sim-to-sim reports; no training dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .trajectories import Trajectory,synthetic

def create_suite(output:Path,seed=731,count=100):
    if count<100:
        raise ValueError("Acceptance suite requires at least 100 episodes per engine")
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    entries=[]
    for index in range(count):
        family=("reach","line","arc","eight","smooth")[index%5]
        case=synthetic(family,int(rng.integers(2**31)),duration=12. if family=="reach" else 60.,
                       amplitude=.6,speed=.3)
        if family=="reach":
            # A frozen radial grid explicitly covers 0.3–2 m, including distant goals.
            radius=.3+1.7*(index//5)/max((count-1)//5,1)
            yaw=rng.uniform(-np.pi,np.pi)
            case.positions[:]=[radius*np.cos(yaw),radius*np.sin(yaw),rng.uniform(.5,.9)]
        case.metadata.update(case_id=f"case_{index:04d}",split="test",suite_seed=seed,
                             reachability_screened=False)
        name=f"case_{index:04d}.npz"
        case.save(output/name)
        entries.append({"file":name,"sha256":hashlib.sha256((output/name).read_bytes()).hexdigest(),
                        "family":family,"case_id":case.metadata["case_id"]})
    manifest={"schema_version":1,"seed":seed,"count":count,"reachability_screened":False,
              "screening_note":"Candidate test set; hardware M0 and reachability review required before acceptance claims.",
              "cases":entries}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest

def load_suite(root:Path):
    manifest=json.loads((root/"manifest.json").read_text())
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
        "mean_rmse_m":float(np.mean([r["rmse_m"] for r in tracking])) if tracking else None}

def compare(physx:Path,mujoco:Path):
    a=json.loads(physx.read_text());b=json.loads(mujoco.read_text())
    for key in ("suite_sha256","policy_sha256","asset_hash","seed"):
        if a[key]!=b[key]:
            raise ValueError(f"Paired evaluations differ in {key}")
    if a["case_ids"]!=b["case_ids"]:
        raise ValueError("Paired evaluations contain different episodes")
    if a["engine"]!="PhysX" or b["engine"]!="MuJoCo":
        raise ValueError("Expected PhysX and MuJoCo reports")
    drop=a["summary"]["reach_success_rate"]-b["summary"]["reach_success_rate"]
    eligible=a["reachability_screened"] and b["reachability_screened"] and len(a["case_ids"])>=100
    return {"seed":a["seed"],"reach_success_drop_percentage_points":100*drop,
            "transfer_drop_within_target":drop<=.1,"eligible_for_acceptance":bool(eligible),
            "physx":a["summary"],"mujoco":b["summary"],
            "three_training_seeds_required":True}

def save_run(output,suite,asset_manifest,bundle,seed,engine,results):
    suite_manifest=json.loads((suite/"manifest.json").read_text())
    report={"engine":engine,"seed":seed,"asset_hash":asset_manifest["asset_hash"],
        "policy_sha256":bundle["policy_sha256"],"suite_sha256":hashlib.sha256((suite/"manifest.json").read_bytes()).hexdigest(),
        "reachability_screened":suite_manifest["reachability_screened"],
        "case_ids":[r["trajectory"]["case_id"] for r in results],"summary":summarize(results),"episodes":results}
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
