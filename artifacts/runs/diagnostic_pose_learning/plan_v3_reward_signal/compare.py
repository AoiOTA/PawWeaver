"""CPU-only E1 analysis; print JSON or write --output after all inputs validate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parent
METRICS=("rmse_m","p95_m","orientation_rmse_rad","orientation_p95_rad")
FEET=("FR_foot","FL_foot","RR_foot","RL_foot")


def read_json(path):
    return json.loads(path.read_text())


def training(path):
    import torch
    run=read_json(path/"run.json")
    rows=[json.loads(line) for line in (path/"metrics.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError(f"No completed updates: {path}")
    iterations=[row["iteration"] for row in rows]
    if iterations!=list(range(iterations[0],iterations[0]+len(rows))):
        raise ValueError(f"Nonconsecutive or duplicate updates: {path}")
    nonfinite=[{"iteration":row["iteration"],"key":key,"value":str(value)}
               for row in rows for key,value in row.items()
               if isinstance(value,(int,float)) and not math.isfinite(value)]
    latest=max(path.glob("checkpoint_*.pt"),key=lambda p:int(p.stem.split("_")[-1]))
    checkpoint=torch.load(latest,map_location="cpu",weights_only=False)
    progress=run["training_progress"]
    if progress["completed_iterations"]!=len(rows) or progress["last_completed_iteration"]!=iterations[-1]:
        raise ValueError(f"run.json and actual metrics completion disagree: {path}")
    if progress["status"] not in ("completed","stopped_early"):
        raise ValueError(f"Training still running or not normally closed: {path}")
    if checkpoint["iteration"]!=iterations[-1]:
        raise ValueError(f"Latest checkpoint does not contain last complete update: {path}")
    transitions=len(rows)*run["num_envs"]*run["config"]["rollout_steps"]
    if progress["completed_transitions"]!=transitions:
        raise ValueError(f"Transition count mismatch: {path}")
    keys=[key for key in rows[0] if key.startswith(("reward_","umi_")) or
          key in ("learning_rate","leg_action_std_mean","arm_action_std_mean",
                  "tracking_error_mean_m","orientation_error_mean_rad","value","surrogate","kl")]
    def window(values):
        return {"iterations":[values[0]["iteration"],values[-1]["iteration"]],"count":len(values),
                "means":{key:float(np.mean([row[key] for row in values]))
                         if all(math.isfinite(row[key]) for row in values) else None for key in keys}}
    saturated=sum(row["torque_saturated"] for row in rows)
    samples=sum(row["torque_samples"] for row in rows)
    return {"path":str(path),"training_progress":progress,"recorded_iterations":len(rows),
            "transitions":transitions,"optimizer_updates":sum(row["optimizer_steps"] for row in rows),
            "all_recorded_scalars_finite":not nonfinite,"nonfinite_entries":nonfinite,
            "all_diagnostic_finite_checks_passed":all(row["finite_checks_passed"] is True for row in rows),
            "falls":sum(row["falls"] for row in rows),"resets":sum(row["resets"] for row in rows),
            "torque_saturated":saturated,"torque_samples":samples,
            "saturation_fraction":saturated/samples if samples else None,
            "loop_seconds":sum(run["num_envs"]*run["config"]["rollout_steps"]/row["steps_per_second"] for row in rows),
            "latest_checkpoint":str(latest),"latest_checkpoint_iteration":checkpoint["iteration"],
            "first25":window(rows[:25]),"last25":window(rows[-25:]),
            "note":"Finite completed updates and training sample statistics, not independent task performance. Reward terms exclude dt and terminal penalty; LR/std are recorded after each update."}


def trace(path,episode):
    with np.load(path,allow_pickle=False) as saved:
        result={key:saved[key] for key in ("times","errors","orientation_errors_rad","base","base_up_z",
                                          "contacts","contact_body_names")}
    for key,value in result.items():
        if key!="contact_body_names" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {key}: {path}")
    ticks=np.rint(result["times"]/.02).astype(np.int64)
    if not len(ticks) or not np.allclose(result["times"],ticks*.02,atol=1e-8,rtol=0) or not np.array_equal(ticks,np.arange(1,len(ticks)+1)):
        raise ValueError(f"Trace is not continuous 50 Hz ticks: {path}")
    if episode["actual_steps"]!=len(ticks) or not 0<len(ticks)<=episode["requested_steps"]:
        raise ValueError(f"Trace/report step count differs: {path}")
    if episode["completed"]!=(episode["actual_steps"]==episode["requested_steps"]):
        raise ValueError(f"Completion flag disagrees with steps: {path}")
    if not np.isclose(episode["elapsed_seconds"],ticks[-1]*.02,atol=1e-8,rtol=0):
        raise ValueError(f"Trace/report elapsed time differs: {path}")
    for key,value in result.items():
        if key!="contact_body_names" and len(value)!=len(ticks):
            raise ValueError(f"Trace field length mismatch: {path}/{key}")
    result["ticks"]=ticks
    return result


def errors(data,end):
    mask=(data["ticks"]>=100)&(data["ticks"]<=end)
    if not mask.any():
        return None
    position=data["errors"][mask].astype(np.float64)
    orientation=data["orientation_errors_rad"][mask].astype(np.float64)
    return {"samples":int(mask.sum()),"rmse_m":float(np.sqrt(np.mean(position**2))),
            "p95_m":float(np.quantile(position,.95)),
            "orientation_rmse_rad":float(np.sqrt(np.mean(orientation**2))),
            "orientation_p95_rad":float(np.quantile(orientation,.95))}


def motion(data):
    names=data["contact_body_names"].tolist()
    if len(names)!=len(set(names)) or any(name not in names for name in FEET):
        raise ValueError("Ambiguous or absent actual foot body names")
    force=data["contacts"]
    if force.shape!=(len(data["ticks"]),len(names)) or np.any(force<0):
        raise ValueError("Expected nonnegative net force magnitudes for each recorded body")
    foot_ids=[names.index(name) for name in FEET]
    nonfeet=[i for i,name in enumerate(names) if name not in (*FEET,"world")]
    count=(force[:,foot_ids]>1.).sum(axis=1)
    base=data["base"]
    if base.shape!=(len(count),3):
        raise ValueError("Expected 3D base positions")
    tilt=np.arccos(np.clip(data["base_up_z"],-1.,1.))
    return {"samples":len(count),"sampling_hz":50,"window":"entire executed trace, including failure",
            "foot_net_force_gt1N_count_samples":{str(n):int((count==n).sum()) for n in range(5)},
            "per_foot_net_force_gt1N_fraction":{name:float(np.mean(force[:,i]>1.)) for name,i in zip(FEET,foot_ids)},
            "robot_nonfoot_net_force_gt5N_any_fraction":float(np.mean((force[:,nonfeet]>5.).any(axis=1))),
            "excluded_nonrobot_contact_bodies":[name for name in names if name=="world"],
            "base_height_min_m":float(base[:,2].min()),"base_height_max_m":float(base[:,2].max()),
            "base_height_final_minus_first_m":float(base[-1,2]-base[0,2]),
            "base_tilt_max_rad":float(tilt.max()),"base_tilt_p95_rad":float(np.quantile(tilt,.95)),
            "base_xy_max_displacement_from_first_sample_m":float(np.linalg.norm(base[:,:2]-base[0,:2],axis=1).max()),
            "note":"Net force magnitude snapshots, not vertical loads or contact pairs; no inference of gait success or all physics substeps. Nonfoot net force can include self-collision."}


def position_pass(episode):
    complete=episode["requested_steps"]>0 and episode["actual_steps"]==episode["requested_steps"] and episode["completed"]
    if not complete or episode["fallen"]:
        return False
    if episode["trajectory"]["family"]=="reach":
        return bool(episode["reached"])
    return episode["rmse_m"] is not None and episode["p95_m"] is not None and episode["rmse_m"]<=.08 and episode["p95_m"]<=.15


def paired(root,engine,suite):
    directories={side:root/f"{side}_{engine}_{suite}" for side in ("control","candidate")}
    reports={side:read_json(path/"report.json") for side,path in directories.items()}
    a,b=reports["control"],reports["candidate"]
    for key in ("asset_hash","suite_sha256","task_kind","seed","diagnostic","case_ids","engine"):
        if a[key]!=b[key]:
            raise ValueError(f"Paired reports differ: {engine}/{suite}/{key}")
    if a["task_kind"]!="world_tcp_pose" or a["engine"]!={"physx":"PhysX","mujoco":"MuJoCo"}[engine]:
        raise ValueError("Expected world-fixed EE pose report and named engine")
    if a["evaluation"]["termination"]!=b["evaluation"]["termination"]:
        raise ValueError("Paired termination rules differ")
    cases={side:{e["trajectory"]["case_id"]:e for e in report["episodes"]} for side,report in reports.items()}
    expected=4 if suite=="workspace4" else 8
    for side,report in reports.items():
        if len(report["episodes"])!=expected or len(cases[side])!=expected or list(cases[side])!=report["case_ids"]:
            raise ValueError(f"Missing or duplicate frozen cases: {side}/{suite}")
    rows=[]
    for case in a["case_ids"]:
        episodes={side:cases[side][case] for side in cases}
        data={side:trace(directories[side]/case/"trace.npz",episodes[side]) for side in cases}
        if episodes["control"]["requested_steps"]!=episodes["candidate"]["requested_steps"]:
            raise ValueError(f"Requested durations differ: {case}")
        end=min(value["ticks"][-1] for value in data.values())
        common={side:errors(value,end) for side,value in data.items()}
        rows.append({"case_id":case,"common_start_tick":100,"common_end_tick":int(end),
                     "common_post2_metrics":common,
                     "candidate_minus_control_common":{key:common["candidate"][key]-common["control"][key]
                         for key in METRICS} if all(value is not None for value in common.values()) else None,
                     "original_executed_window_metrics":episodes,
                     "completed_position_criteria_only":{side:position_pass(episode) for side,episode in episodes.items()},
                     "motion_contact_descriptors":{side:motion(value) for side,value in data.items()}})
    missing=[row["case_id"] for row in rows if row["candidate_minus_control_common"] is None]
    return {"engine":a["engine"],"suite":suite,"suite_sha256":a["suite_sha256"],
            "policies":{side:report["policy_sha256"] for side,report in reports.items()},
            "cases":rows,"cases_without_common_post2_window":missing,
            "common_window_per_case_metric_means":{side:{key:float(np.mean([row["common_post2_metrics"][side][key] for row in rows]))
                for key in METRICS} for side in cases} if not missing else None,
            "counts":{side:{"cases":expected,"completed":sum(e["completed"] for e in cases[side].values()),
                "falls":sum(e["fallen"] for e in cases[side].values()),
                "position_criteria_only_passes":sum(row["completed_position_criteria_only"][side] for row in rows)} for side in cases},
            "orientation_acceptance":None,
            "note":"All failures retained. Common-window means average matched per-case metrics and are null if any case lacks a window. Original unequal-duration metrics are preserved without deltas or mixed-duration group means. Position criteria here apply only to these development cases, not the formal 60-second/100-episode/3-seed acceptance."}


def build(root):
    result={"training":{side:training(root/f"{side}100") for side in ("control","candidate")},
            "comparisons":{f"{engine}_{suite}":paired(root,engine,suite)
                           for engine in ("physx","mujoco") for suite in ("workspace4","test8")},
            "evidence_limit":"E1 development comparison on provisional hardware parameters; no full pose, hardware, whole-body capability, or formal milestone acceptance.",
            "orientation_acceptance":None}
    for side in ("control","candidate"):
        bundle=root/f"{side}100"/"bundle"
        digest=hashlib.sha256((bundle/"policy.pt").read_bytes()).hexdigest()
        if read_json(bundle/"manifest.json")["policy_sha256"]!=digest:
            raise ValueError(f"Bundle policy hash mismatch: {side}")
        if any(pair["policies"][side]!=digest for pair in result["comparisons"].values()):
            raise ValueError(f"Evaluation used a different policy: {side}")
    return result


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"trace.npz"
        n=150
        values=dict(times=np.arange(1,n+1)*.02,errors=np.full(n,.04),orientation_errors_rad=np.full(n,.2),
                    base=np.tile([0.,0.,.3],(n,1)),base_up_z=np.ones(n),
                    contacts=np.tile([100.,2.,2.,0.,0.,0.],(n,1)),contact_body_names=np.array(["world",*FEET,"base_link"]))
        np.savez(path,**values)
        episode=dict(actual_steps=n,requested_steps=200,completed=False,elapsed_seconds=n*.02,
                     fallen=True,trajectory={"family":"smooth"},rmse_m=.04,p95_m=.04)
        data=trace(path,episode)
        assert errors(data,99) is None
        matched=errors(data,125)
        assert matched["samples"]==26 and math.isclose(matched["rmse_m"],.04)
        assert math.isclose(matched["orientation_p95_rad"],.2)
        assert not position_pass(episode)
        descriptor=motion(data)
        assert descriptor["foot_net_force_gt1N_count_samples"]["2"]==n
        assert descriptor["robot_nonfoot_net_force_gt5N_any_fraction"]==0.
        try:
            trace(path,dict(episode,completed=True))
        except ValueError:
            pass
        else:
            raise AssertionError("False full-duration flag was accepted")
        for side in ("control","candidate"):
            folder=Path(directory)/f"{side}_physx_workspace4"
            episodes=[]
            for case in ("low","high","lateral","far"):
                length=50 if side=="candidate" and case=="low" else n
                target=folder/case
                target.mkdir(parents=True)
                np.savez(target/"trace.npz",**{key:value if key=="contact_body_names" else value[:length]
                                              for key,value in values.items()})
                episodes.append(dict(episode,actual_steps=length,elapsed_seconds=length*.02,
                                     trajectory={"family":"smooth","case_id":case}))
            report=dict(asset_hash="synthetic",suite_sha256="synthetic",policy_sha256=side,
                        task_kind="world_tcp_pose",seed=0,diagnostic=True,engine="PhysX",
                        evaluation={"termination":{}},case_ids=[e["trajectory"]["case_id"] for e in episodes],episodes=episodes)
            (folder/"report.json").write_text(json.dumps(report))
        result=paired(Path(directory),"physx","workspace4")
        assert len(result["cases"])==4
        assert result["cases_without_common_post2_window"]==["low"]
        assert result["common_window_per_case_metric_means"] is None
        assert result["counts"]["candidate"]["falls"]==4
        assert result["counts"]["candidate"]["position_criteria_only_passes"]==0
    print("Synthetic check passed: truncated failure retained, empty post2 window null, common ticks matched, world reaction excluded.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        text=json.dumps(build(args.root),indent=2,allow_nan=False)+"\n"
        if args.output:
            args.output.write_text(text)
        else:
            print(text,end="")
