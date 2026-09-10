"""CPU aggregation of the completed lower-LR-floor E2 run and its saved readouts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_totals(rows,groups):
    total_transitions=sum(sum(row[f"demonstration_{group}_transitions"] for group in groups) for row in rows)
    result={}
    for group in groups:
        prefix=f"demonstration_{group}_"
        fields=("transitions","falls","resets","timeouts","ended_episodes","ended_episode_seconds_sum",
                "position_error_sum_m","orientation_error_sum_rad","nonterminal_preclip_sum","nonterminal_postclip_sum")
        values={key:sum(row[prefix+key] for row in rows) for key in fields}
        samples=values["transitions"]
        values["transition_fraction"]=samples/total_transitions if total_transitions else None
        for summed,mean in (("position_error_sum_m","position_error_mean_m"),
                            ("orientation_error_sum_rad","orientation_error_mean_rad"),
                            ("nonterminal_preclip_sum","nonterminal_preclip_mean"),
                            ("nonterminal_postclip_sum","nonterminal_postclip_mean")):
            values[mean]=values[summed]/samples if samples else None
        values["mean_ended_episode_seconds"]=values["ended_episode_seconds_sum"]/values["ended_episodes"] if values["ended_episodes"] else None
        result[group]=values
    return result


def training_summary():
    import torch
    folder=ROOT/"train_lr_floor1000"
    run=read(folder/"run.json")
    rows=[json.loads(line) for line in (folder/"metrics.jsonl").read_text().splitlines()]
    progress=run["training_progress"]
    if [row["iteration"] for row in rows]!=list(range(1000)) or progress!={
        "status":"completed","stop_reason":"iteration_budget","requested_iterations":1000,
        "completed_iterations":1000,"start_iteration":0,"last_completed_iteration":999,"completed_transitions":98304000}:
        raise ValueError("Expected actual complete continuous 1000-update-iteration E2 run")
    config=run["config"]
    transitions=len(rows)*run["num_envs"]*config["rollout_steps"]
    if transitions!=progress["completed_transitions"]:
        raise ValueError("Actual transition budget differs from recorded progress")
    checkpoint_path=folder/"checkpoint_000999.pt"
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    if checkpoint["iteration"]!=999 or checkpoint["metadata"]["training_progress"]!=progress:
        raise ValueError("Final checkpoint does not include the completed run")
    bundle_path=ROOT/"lr_floor_checkpoint999_bundle"
    bundle=read(bundle_path/"manifest.json")
    if bundle["asset_hash"]!=run["asset_hash"] or digest(bundle_path/"policy.pt")!=bundle["policy_sha256"]:
        raise ValueError("Final bundle identity mismatch")
    nonfinite=[{"iteration":row["iteration"],"field":key,"value":str(value)} for row in rows for key,value in row.items()
               if isinstance(value,(int,float)) and not math.isfinite(value)]
    groups=group_totals(rows,list(config["demonstration_group_weights"]))
    if sum(values["transitions"] for values in groups.values())!=transitions:
        raise ValueError("Group exposure does not cover all actual transitions")
    for key in ("falls","resets"):
        if sum(values[key] for values in groups.values())!=sum(row[key] for row in rows):
            raise ValueError(f"Group and global {key} disagree")
    updates=sum(row["optimizer_steps"] for row in rows)
    if updates!=len(rows)*config["learning_epochs"]*config["mini_batches"]:
        raise ValueError("Recorded optimizer steps differ from completed PPO budget")
    saturated=sum(row["torque_saturated"] for row in rows)
    torque_samples=sum(row["torque_samples"] for row in rows)
    keys=[key for key in rows[0] if key.startswith(("reward_","umi_")) or key in (
        "learning_rate","leg_action_std_mean","arm_action_std_mean","tracking_error_mean_m",
        "orientation_error_mean_rad","kl","value","surrogate")]
    def window(selected):
        return {"first_iteration":selected[0]["iteration"],"last_iteration":selected[-1]["iteration"],
                "count":len(selected),"means":{key:float(np.mean([row[key] for row in selected])) for key in keys},
                "groups":group_totals(selected,list(groups)),"falls":sum(row["falls"] for row in selected)}
    return {"run":"train_lr_floor1000","training_progress":progress,"actual_process_exit_code":0,
            "exit_code_source":"Master GPU operator confirmed actual process completion; separate from metrics inference.",
            "optimizer_updates":updates,"transitions":transitions,
            "loop_seconds":sum(run["num_envs"]*config["rollout_steps"]/row["steps_per_second"] for row in rows),
            "loop_seconds_scope":"Sum of recorded rollout/PPO loop timings; excludes subsequent logging/checkpoint/export and is not whole-process wall time.",
            "all_recorded_scalars_finite":not nonfinite,"nonfinite_entries":nonfinite,
            "all_diagnostic_finite_checks_passed":all(row["finite_checks_passed"] is True for row in rows),
            "falls":sum(row["falls"] for row in rows),"resets":sum(row["resets"] for row in rows),
            "torque_saturated":saturated,"torque_samples":torque_samples,"saturation_fraction":saturated/torque_samples,
            "collision_control_samples":sum(row["collision_control_samples"] for row in rows),
            "groups":groups,"configured_demonstration_group_weights":config["demonstration_group_weights"],
            "group_scope":"Actual per-transition task exposure before reset, not equal group weights or learned capability. Duration averages include ended episodes only; ongoing final episodes remain censored. Reward sums exclude dt and terminal penalty.",
            "first25":window(rows[:25]),"last25":window(rows[-25:]),
            "quarter_windows":{f"{start}-{start+249}":window(rows[start:start+250]) for start in range(0,1000,250)},
            "learning_rate":{"initial_config":config["learning_rate"],"floor":config["minimum_learning_rate"],
                "recorded_min":min(row["learning_rate"] for row in rows),"recorded_max":max(row["learning_rate"] for row in rows),
                "last":rows[-1]["learning_rate"]},
            "metadata":{"git_commit":run["git_commit"],"git_dirty":run["git_dirty"],"seed":run["seed"],
                "asset_hash":run["asset_hash"],"initialize_from":run["initialize_from"],"source_sha256":run["source_sha256"]},
            "final_checkpoint":str(checkpoint_path),"final_checkpoint_sha256":digest(checkpoint_path),
            "final_policy_sha256":bundle["policy_sha256"],"trained":bundle["trained"],
            "scope":"This restarted lower-LR-floor 1000 run only. The earlier stopped 78-iteration attempt is separate; no equal-budget causal comparison against it. Finite training and group names do not establish task or whole-body capability acceptance.",
            "orientation_acceptance":None}


def report_and_traces(folder):
    report=read(folder/"report.json")
    episodes=report["episodes"]
    if [e["trajectory"]["case_id"] for e in episodes]!=report["case_ids"] or len(set(report["case_ids"]))!=len(episodes):
        raise ValueError(f"Missing/duplicate episode identities: {folder}")
    traces={}
    for episode in episodes:
        case=episode["trajectory"]["case_id"]
        if episode["policy_sha256"]!=report["policy_sha256"]:
            raise ValueError(f"Episode/report policy differs: {folder}/{case}")
        with np.load(folder/case/"trace.npz",allow_pickle=False) as saved:
            data={key:saved[key] for key in ("times","errors","orientation_errors_rad","base","base_up_z")}
        if any(not np.isfinite(value).all() for value in data.values()):
            raise ValueError(f"Nonfinite trace: {folder}/{case}")
        ticks=np.rint(data["times"]/.02).astype(int)
        if not np.array_equal(ticks,np.arange(1,len(ticks)+1)) or not np.allclose(data["times"],ticks*.02,rtol=0,atol=1e-8):
            raise ValueError(f"Noncontinuous control ticks: {folder}/{case}")
        if len(ticks)!=episode["actual_steps"] or not 0<len(ticks)<=episode["requested_steps"]:
            raise ValueError(f"Trace/report step count mismatch: {folder}/{case}")
        if episode["completed"]!=(episode["actual_steps"]==episode["requested_steps"]):
            raise ValueError(f"Completion/step count mismatch: {folder}/{case}")
        data["ticks"]=ticks
        traces[case]=data
    return report,traces


def error_metrics(data,mask):
    if not mask.any():
        return None
    e=data["errors"][mask].astype(float);a=data["orientation_errors_rad"][mask].astype(float)
    return {"position_rmse_m":float(np.sqrt(np.mean(e**2))),"position_p95_m":float(np.quantile(e,.95)),
            "orientation_rmse_rad":float(np.sqrt(np.mean(a**2))),"orientation_p95_rad":float(np.quantile(a,.95))}


def hold_metrics(data,start_tick,end_tick):
    mask=(data["ticks"]>=start_tick)&(data["ticks"]<=end_tick)
    result={"planned_tick_interval_inclusive":[start_tick,end_tick],"actual_samples":int(mask.sum()),
            "requested_samples":end_tick-start_tick+1,"completed_requested_window":int(mask.sum())==end_tick-start_tick+1,
            "first_actual_s":float(data["times"][mask][0]) if mask.any() else None,
            "last_actual_s":float(data["times"][mask][-1]) if mask.any() else None}
    result.update(error_metrics(data,mask) or {key:None for key in (
        "position_rmse_m","position_p95_m","orientation_rmse_rad","orientation_p95_rad")})
    result["base_height_mean_m"]=float(data["base"][mask,2].mean()) if mask.any() else None
    result["base_tilt_p95_rad"]=float(np.quantile(np.arccos(np.clip(data["base_up_z"][mask],-1.,1.)),.95)) if mask.any() else None
    return result


def common_comparison(before_folder,after_folder,*,kind="same_engine"):
    a,ta=report_and_traces(before_folder);b,tb=report_and_traces(after_folder)
    for key in ("task_kind","asset_hash","suite_sha256","seed","case_ids"):
        if a[key]!=b[key]:
            raise ValueError(f"Comparison input mismatch: {key}")
    if kind=="same_engine" and a["engine"]!=b["engine"]:
        raise ValueError("Learning comparisons require same engine")
    if kind=="same_policy_cross_engine" and a["policy_sha256"]!=b["policy_sha256"]:
        raise ValueError("Cross-engine comparison requires same final policy")
    if a["evaluation"]["termination"]!=b["evaluation"]["termination"]:
        raise ValueError("Comparison termination rules differ")
    pairs=[]
    for ea,eb in zip(a["episodes"],b["episodes"]):
        case=ea["trajectory"]["case_id"]
        if ea["requested_steps"]!=eb["requested_steps"]:
            raise ValueError(f"Comparison requested durations differ: {case}")
        end=min(ta[case]["ticks"][-1],tb[case]["ticks"][-1])
        common={side:error_metrics(data,(data["ticks"]>=100)&(data["ticks"]<=end))
                for side,data in (("before",ta[case]),("after",tb[case]))}
        hold=None
        if "hold_start_s" in ea["trajectory"]:
            if ea["trajectory"]["hold_start_s"]!=eb["trajectory"]["hold_start_s"]:
                raise ValueError(f"Comparison hold starts differ: {case}")
            hold_start=round(ea["trajectory"]["hold_start_s"]/.02)+1
            hold_end=ea["requested_steps"]
        elif a["case_ids"]==["low","high","lateral","far"]:
            hold_start=round(ea["requested_steps"]*.4);hold_end=round(ea["requested_steps"]*.6)
        else:
            hold_start=hold_end=None
        if hold_start is not None:
            actual_end=min(int(end),hold_end)
            hold_values={side:error_metrics(data,(data["ticks"]>=hold_start)&(data["ticks"]<=actual_end))
                         for side,data in (("before",ta[case]),("after",tb[case]))}
            hold={"planned_tick_interval_inclusive":[hold_start,hold_end],"common_end_tick":actual_end,
                  "common_samples":max(0,actual_end-hold_start+1),"metrics":hold_values,
                  "after_minus_before":{key:hold_values["after"][key]-hold_values["before"][key] for key in hold_values["before"]}
                      if hold_values["before"] is not None and hold_values["after"] is not None else None}
        pairs.append({"case_id":case,"common_start_tick":100,"common_end_tick":int(end),
                      "common_samples":max(0,int(end)-99),"common_post2":common,
                      "common_hold":hold,
                      "after_minus_before":{key:common["after"][key]-common["before"][key] for key in common["before"]}
                          if common["before"] is not None and common["after"] is not None else None,
                      "original_executed_window_metrics":{"before":ea,"after":eb}})
    absent=[row["case_id"] for row in pairs if row["after_minus_before"] is None]
    return {"before_path":str(before_folder),"after_path":str(after_folder),"comparison_kind":kind,
            "before_engine":a["engine"],"after_engine":b["engine"],
            "before_policy_sha256":a["policy_sha256"],"after_policy_sha256":b["policy_sha256"],
            "cases":pairs,"cases_without_common_post2":absent,
            "equal_case_common_window_means":{side:{key:float(np.mean([row["common_post2"][side][key] for row in pairs]))
                for key in pairs[0]["common_post2"][side]} for side in ("before","after")} if not absent else None,
            "scope":"Each delta uses identical 50Hz post2 ticks ending at the earlier stop. All failures and original unequal-duration metrics retained; no raw-duration delta. A missing common window makes group means null.",
            "orientation_acceptance":None}


def workspace_readout(engine):
    folder=ROOT/f"lr_floor_checkpoint999_{engine}_workspace4"
    report,traces=report_and_traces(folder)
    support=read(ROOT/f"lr_floor_checkpoint999_{engine}_support.json")
    if support["policy_sha256"]!=report["policy_sha256"]:
        raise ValueError("Final support/report policy differs")
    if report["case_ids"]!=["low","high","lateral","far"]:
        raise ValueError("Expected all fixed workspace4 cases")
    support_cases={row["case_id"]:row for row in support["cases"]}
    cases=[]
    for episode in report["episodes"]:
        case=episode["trajectory"]["case_id"]
        duration=episode["requested_steps"]*.02
        start,end=round(duration*.4/.02),round(duration*.6/.02)
        support_hold=support_cases[case]["windows"]["designed_middle_hold"]
        hold=hold_metrics(traces[case],start,end)
        hold["planned_window_seconds"]=[duration*.4,duration*.6]
        cases.append({"case_id":case,**{key:episode[key] for key in ("requested_steps","actual_steps","completed","fallen")},
                      "position_rmse_m":episode["rmse_m"],"position_p95_m":episode["p95_m"],
                      "orientation_rmse_rad":episode["orientation_rmse_rad"],"orientation_p95_rad":episode["orientation_p95_rad"],
                      "hold_foot_bottom_mean_m":{name:values["sphere_bottom_height_mean_m"] for name,values in support_hold["feet"].items()} if support_hold else None,
                      "hold_nonfoot_ground_pair_fraction":support_hold["nonfoot_ground_pairs"]["any_fraction"]
                          if support_hold and support_hold["nonfoot_ground_pairs"] is not None else None,
                      "hold_support_samples":support_hold["samples"] if support_hold else 0,
                      "hold_support_scope":"Support excludes final post-state without a next observation; task hold metrics include all recorded states, including failure.",
                      "designed_hold_metrics":hold})
    return {"completed_iterations":1000,"new_transitions":98304000,"evaluation_exit_code":0,"support_readout_exit_code":0,
            "engine":report["engine"],"policy_sha256":report["policy_sha256"],"cases":cases}


def train10_readout(engine):
    folder=ROOT/f"lr_floor_checkpoint999_{engine}_train10"
    report,traces=report_and_traces(folder)
    if report["case_ids"]!=[e["case_id"] for e in read(ROOT/"train/manifest.json")["cases"]]:
        raise ValueError("Expected all ten training tasks")
    rows=[]
    for episode in report["episodes"]:
        case=episode["trajectory"]["case_id"]
        # Preserve existing train10 format: strictly after hold_start_s.
        start=episode["trajectory"]["hold_start_s"]
        hold=hold_metrics(traces[case],round(start/.02)+1,episode["requested_steps"])
        hold.update(requested_start_s=start,requested_end_s=episode["requested_steps"]*.02,samples=hold["actual_samples"])
        rows.append({"case_id":case,**{key:episode[key] for key in ("completed","fallen","requested_steps","actual_steps")},
                     "position_rmse_m":episode["rmse_m"],"position_p95_m":episode["p95_m"],
                     "orientation_rmse_rad":episode["orientation_rmse_rad"],"orientation_p95_rad":episode["orientation_p95_rad"],"hold":hold})
    return {"evaluation_exit_code":0,"engine":report["engine"],"policy_sha256":report["policy_sha256"],"cases":rows}


def evaluation_readouts():
    learning=read(ROOT/"learning_readouts.json")
    train10=read(ROOT/"train10_learning_readouts.json")
    summary=read(ROOT/"final1000_summary.json")
    final_hash=summary["final_policy_sha256"]
    reports={}
    for engine in ("physx","mujoco"):
        for suite in ("workspace4","test8","train10"):
            folder=ROOT/f"lr_floor_checkpoint999_{engine}_{suite}"
            report,_=report_and_traces(folder)
            if report["policy_sha256"]!=final_hash:
                raise ValueError("Final evaluation used a different policy")
            reports[f"{engine}_{suite}"]={"path":str(folder),"actual_process_exit_code":0,
                "case_count":len(report["episodes"]),"completed_case_ids":[e["trajectory"]["case_id"] for e in report["episodes"] if e["completed"]],
                "failed_case_ids":[e["trajectory"]["case_id"] for e in report["episodes"] if not e["completed"] or e["fallen"]],
                "episodes":report["episodes"],"orientation_acceptance":None}
    learning["checkpoints"]["999"]=workspace_readout("mujoco")
    learning["final_physx_workspace4"]=workspace_readout("physx")
    initial=ROOT.parent/"plan_v3_reward_signal"
    learning["initial_vs_final"]={f"{engine}_{suite}":common_comparison(initial/f"candidate_{engine}_{suite}",ROOT/f"lr_floor_checkpoint999_{engine}_{suite}")
        for engine in ("physx","mujoco") for suite in ("workspace4","test8")}
    learning["development_vs_final"]={str(checkpoint):common_comparison(ROOT/f"lr_floor_checkpoint{checkpoint}_mujoco_workspace4",ROOT/"lr_floor_checkpoint999_mujoco_workspace4")
        for checkpoint in (250,500)}
    learning["comparison_scope"]="Initial is the selected E1 candidate initializer before this E2 run. Development250/500 comparisons remain separate along the same training trajectory; a checkpoint index is zero-based. Only common-tick deltas support learning direction comparisons."
    train10["readouts"]["checkpoint999"]=train10_readout("mujoco")
    train10["final_physx"]=train10_readout("physx")
    train10["initial_vs_final"]=common_comparison(ROOT/"initial_mujoco_train10",ROOT/"lr_floor_checkpoint999_mujoco_train10")
    train10["development500_vs_final"]=common_comparison(ROOT/"lr_floor_checkpoint500_mujoco_train10",ROOT/"lr_floor_checkpoint999_mujoco_train10")
    train10["final_cross_engine"]=common_comparison(ROOT/"lr_floor_checkpoint999_physx_train10",ROOT/"lr_floor_checkpoint999_mujoco_train10",kind="same_policy_cross_engine")
    summary["final_evaluations"]=reports
    summary["evaluation_exit_code_source"]="Master operator confirmed each full evaluation process exit0 before this aggregation; failed task episodes remain present."
    # Complete every read and validation before modifying the three owned JSONs.
    outputs=[(filename,json.dumps(result,indent=2,allow_nan=False)+"\n") for filename,result in (
        ("final1000_summary.json",summary),("learning_readouts.json",learning),("train10_learning_readouts.json",train10))]
    for filename,text in outputs:
        (ROOT/filename).write_text(text)
    return {key:{"case_count":value["case_count"],"completed":len(value["completed_case_ids"]),"failed":value["failed_case_ids"]} for key,value in reports.items()}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--training",action="store_true")
    mode.add_argument("--readouts",action="store_true",help="Use only after the operator confirms all six evaluations and both support readouts exited")
    args=parser.parse_args()
    if args.training:
        result=training_summary()
        (ROOT/"final1000_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({key:result[key] for key in ("optimizer_updates","transitions","loop_seconds","falls","resets","groups")},indent=2))
    else:
        print(json.dumps(evaluation_readouts(),indent=2))
