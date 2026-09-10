"""Read-only per-case E3 development summaries; stdout unless --output is given."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
METRICS = ("ee_position_m", "ee_orientation_rad", "base_linear_velocity_mps", "base_yaw_rate_radps")


def metrics(values, mask):
    if not np.any(mask):
        return None
    return {key: {"rmse": float(np.sqrt(np.mean(value[mask] ** 2))),
                  "p95": float(np.quantile(value[mask], .95))}
            for key, value in values.items()}


def summarize(folder):
    if not folder.exists():
        return {"status": "not_run", "directory": str(folder)}
    report_path = folder / "report.json"
    if not report_path.exists():
        return {"status": "incomplete_no_report", "directory": str(folder)}
    report = json.loads(report_path.read_text())
    if report["task_kind"] != "velocity_ee_pose" or report["ee_target_frame"] != "base_xy_yaw_ground_z":
        raise ValueError(f"Unexpected task/frame: {folder}")
    cases = report["case_ids"]
    if [r["trajectory"]["case_id"] for r in report["episodes"]] != cases or len(set(cases)) != len(cases):
        raise ValueError(f"Missing, duplicated, or reordered report cases: {folder}")
    dt = report["evaluation"]["control_dt"]
    result = {"status": "reported", "directory": str(folder), "engine": report["engine"],
              "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
              "policy_sha256": report["policy_sha256"], "suite_sha256": report["suite_sha256"],
              "completed_episodes": report["summary"]["completed_episodes"],
              "episodes": []}
    for row in report["episodes"]:
        case = row["trajectory"]["case_id"]
        trace_path = folder / case / "trace.npz"
        with np.load(trace_path, allow_pickle=False) as trace:
            times = trace["times"]
            values = dict(zip(METRICS, (trace["errors"], trace["orientation_errors_rad"],
                np.linalg.norm(trace["base_velocity_yaw"][:, :2] - trace["velocity_commands_yaw"][:, :2], axis=1),
                np.abs(trace["base_yaw_rate"] - trace["velocity_commands_yaw"][:, 2]))))
            if len(times) != row["actual_steps"] or len(times) == 0:
                raise ValueError(f"Trace/report count mismatch or empty trace: {trace_path}")
            np.testing.assert_allclose(times, np.arange(1, len(times) + 1) * dt, atol=1e-5, rtol=0)
            if any(v.shape != times.shape or not np.isfinite(v).all() for v in values.values()):
                raise ValueError(f"Invalid trace errors: {trace_path}")
            duration = row["requested_steps"] * dt
            target_ticks = np.arange(len(times))
            target_times = target_ticks * dt
            windows = {}
            for name, start in (("post_transient", 2.),
                                ("steady_command", row["trajectory"]["command_ramp_end_s"]),
                                ("pose_hold", row["trajectory"]["pose_hold_start_s"])):
                # Target k*dt is evaluated against the resulting state (k+1)*dt.
                # Keep the established post2 metric's poststate-time definition.
                first_tick = int(np.ceil(start / dt - 1e-9))
                if name == "post_transient":
                    first_tick -= 1
                    mask = (times >= start) & (target_ticks < row["requested_steps"])
                else:
                    mask = (target_ticks >= first_tick) & (target_ticks < row["requested_steps"])
                complete = bool(row["completed"] and not row["fallen"]
                                and len(times) == row["requested_steps"])
                observed = metrics(values, mask)
                windows[name] = {
                    "selection_time": "poststate_time" if name == "post_transient" else "target_time",
                    "boundary_convention": "Inclusive first/last sampled timestamps; target60 is not an extra requested control tick.",
                    "requested_target_time_bounds_s": [first_tick * dt, duration - dt],
                    "requested_poststate_time_bounds_s": [(first_tick + 1) * dt, duration],
                    "observed_target_time_bounds_s": [float(target_times[mask][0]), float(target_times[mask][-1])] if np.any(mask) else None,
                    "observed_poststate_time_bounds_s": [float(times[mask][0]), float(times[mask][-1])] if np.any(mask) else None,
                    "samples": int(mask.sum()), "complete_no_fall_window": complete,
                    "full_window_metrics": observed if complete else None,
                    "executed_fragment_metrics": observed if not complete else None}
            result["episodes"].append({"case_id": case, "completed": row["completed"],
                "fallen": row["fallen"], "termination_reason": row["termination_reason"],
                "requested_duration_s": duration, "actual_duration_s": float(times[-1]),
                "actual_steps": len(times), "windows": windows,
                "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=DIRECTORY")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = {"initial": OUT / "initial_mujoco_dev8", "250": OUT / "checkpoint250_mujoco_dev8",
            "500": OUT / "checkpoint500_mujoco_dev8", "final": OUT / "final_mujoco_dev8"}
    for entry in args.run:
        label, directory = entry.split("=", 1)
        runs[label] = Path(directory)
    result = {"task_kind": "velocity_ee_pose", "ee_target_frame": "base_xy_yaw_ground_z",
        "command_source": "preset_trajectory", "deployment_commands": "Future operator; not implemented here",
        "evidence_limit": "Training-source development cases, provisional hardware, trained=false; no formal acceptance, world-fixed EE-only success, or causal attribution to extra inputs. No mixed fragment mean.",
        "runs": {label: summarize(folder) for label, folder in runs.items()}}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
