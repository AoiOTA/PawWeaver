"""Exercise actual wrist rendering, 30/50 Hz scheduling, delay and hold on synthetic boxes."""
import json
from pathlib import Path
import numpy as np
from pawweaver.mujoco_runtime import MujocoRunner
from pawweaver.visual_runtime import run_visual
from pawweaver.trajectories import Trajectory
from pawweaver.math import rpy_quat

root=Path("artifacts/software-fixture")
runner=MujocoRunner(root,root/"bundle",software_fixture=True)
start=runner.data.camera("rgbd_color").xpos+np.array([.7,0,0])
trajectory=Trajectory(np.array([0.,1.]),np.repeat(start[None],2,axis=0),
    np.repeat(rpy_quat([0,-np.pi/2,0])[None],2,axis=0),{"family":"reach","source":"synthetic"})
scenario={"marker_id":7,"marker_size_m":.16,"marker_to_goal":[0,0,0],"marker_to_goal_quat_wxyz":[1,0,0,0],
          "latency_s":.01,"max_age_s":.025,"occlusion_intervals_s":[[.02,.06]],"depth_missing_probability":.05,
          "position_noise_m":.001,"seed":731}
report=run_visual(runner,root,trajectory,scenario,root/"vision",duration=.14,record=True)
assert len(report["captures"])==5,len(report["captures"])
assert report["measurement_rmse_m"] is not None and report["measurement_rmse_m"]<.02,report["measurement_rmse_m"]
assert 0<report["hold_fraction"]<1
assert any(r["occluded"] and not r["measured"] for r in report["captures"])
assert report["recovery_seconds"][0] is not None
result={k:v for k,v in report.items() if k not in ("controls","captures","scenario")}
result.update(passed=True,hardware_validation=False,kind="synthetic_rendered_closed_loop_test")
(root/"vision/check.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
