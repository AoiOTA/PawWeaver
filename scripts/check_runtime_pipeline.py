"""Run in pawweaver-runtime to verify CPU inference and independent MuJoCo deployment plumbing."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from pawweaver.mujoco_runtime import MujocoRunner
from pawweaver.trajectories import synthetic

assert importlib.util.find_spec("isaaclab") is None,"Run this check in pawweaver-runtime"
assert importlib.util.find_spec("rsl_rl") is None,"Runtime must remain independent of training framework"
root=Path("artifacts/software-fixture")
runner=MujocoRunner(root,root/"bundle",software_fixture=True)
golden=np.load(root/"bundle/golden.npz",allow_pickle=False)
with torch.inference_mode():
    error=float(np.max(np.abs(runner.policy(torch.from_numpy(golden["observation"])).numpy()-golden["action"])))
assert error<1e-5,error
trajectory=synthetic("line",7,duration=.4).align(runner.state().tcp_pos_w.numpy()[0])
result=runner.evaluate(trajectory,root/"runtime")
report={"kind":"synthetic_software_test","hardware_validation":False,"gpu_cpu_action_max_difference":error,
        "physics_steps":round(runner.data.time/.002),"passed":True}
(root/"runtime/report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
