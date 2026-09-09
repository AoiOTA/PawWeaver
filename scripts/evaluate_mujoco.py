"""Run a frozen goal suite with the standalone exported policy."""
import argparse
import json
from pathlib import Path
import torch
from pawweaver.mujoco_runtime import MujocoRunner
from pawweaver.evaluation import load_suite,save_run

parser=argparse.ArgumentParser(description=__doc__)
for name in ("asset","bundle","suite","output"):
    parser.add_argument("--"+name,type=Path,required=True)
parser.add_argument("--seed",type=int,required=True,help="Training seed of this policy")
args=parser.parse_args()
torch.set_num_threads(1)
runner=MujocoRunner(args.asset,args.bundle)
if runner.bundle.get("training_metadata",{}).get("seed")!=args.seed:
    raise ValueError("Evaluation seed must match the training seed stored in the policy bundle")
results=[]
for trajectory in load_suite(args.suite):
    result=runner.evaluate(trajectory,args.output/trajectory.metadata["case_id"])
    results.append(result)
    print(json.dumps(result),flush=True)
save_run(args.output,args.suite,runner.manifest,runner.bundle,args.seed,"MuJoCo",results)
