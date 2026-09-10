"""Evaluate explicit velocity commands plus yaw-frame EE targets in MuJoCo."""
import argparse
import json
from pathlib import Path

import torch
from pawweaver.mujoco_runtime import CommandedMujocoRunner
from pawweaver.commanded_pose import load_command_suite
from pawweaver.commanded_evaluation import save_command_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('asset','bundle','suite','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--provisional-spec',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    trajectories=load_command_suite(args.suite)
    runner=CommandedMujocoRunner(args.asset,args.bundle,diagnostic=args.diagnostic,provisional_spec=args.provisional_spec)
    if runner.bundle.get('training_metadata',{}).get('seed')!=args.seed:
        raise ValueError('Evaluation seed must match the training seed')
    results=[]
    for trajectory in trajectories:
        result=runner.evaluate(trajectory,args.output/trajectory.metadata['case_id'])
        results.append(result)
        print(json.dumps(result),flush=True)
    save_command_run(args.output,args.suite,runner.manifest,runner.bundle,args.seed,'MuJoCo',results,
        evaluation=dict(termination=runner.termination,provisional_spec=runner.provisional,
            control_dt=.02,command_source='preset_trajectory'))


if __name__=='__main__':
    main()
