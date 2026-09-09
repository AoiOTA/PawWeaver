"""Evaluate the identical TorchScript bundle and frozen world targets in PhysX."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
for name in ("asset","bundle","suite","output"):
    parser.add_argument("--"+name,type=Path,required=True)
parser.add_argument("--seed",type=int,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
from pawweaver.assets.build import verify_asset
manifest=verify_asset(args.asset)
launcher=AppLauncher(args)
try:
    import numpy as np
    import torch
    from pawweaver.bundle import load_bundle
    from pawweaver.isaac_env import WholeBodyEnv
    from pawweaver.evaluation import load_suite,save_run
    from pawweaver.task import episode_metrics
    policy,spec,bundle=load_bundle(args.bundle,manifest["asset_hash"])
    if bundle.get("training_metadata",{}).get("seed")!=args.seed:
        raise ValueError("Evaluation seed must match the training seed stored in the policy bundle")
    policy=policy.to(args.device)
    trajectories=load_suite(args.suite)
    config=dict(bundle["training_config"],domain_randomization=False,adaptive_sampling=False,
                episode_seconds=max(t.timestamps[-1] for t in trajectories)+.1)
    env=WholeBodyEnv(args.asset,config,1,args.device,args.seed)
    ids=torch.tensor([0],device=args.device)
    results=[]
    for trajectory in trajectories:
        env.reset(ids)
        times=np.arange(env.max_episode_length+5)*.02
        targets=trajectory.sample(times)+env.scene.env_origins[0].cpu().numpy()
        env.reference.positions[0]=torch.tensor(targets,device=args.device,dtype=torch.float32)
        state=env.state()
        env.observations.reset(ids,state,env.reference.current(env.episode_length_buf),torch.zeros(1,device=args.device))
        env.previous_error.copy_((env.reference.current(env.episode_length_buf)-state.tcp_pos_w).norm(dim=-1))
        rows={key:[] for key in ("times","errors","base","tcp","goal","torques","velocities","actions","observations","contacts")}
        fallen=False
        with torch.inference_mode():
            for index in range(round(trajectory.timestamps[-1]/.02)):
                observation=env.get_observations()["policy"]
                action=policy(observation).clamp(-1,1)
                _,_,done,extras=env.step(action,auto_reset=False)
                state=env.state()
                timestamp=(index+1)*.02
                tcp=state.tcp_pos_w[0].cpu().numpy()-env.scene.env_origins[0].cpu().numpy()
                base=state.base_pos_w[0].cpu().numpy()-env.scene.env_origins[0].cpu().numpy()
                goal=trajectory.sample(timestamp)
                values=(timestamp,float(np.linalg.norm(tcp-goal)),base,tcp,goal,env.torque[0].cpu().numpy(),
                        state.joint_vel[0].cpu().numpy(),action[0].cpu().numpy(),observation[0].cpu().numpy(),env.contacts[0].cpu().numpy())
                for key,value in zip(rows,values):
                    rows[key].append(np.array(value,copy=True))
                fallen=bool(done[0] and not extras["time_outs"][0])
                if fallen:
                    break
        result=episode_metrics(rows["times"],rows["errors"],rows["base"],rows["torques"],rows["velocities"],fallen)
        result.update(engine="PhysX",trajectory=trajectory.metadata,bundle_hash=bundle["policy_sha256"])
        output=args.output/trajectory.metadata["case_id"];output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/"trace.npz",**{key:np.asarray(value) for key,value in rows.items()})
        (output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
        results.append(result)
        print(json.dumps(result),flush=True)
    save_run(args.output,args.suite,manifest,bundle,args.seed,"PhysX",results)
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    import sys
    launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))
