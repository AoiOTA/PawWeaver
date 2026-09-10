"""Evaluate a frozen world TCP pose suite in PhysX, with fixed batch layout."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def batch_reference(trajectories, origins, steps):
    """Apply only the environment translation; each trajectory retains its own clock."""
    positions=[]; orientations=[]
    for trajectory,origin in zip(trajectories,origins):
        times=trajectory.timestamps[0]+np.arange(steps)*.02
        positions.append(trajectory.sample(times)+origin)
        orientations.append(trajectory.sample_orientation(times))
    return np.asarray(positions),np.asarray(orientations)


def append_batch_step(rows, lengths, fallen, step, values, step_fallen):
    """Archive each case through its first fall or its own requested final step."""
    for slot,record in enumerate(rows):
        if fallen[slot] or step>lengths[slot]:
            continue
        record["times"].append(step*.02)
        for key,value in values.items():
            record[key].append(np.array(value[slot],copy=True))
        fallen[slot]=bool(step_fallen[slot])
    return all(fallen[slot] or step>=lengths[slot] for slot in range(len(rows)))


def save_trace(path, record, contact_body_names):
    """Keep the actual simulator body order beside its contact columns."""
    np.savez_compressed(path,contact_body_names=np.asarray(contact_body_names,dtype=str),
                        **{key:np.asarray(value) for key,value in record.items()})


def main():
    from isaaclab.app import AppLauncher
    from pawweaver.training_inputs import training_inputs,check_training_identity
    from pawweaver.bundle import load_bundle
    from pawweaver.evaluation import load_suite,save_run,completion_status
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("asset","bundle","suite","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--num-envs",type=int,default=1)
    parser.add_argument("--diagnostic",action="store_true",help="Provisional engineering evaluation only")
    parser.add_argument("--provisional-spec",type=Path,help="Required sourced actuator spec for diagnostic mode")
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.num_envs<1:
        parser.error("--num-envs must be positive")
    # Validate the actual asset, actuators and provenance before starting GPU simulation.
    manifest,spec,provisional=training_inputs(args.asset,diagnostic=args.diagnostic,provisional_spec=args.provisional_spec)
    policy,bundle_spec,bundle=load_bundle(args.bundle,manifest["asset_hash"],require_trained=not args.diagnostic)
    check_training_identity(dict(bundle["training_metadata"],asset_hash=bundle["asset_hash"]),
                            dict(asset_hash=manifest["asset_hash"],diagnostic=args.diagnostic,provisional_spec=provisional))
    if bundle_spec.to_dict()!=spec.to_dict():
        raise ValueError("Evaluation actuator spec differs from the policy bundle")
    if bundle.get("training_metadata",{}).get("seed")!=args.seed:
        raise ValueError("Evaluation seed must match the training seed stored in the policy bundle")
    trajectories=load_suite(args.suite)
    lengths=[round((t.timestamps[-1]-t.timestamps[0])/.02) for t in trajectories]
    if not lengths or min(lengths)<1:
        raise ValueError("Evaluation requires nonempty cases lasting at least one control step")
    launcher=AppLauncher(args)
    try:
        import torch
        from pawweaver.isaac_env import WholeBodyEnv
        from pawweaver.task import episode_metrics
        from pawweaver.math import quat_angle_error
        policy=policy.to(args.device)
        config=dict(bundle["training_config"],domain_randomization=False,adaptive_sampling=False,
                    episode_seconds=max(t.timestamps[-1]-t.timestamps[0] for t in trajectories)+.1)
        env=WholeBodyEnv(args.asset,config,args.num_envs,args.device,args.seed,
                        diagnostic=args.diagnostic,provisional_spec=args.provisional_spec)
        ids=torch.arange(args.num_envs,device=args.device)
        origins=env.scene.env_origins.cpu().numpy()
        results=[]; layout=[]
        started=time.perf_counter()
        keys=("times","errors","base","tcp","goal","tcp_quat_w","goal_quat_w","orientation_errors_rad",
              "torques","velocities","actions","observations","contacts","base_up_z","fall_height","fall_tilt")
        for first in range(0,len(trajectories),args.num_envs):
            batch=trajectories[first:first+args.num_envs]
            batch_lengths=lengths[first:first+args.num_envs]
            count=len(batch)
            layout.append([t.metadata["case_id"] for t in batch]+[None]*(args.num_envs-count))
            env.reset(ids)
            positions,orientations=batch_reference(batch,origins[:count],env.max_episode_length+5)
            env.reference.positions[:count]=torch.as_tensor(positions,device=args.device,dtype=torch.float32)
            env.reference.orientations_wxyz[:count]=torch.as_tensor(orientations,device=args.device,dtype=torch.float32)
            state=env.state()
            env.observations.reset(ids,state,env.reference.current(env.episode_length_buf),torch.zeros(args.num_envs,device=args.device),
                                   goal_quat_w=env.reference.current_orientation(env.episode_length_buf))
            env.previous_error.copy_((env.reference.current(env.episode_length_buf)-state.tcp_pos_w).norm(dim=-1))
            rows=[{key:[] for key in keys} for _ in batch]
            fallen=np.zeros(count,dtype=bool)
            with torch.no_grad():
                for step in range(1,max(batch_lengths)+1):
                    observation=env.get_observations()["policy"]
                    action=policy(observation).clamp(-1,1)
                    _,_,done,extras=env.step(action,auto_reset=False)
                    state=env.state()
                    tcp=state.tcp_pos_w[:count].cpu().numpy()-origins[:count]
                    goals=positions[:,step]-origins[:count]
                    goal_quats=orientations[:,step]
                    angles=quat_angle_error(state.tcp_quat_w[:count],torch.as_tensor(goal_quats,device=args.device,dtype=state.tcp_quat_w.dtype))
                    values=dict(errors=np.linalg.norm(tcp-goals,axis=-1),base=state.base_pos_w[:count].cpu().numpy()-origins[:count],
                        tcp=tcp,goal=goals,tcp_quat_w=state.tcp_quat_w[:count].cpu().numpy(),goal_quat_w=goal_quats,
                        orientation_errors_rad=angles.cpu().numpy(),torques=env.torque[:count].cpu().numpy(),
                        velocities=state.joint_vel[:count].cpu().numpy(),actions=action[:count].cpu().numpy(),
                        observations=observation[:count].cpu().numpy(),contacts=env.contacts[:count].cpu().numpy(),
                        base_up_z=extras["base_up_z"][:count].cpu().numpy(),
                        fall_height=extras["fall_height"][:count].cpu().numpy(),
                        fall_tilt=extras["fall_tilt"][:count].cpu().numpy())
                    step_fallen=(done[:count]&~extras["time_outs"][:count]).cpu().numpy()
                    if append_batch_step(rows,batch_lengths,fallen,step,values,step_fallen):
                        break
            for slot,(trajectory,record) in enumerate(zip(batch,rows)):
                result=episode_metrics(record["times"],record["errors"],record["base"],record["torques"],record["velocities"],fallen[slot],
                                       orientation_errors=record["orientation_errors_rad"])
                result.update(completion_status(batch_lengths[slot],len(record["times"]),fallen[slot]))
                result.update(engine="PhysX",trajectory=trajectory.metadata,policy_sha256=bundle["policy_sha256"],
                              diagnostic=args.diagnostic,elapsed_seconds=record["times"][-1],batch_index=len(layout)-1,env_index=slot,
                              fall_height=bool(record["fall_height"][-1]),fall_tilt=bool(record["fall_tilt"][-1]))
                output=args.output/trajectory.metadata["case_id"];output.mkdir(parents=True,exist_ok=True)
                save_trace(output/"trace.npz",record,env.robot.body_names)
                (output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
                results.append(result)
                print(json.dumps(result),flush=True)
        evaluation=dict(num_envs=args.num_envs,batch_layout=layout,wall_elapsed_seconds=time.perf_counter()-started,
                        diagnostic=args.diagnostic,provisional_spec=provisional,termination=env.termination,
                        evidence_limit="Provisional engineering only; no hardware validity or formal acceptance." if args.diagnostic else "Pose acceptance thresholds remain unspecified.")
        save_run(args.output,args.suite,manifest,bundle,args.seed,"PhysX",results,evaluation=evaluation)
        env.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        import sys
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__=="__main__":
    main()
