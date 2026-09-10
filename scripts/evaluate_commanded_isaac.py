"""Evaluate velocity commands plus yaw-frame EE targets using fixed PhysX batches."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def batch_reference(trajectories,steps):
    """Task-frame samples stay task-frame; environment origins are not added."""
    positions=[];orientations=[];commands=[]
    for trajectory in trajectories:
        times=trajectory.timestamps[0]+np.arange(steps)*.02
        positions.append(trajectory.sample(times))
        orientations.append(trajectory.sample_orientation(times))
        commands.append(trajectory.sample_command(times))
    return np.asarray(positions),np.asarray(orientations),np.asarray(commands)


def append_step(rows,lengths,fallen,step,values,step_fallen):
    for slot,row in enumerate(rows):
        if fallen[slot] or step>lengths[slot]:
            continue
        row['times'].append(step*.02)
        for key,value in values.items():
            row[key].append(np.array(value[slot],copy=True))
        fallen[slot]=bool(step_fallen[slot])
    return all(fallen[slot] or step>=lengths[slot] for slot in range(len(rows)))


def verify_scored_step(extras,expected,count):
    """Check the Env's scored k sample against independently reconstructed k data."""
    import torch
    for key,value in expected.items():
        if not torch.allclose(extras[key][:count],value,rtol=1e-5,atol=1e-6):
            raise ValueError(f'Env/evaluator commanded step differs in {key}')


def main():
    from isaaclab.app import AppLauncher
    from pawweaver.training_inputs import training_inputs,check_training_identity
    from pawweaver.bundle import load_bundle
    from pawweaver.observations import CommandObservationSpec
    from pawweaver.commanded_pose import load_command_suite,task_pose_to_world,yaw_linear_velocity,yaw_rate
    from pawweaver.commanded_evaluation import commanded_metrics,save_command_run
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('asset','bundle','suite','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--num-envs',type=int,default=1)
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--provisional-spec',type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.num_envs<1:
        parser.error('--num-envs must be positive')
    manifest,spec,provisional=training_inputs(args.asset,diagnostic=args.diagnostic,provisional_spec=args.provisional_spec)
    policy,bundle_spec,bundle=load_bundle(args.bundle,manifest['asset_hash'],require_trained=not args.diagnostic,
        observation_spec=CommandObservationSpec())
    if bundle['training_config'].get('task_mode')!='velocity_ee_pose':
        raise ValueError('Command evaluator requires velocity_ee_pose training config')
    check_training_identity(dict(bundle['training_metadata'],asset_hash=bundle['asset_hash']),
        dict(asset_hash=manifest['asset_hash'],diagnostic=args.diagnostic,provisional_spec=provisional))
    if bundle_spec.to_dict()!=spec.to_dict():
        raise ValueError('Evaluation actuator spec differs from policy bundle')
    if bundle['training_metadata'].get('seed')!=args.seed:
        raise ValueError('Evaluation seed must match the training seed')
    trajectories=load_command_suite(args.suite)
    lengths=[round((t.timestamps[-1]-t.timestamps[0])/.02) for t in trajectories]
    if not lengths or min(lengths)<1:
        raise ValueError('Evaluation requires nonempty cases lasting at least one control step')
    launcher=AppLauncher(args)
    try:
        import torch
        from pawweaver.isaac_env import WholeBodyEnv
        from pawweaver.math import quat_angle_error
        policy=policy.to(args.device)
        config=dict(bundle['training_config'],domain_randomization=False,adaptive_sampling=False,
            episode_seconds=max(t.timestamps[-1]-t.timestamps[0] for t in trajectories)+.1)
        env=WholeBodyEnv(args.asset,config,args.num_envs,args.device,args.seed,
            diagnostic=args.diagnostic,provisional_spec=args.provisional_spec)
        ids=torch.arange(args.num_envs,device=args.device)
        ground_z=env.scene.env_origins[:,2]
        results=[];layout=[];started=time.perf_counter()
        keys=('times','errors','base','base_quat_w','ground_z','tcp','goal','goal_task','goal_quat_task',
            'velocity_commands_yaw','base_velocity_yaw','base_yaw_rate','joint_positions',
            'torques','velocities','actions','observations','tcp_quat_w','goal_quat_w',
            'orientation_errors_rad','contacts','base_up_z','fall_height','fall_tilt')
        for first in range(0,len(trajectories),args.num_envs):
            batch=trajectories[first:first+args.num_envs];batch_lengths=lengths[first:first+args.num_envs]
            count=len(batch)
            layout.append([t.metadata['case_id'] for t in batch]+[None]*(args.num_envs-count))
            env.reset(ids)
            positions,orientations,commands=batch_reference(batch,env.reference.positions.shape[1])
            env.reference.positions[:count]=torch.as_tensor(positions,device=args.device,dtype=torch.float32)
            env.reference.orientations_wxyz[:count]=torch.as_tensor(orientations,device=args.device,dtype=torch.float32)
            env.reference.velocity_commands_yaw[:count]=torch.as_tensor(commands,device=args.device,dtype=torch.float32)
            state=env.state()
            initial_t=env.reference.current(env.episode_length_buf)
            initial_q=env.reference.current_orientation(env.episode_length_buf)
            env.observations.reset(ids,state,initial_t,torch.zeros(args.num_envs,device=args.device),goal_quat_t=initial_q)
            initial_w,_=task_pose_to_world(initial_t,initial_q,state.base_pos_w,state.base_quat_w,ground_z)
            env.previous_error.copy_((initial_w-state.tcp_pos_w).norm(dim=-1))
            rows=[{key:[] for key in keys} for _ in batch];fallen=np.zeros(count,dtype=bool)
            with torch.no_grad():
                for k in range(max(batch_lengths)):
                    pre_quat=env.state().base_quat_w[:count].clone()
                    observation=env.get_observations()['policy']
                    action=policy(observation).clamp(-1,1)
                    _,_,done,extras=env.step(action,auto_reset=False)
                    state=env.state()
                    target_t=torch.as_tensor(positions[:,k],device=args.device,dtype=state.tcp_pos_w.dtype)
                    quat_t=torch.as_tensor(orientations[:,k],device=args.device,dtype=state.tcp_pos_w.dtype)
                    goal,goal_quat=task_pose_to_world(target_t,quat_t,state.base_pos_w[:count],state.base_quat_w[:count],ground_z[:count])
                    linear=yaw_linear_velocity(state.base_lin_vel_b[:count],state.base_quat_w[:count])
                    rate=yaw_rate(pre_quat,state.base_quat_w[:count],.02)
                    verify_scored_step(extras,dict(commanded_position_t=target_t,commanded_orientation_t=quat_t,
                        velocity_command_yaw=torch.as_tensor(commands[:,k],device=args.device,dtype=target_t.dtype),
                        commanded_goal_w=goal,commanded_goal_quat_w=goal_quat,
                        base_linear_velocity_yaw=linear,base_yaw_rate=rate),count)
                    values=dict(errors=(state.tcp_pos_w[:count]-goal).norm(dim=-1).cpu().numpy(),
                        base=state.base_pos_w[:count].cpu().numpy(),base_quat_w=state.base_quat_w[:count].cpu().numpy(),
                        ground_z=ground_z[:count].cpu().numpy(),tcp=state.tcp_pos_w[:count].cpu().numpy(),
                        goal=goal.cpu().numpy(),goal_task=positions[:,k],goal_quat_task=orientations[:,k],
                        velocity_commands_yaw=commands[:,k],base_velocity_yaw=linear.cpu().numpy(),base_yaw_rate=rate.cpu().numpy(),
                        joint_positions=state.joint_pos[:count].cpu().numpy(),torques=env.torque[:count].cpu().numpy(),
                        velocities=state.joint_vel[:count].cpu().numpy(),actions=action[:count].cpu().numpy(),
                        observations=observation[:count].cpu().numpy(),tcp_quat_w=state.tcp_quat_w[:count].cpu().numpy(),
                        goal_quat_w=goal_quat.cpu().numpy(),orientation_errors_rad=quat_angle_error(state.tcp_quat_w[:count],goal_quat).cpu().numpy(),
                        contacts=env.contacts[:count].cpu().numpy(),base_up_z=extras['base_up_z'][:count].cpu().numpy(),
                        fall_height=extras['fall_height'][:count].cpu().numpy(),fall_tilt=extras['fall_tilt'][:count].cpu().numpy())
                    step_fallen=(done[:count]&~extras['time_outs'][:count]).cpu().numpy()
                    if append_step(rows,batch_lengths,fallen,k+1,values,step_fallen):
                        break
            for slot,(trajectory,record) in enumerate(zip(batch,rows)):
                result=commanded_metrics(record,batch_lengths[slot],fallen[slot])
                result.update(engine='PhysX',trajectory=trajectory.metadata,policy_sha256=bundle['policy_sha256'],
                    diagnostic=args.diagnostic,trained=bool(bundle['trained']),batch_index=len(layout)-1,env_index=slot,
                    fall_height=bool(record['fall_height'][-1]),fall_tilt=bool(record['fall_tilt'][-1]))
                output=args.output/trajectory.metadata['case_id'];output.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(output/'trace.npz',contact_body_names=np.asarray(env.robot.body_names,dtype=str),
                    **{key:np.asarray(value) for key,value in record.items()})
                (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                results.append(result);print(json.dumps(result),flush=True)
        save_command_run(args.output,args.suite,manifest,bundle,args.seed,'PhysX',results,
            evaluation=dict(num_envs=args.num_envs,batch_layout=layout,wall_elapsed_seconds=time.perf_counter()-started,
                diagnostic=args.diagnostic,provisional_spec=provisional,termination=env.termination,control_dt=.02,
                command_source='preset_trajectory'))
        env.close()
    finally:
        import sys
        launcher.app.close(exit_code=int(sys.exc_info()[0] is not None))


if __name__=='__main__':
    main()
