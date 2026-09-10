"""Independent MuJoCo policy runner; no Isaac Lab, RSL-RL or USD dependencies."""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import mujoco
from .assets.build import verify_asset
from .bundle import load_bundle
from .contracts import RobotState,JOINT_NAMES,FOOT_NAMES,GoalSample
from .control import JointPD
from .observations import ObservationBuilder,ObservationSpec
from .trajectories import Trajectory
from .task import episode_metrics,termination_config,fall_causes
from .math import quat_angle_error
from .training_inputs import training_inputs,check_training_identity

class MujocoRunner:
    def observation_spec(self):
        return ObservationSpec()

    def make_observations(self):
        return ObservationBuilder(1,self.pd.default_pos)

    def __init__(self,asset:Path,bundle:Path,*,software_fixture=False,diagnostic=False,provisional_spec:Path|None=None):
        if diagnostic != (provisional_spec is not None):
            raise ValueError("--diagnostic requires --provisional-spec, which is diagnostic-only")
        if software_fixture and diagnostic:
            raise ValueError("Software fixture and diagnostic modes are mutually exclusive")
        self.diagnostic=diagnostic
        self.provisional=None
        if diagnostic:
            self.manifest,expected_spec,self.provisional=training_inputs(asset,diagnostic=True,provisional_spec=provisional_spec)
        else:
            self.manifest=verify_asset(asset)
        if software_fixture and self.manifest["robot"]!="synthetic_software_fixture":
            raise ValueError("Software test mode only accepts the synthetic box fixture")
        self.policy,self.spec,self.bundle=load_bundle(bundle,self.manifest["asset_hash"],require_trained=not (software_fixture or diagnostic),
            observation_spec=self.observation_spec())
        self.termination=termination_config(self.bundle["training_config"])
        if diagnostic:
            check_training_identity(dict(self.bundle.get("training_metadata",{}),asset_hash=self.bundle["asset_hash"]),
                dict(asset_hash=self.manifest["asset_hash"],diagnostic=True,provisional_spec=self.provisional))
            if self.spec.to_dict()!=expected_spec.to_dict():
                raise ValueError("Evaluation actuator spec differs from the policy bundle")
            # Compile passive dynamics before MuJoCo derives inertia/constraint constants.
            # Joint q0 is a reset pose, not a joint reference-angle offset.
            root=ET.parse(asset/"robot.xml").getroot()
            root.find("compiler").set("meshdir",str((asset/"meshes").resolve()))
            height=float(self.bundle["training_config"].get("initial_base_height_m",.5))
            if not np.isfinite(height) or height<=0:
                raise ValueError("Initial base height must be finite and positive")
            root.find(".//body[@name='base_link']").set("pos",f"0 0 {height}")
            for index,name in enumerate(JOINT_NAMES):
                joint=root.find(f".//joint[@name='{name}']")
                for field in ("armature","damping","frictionloss"):
                    joint.set(field,str(getattr(self.spec,field)[index]))
                root.find(f".//motor[@name='{name}_motor']").set("ctrlrange",f"{-self.spec.effort[index]} {self.spec.effort[index]}")
            self.diagnostic_model_xml=ET.tostring(root,encoding="unicode")
            self.model=mujoco.MjModel.from_xml_string(self.diagnostic_model_xml)
        else:
            self.model=mujoco.MjModel.from_xml_path(str(asset/"robot.xml"))
        self.data=mujoco.MjData(self.model)
        self.q_indices=[self.model.joint(name).qposadr[0] for name in JOINT_NAMES]
        self.v_indices=[self.model.joint(name).dofadr[0] for name in JOINT_NAMES]
        self.motor_indices=[self.model.actuator(name+"_motor").id for name in JOINT_NAMES]
        if diagnostic:
            for field in ("armature","damping","frictionloss"):
                if not np.array_equal(getattr(self.model,"dof_"+field)[self.v_indices],getattr(self.spec,field)):
                    raise ValueError(f"Compiled MuJoCo {field} differs from provisional spec")
            if not np.array_equal(self.model.actuator_ctrlrange[self.motor_indices],np.column_stack((-np.array(self.spec.effort),self.spec.effort))):
                raise ValueError("Compiled MuJoCo effort differs from provisional spec")
            if self.model.opt.timestep!=self.spec.physics_dt:
                raise ValueError("MuJoCo timestep differs from policy actuator spec")
        self.pd=JointPD(self.spec,1)
        self.observations=self.make_observations()
        self.action=torch.zeros(1,18)
        self.torque=np.zeros(18)
        self.physics_callback=None
        self.holding=False
        self.reset()

    def tensor(self,array):
        return torch.tensor(np.asarray(array).copy(),dtype=torch.float32).reshape(1,-1)

    def state(self):
        base=self.data.body("base_link")
        rotation=base.xmat.reshape(3,3)
        velocity=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,base.id,velocity,0)
        tcp=self.data.body("tcp")
        return RobotState(joint_pos=self.tensor(self.data.qpos[self.q_indices]),
            joint_vel=self.tensor(self.data.qvel[self.v_indices]),base_pos_w=self.tensor(base.xpos),
            base_quat_w=self.tensor(base.xquat),base_ang_vel_b=self.tensor(rotation.T@velocity[:3]),
            tcp_pos_w=self.tensor(tcp.xpos),tcp_quat_w=self.tensor(tcp.xquat),
            base_lin_vel_b=self.tensor(rotation.T@(velocity[3:]+np.cross(velocity[:3],base.xpos-base.xipos))))

    def reset(self,goal=None,*,goal_quat_w=None):
        if (goal is None)!=(goal_quat_w is None):
            raise ValueError("A goal requires both position and world-frame orientation")
        state=self._reset_physics()
        target=state.tcp_pos_w if goal is None else self.tensor(goal)
        target_quat=state.tcp_quat_w if goal is None else self.tensor(goal_quat_w)
        self.observations.reset(torch.tensor([0]),state,target,torch.zeros(1),goal_quat_w=target_quat)

    def _reset_physics(self):
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.q_indices]=self.spec.default_pos
        mujoco.mj_forward(self.model,self.data)
        self.pd.reset(torch.tensor([0]))
        self.action.zero_()
        self.holding=False
        return self.state()

    @torch.no_grad()
    def step(self,goal=None,*,goal_quat_w=None,measurement:GoalSample|None=None,hold=False):
        if (goal is None)!=(goal_quat_w is None):
            raise ValueError("A goal requires both position and world-frame orientation")
        if goal is not None and measurement is not None:
            raise ValueError("Provide a goal pose or a measured pose, not both")
        state=self.state()
        now=torch.tensor([self.data.time],dtype=torch.float32)
        if goal is not None:
            self.observations.push_goal(self.tensor(goal),now,torch.ones(1,dtype=torch.bool),torch.ones(1),
                orientation_wxyz=self.tensor(goal_quat_w))
        elif measurement is not None:
            self.observations.push_goal(self.tensor(measurement.position),torch.tensor([measurement.timestamp]),
                torch.tensor([measurement.valid]),torch.tensor([measurement.confidence]),
                orientation_wxyz=self.tensor(measurement.orientation_wxyz))
        observation=self.observations.build(state,self.action,now)
        if hold:
            if not self.holding:
                self.action=((state.joint_pos-self.pd.default_pos)/self.pd.action_scale).clamp(-1,1)
        else:
            self.action=self.policy(observation).clamp(-1,1)
        self.holding=hold
        self._apply_action()
        self.observations.push_state(self.state())
        return observation.numpy()[0],self.action.numpy()[0]

    def _apply_action(self):
        """Execute the existing 18-joint PD target over one control tick."""
        self.pd.command(self.action)
        for _ in range(self.spec.decimation):
            torque=self.pd.torque(self.tensor(self.data.qpos[self.q_indices]),self.tensor(self.data.qvel[self.v_indices]))
            self.torque=torque[0].numpy()
            self.data.ctrl[self.motor_indices]=self.torque
            mujoco.mj_step(self.model,self.data)
            if self.physics_callback is not None:
                self.physics_callback(self)

    def _contact_snapshot(self):
        """Net body force magnitudes (N) from the current solver contact snapshot.

        Called after the final mj_step at 50 Hz; this is not the full 2 ms
        contact history. Count only floor pairs whose other body is not a foot.
        """
        forces=np.zeros((self.model.nbody,3))
        floor_id=self.model.geom("floor").id
        nonfoot_ground_contact_count=0
        wrench=np.zeros(6)
        for index in range(self.data.ncon):
            contact=self.data.contact[index]
            body1,body2=self.model.geom_bodyid[[contact.geom1,contact.geom2]]
            mujoco.mj_contactForce(self.model,self.data,index,wrench)
            force_world=contact.frame.reshape(3,3).T@wrench[:3]
            forces[body1]-=force_world
            forces[body2]+=force_world
            if contact.geom1==floor_id or contact.geom2==floor_id:
                other_body=body2 if contact.geom1==floor_id else body1
                if self.model.body(other_body).name not in FOOT_NAMES:
                    nonfoot_ground_contact_count+=1
        return np.linalg.norm(forces,axis=1),nonfoot_ground_contact_count

    def evaluate(self,trajectory:Trajectory,output:Path):
        start_time=float(trajectory.timestamps[0])
        steps=round((trajectory.timestamps[-1]-start_time)/.02)
        if steps<1:
            raise ValueError("Evaluation requires a case lasting at least one control step")
        self.reset(trajectory.positions[0],goal_quat_w=trajectory.sample_orientation(trajectory.timestamps[0]))
        rows={key:[] for key in ("times","errors","base","tcp","goal","torques","velocities","actions","observations",
            "tcp_quat_w","goal_quat_w","orientation_errors_rad","contacts","nonfoot_ground_contact_count",
            "base_up_z","fall_height","fall_tilt")}
        contact_body_names=np.asarray([self.model.body(index).name for index in range(self.model.nbody)],dtype=str)
        fallen=False
        for _ in range(steps):
            goal=trajectory.sample(start_time+self.data.time)
            obs,action=self.step(goal,goal_quat_w=trajectory.sample_orientation(start_time+self.data.time))
            state=self.state()
            # Score the target at the same time as the resulting state.
            target=trajectory.sample(start_time+self.data.time)
            target_quat=trajectory.sample_orientation(start_time+self.data.time)
            error=np.linalg.norm(state.tcp_pos_w.numpy()[0]-target)
            orientation_error=float(quat_angle_error(state.tcp_quat_w,self.tensor(target_quat))[0])
            contacts,nonfoot_ground_contact_count=self._contact_snapshot()
            base=self.data.body("base_link")
            base_up_z=base.xmat.reshape(3,3)[2,2]
            fall_height,fall_tilt=fall_causes(base.xpos[2],base_up_z,self.termination)
            values=(self.data.time,error,state.base_pos_w.numpy()[0],state.tcp_pos_w.numpy()[0],target,
                    self.torque.copy(),self.data.qvel[self.v_indices].copy(),action,obs,
                    state.tcp_quat_w.numpy()[0],target_quat,orientation_error,contacts,nonfoot_ground_contact_count,
                    base_up_z,fall_height,fall_tilt)
            for key,value in zip(rows,values):
                rows[key].append(value)
            fallen=fall_height|fall_tilt
            if fallen:
                break
        result=episode_metrics(rows["times"],rows["errors"],rows["base"],rows["torques"],rows["velocities"],fallen,
            orientation_errors=rows["orientation_errors_rad"])
        from .evaluation import completion_status
        result.update(completion_status(steps,len(rows["times"]),fallen))
        result.update(engine="MuJoCo",trajectory=trajectory.metadata,
            policy_sha256=self.bundle["policy_sha256"],diagnostic=self.diagnostic,
            trained=bool(self.bundle["trained"]),elapsed_seconds=float(self.data.time),
            fall_height=bool(fall_height),fall_tilt=bool(fall_tilt))
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/"trace.npz",contact_body_names=contact_body_names,
            **{key:np.asarray(value) for key,value in rows.items()})
        (output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
        return result

class CommandedMujocoRunner(MujocoRunner):
    """Separate 279-input route; reuse passive physics and the same 18-joint PD."""
    def observation_spec(self):
        from .observations import CommandObservationSpec
        return CommandObservationSpec()

    def make_observations(self):
        from .observations import CommandObservationBuilder
        if self.bundle['training_config'].get('task_mode')!='velocity_ee_pose':
            raise ValueError('Commanded runtime requires velocity_ee_pose training config')
        self.ground_z=torch.zeros(1)
        return CommandObservationBuilder(1,self.pd.default_pos)

    def reset(self,goal_t=None,*,goal_quat_t=None):
        from .commanded_pose import yaw_quaternion
        from .math import quat_apply_inverse,quat_conjugate,quat_mul
        if (goal_t is None)!=(goal_quat_t is None):
            raise ValueError('Commanded goal requires both task-frame position and orientation')
        state=self._reset_physics()
        if goal_t is None:
            origin=state.base_pos_w.clone();origin[:,2]=self.ground_z
            yaw=yaw_quaternion(state.base_quat_w)
            target=quat_apply_inverse(yaw,state.tcp_pos_w-origin)
            target_quat=quat_mul(quat_conjugate(yaw),state.tcp_quat_w)
        else:
            target=self.tensor(goal_t);target_quat=self.tensor(goal_quat_t)
        self.observations.reset(torch.tensor([0]),state,target,torch.zeros(1),goal_quat_t=target_quat)

    @torch.no_grad()
    def step(self,goal_t,*,goal_quat_t,velocity_command):
        state=self.state()
        now=torch.tensor([self.data.time],dtype=torch.float32)
        self.observations.push_goal(self.tensor(goal_t),now,torch.ones(1,dtype=torch.bool),torch.ones(1),
            orientation_t=self.tensor(goal_quat_t))
        observation=self.observations.build(state,self.action,now,
            velocity_command=self.tensor(velocity_command),ground_z=self.ground_z)
        self.action=self.policy(observation).clamp(-1,1)
        self._apply_action()
        self.observations.push_state(self.state())
        return observation.numpy()[0],self.action.numpy()[0]

    def evaluate(self,trajectory,output:Path):
        from .commanded_pose import CommandedPoseTrajectory,task_pose_to_world,yaw_linear_velocity,yaw_rate
        from .commanded_evaluation import commanded_metrics
        if not isinstance(trajectory,CommandedPoseTrajectory):
            raise ValueError('Commanded runtime requires an explicit commanded trajectory')
        start=float(trajectory.timestamps[0])
        steps=round((trajectory.timestamps[-1]-start)/.02)
        if steps<1:
            raise ValueError('Evaluation requires at least one control step')
        self.reset(trajectory.sample(start),goal_quat_t=trajectory.sample_orientation(start))
        keys=('times','errors','base','base_quat_w','tcp','goal','goal_task','goal_quat_task',
            'velocity_commands_yaw','base_velocity_yaw','base_yaw_rate','joint_positions',
            'torques','velocities','actions','observations','tcp_quat_w','goal_quat_w',
            'orientation_errors_rad','contacts','nonfoot_ground_contact_count','base_up_z','fall_height','fall_tilt')
        rows={key:[] for key in keys}
        fallen=False
        for k in range(steps):
            # The same task-frame sample drives pre-state control and post-state scoring.
            sample_time=start+k*.02
            target_t=trajectory.sample(sample_time);quat_t=trajectory.sample_orientation(sample_time)
            command=trajectory.sample_command(sample_time)
            pre=self.state()
            observation,action=self.step(target_t,goal_quat_t=quat_t,velocity_command=command)
            state=self.state()
            goal,goal_quat=task_pose_to_world(self.tensor(target_t),self.tensor(quat_t),
                state.base_pos_w,state.base_quat_w,self.ground_z)
            linear=yaw_linear_velocity(state.base_lin_vel_b,state.base_quat_w)
            rate=yaw_rate(pre.base_quat_w,state.base_quat_w,.02)
            contacts,pairs=self._contact_snapshot()
            up=self.data.body('base_link').xmat.reshape(3,3)[2,2]
            fall_height,fall_tilt=fall_causes(float(state.base_pos_w[0,2]),up,self.termination)
            values=dict(times=(k+1)*.02,errors=float((state.tcp_pos_w-goal).norm(dim=-1)[0]),
                base=state.base_pos_w.numpy()[0],base_quat_w=state.base_quat_w.numpy()[0],
                tcp=state.tcp_pos_w.numpy()[0],goal=goal.numpy()[0],goal_task=target_t,goal_quat_task=quat_t,
                velocity_commands_yaw=command,base_velocity_yaw=linear.numpy()[0],base_yaw_rate=float(rate[0]),
                joint_positions=state.joint_pos.numpy()[0],torques=self.torque.copy(),
                velocities=state.joint_vel.numpy()[0],actions=action,observations=observation,
                tcp_quat_w=state.tcp_quat_w.numpy()[0],goal_quat_w=goal_quat.numpy()[0],
                orientation_errors_rad=float(quat_angle_error(state.tcp_quat_w,goal_quat)[0]),
                contacts=contacts,nonfoot_ground_contact_count=pairs,base_up_z=up,
                fall_height=bool(fall_height),fall_tilt=bool(fall_tilt))
            for key,value in values.items():
                rows[key].append(value)
            fallen=bool(fall_height or fall_tilt)
            if fallen:
                break
        result=commanded_metrics(rows,steps,fallen)
        result.update(engine='MuJoCo',trajectory=trajectory.metadata,policy_sha256=self.bundle['policy_sha256'],
            diagnostic=self.diagnostic,trained=bool(self.bundle['trained']),
            fall_height=bool(fall_height),fall_tilt=bool(fall_tilt))
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'trace.npz',contact_body_names=np.asarray(
            [self.model.body(i).name for i in range(self.model.nbody)],dtype=str),
            **{key:np.asarray(value) for key,value in rows.items()})
        (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset",type=Path,required=True)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--trajectory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--diagnostic",action="store_true",help="Provisional engineering evaluation only")
    parser.add_argument("--provisional-spec",type=Path,help="Required sourced actuator spec for diagnostic mode")
    args=parser.parse_args()
    torch.set_num_threads(1)
    runner=MujocoRunner(args.asset,args.bundle,diagnostic=args.diagnostic,provisional_spec=args.provisional_spec)
    print(json.dumps(runner.evaluate(Trajectory.load(args.trajectory),args.output),indent=2))

if __name__=="__main__":
    main()
