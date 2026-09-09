"""Independent MuJoCo policy runner; no Isaac Lab, RSL-RL, USD or training imports."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from .assets.build import verify_asset
from .bundle import load_bundle
from .contracts import RobotState,JOINT_NAMES,GoalSample
from .control import JointPD
from .observations import ObservationBuilder
from .trajectories import Trajectory
from .task import episode_metrics

class MujocoRunner:
    def __init__(self,asset:Path,bundle:Path,*,software_fixture=False):
        self.manifest=verify_asset(asset)
        if software_fixture and self.manifest["robot"]!="synthetic_software_fixture":
            raise ValueError("Software test mode only accepts the synthetic box fixture")
        self.policy,self.spec,self.bundle=load_bundle(bundle,self.manifest["asset_hash"],require_trained=not software_fixture)
        self.model=mujoco.MjModel.from_xml_path(str(asset/"robot.xml"))
        self.data=mujoco.MjData(self.model)
        self.q_indices=[self.model.joint(name).qposadr[0] for name in JOINT_NAMES]
        self.v_indices=[self.model.joint(name).dofadr[0] for name in JOINT_NAMES]
        self.motor_indices=[self.model.actuator(name+"_motor").id for name in JOINT_NAMES]
        self.pd=JointPD(self.spec,1)
        self.observations=ObservationBuilder(1,self.pd.default_pos)
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
        return RobotState(self.tensor(self.data.qpos[self.q_indices]),self.tensor(self.data.qvel[self.v_indices]),
            self.tensor(base.xpos),self.tensor(base.xquat),self.tensor(rotation.T@velocity[:3]),
            self.tensor(self.data.body("tcp").xpos),
            self.tensor(rotation.T@(velocity[3:]+np.cross(velocity[:3],base.xpos-base.xipos))))

    def reset(self,goal=None):
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.q_indices]=self.spec.default_pos
        mujoco.mj_forward(self.model,self.data)
        self.pd.reset(torch.tensor([0]))
        self.action.zero_()
        self.holding=False
        state=self.state()
        target=state.tcp_pos_w if goal is None else self.tensor(goal)
        self.observations.reset(torch.tensor([0]),state,target,torch.zeros(1))

    @torch.inference_mode()
    def step(self,goal=None,*,measurement:GoalSample|None=None,hold=False):
        state=self.state()
        now=torch.tensor([self.data.time],dtype=torch.float32)
        if goal is not None:
            self.observations.push_goal(self.tensor(goal),now,torch.ones(1,dtype=torch.bool),torch.ones(1))
        elif measurement is not None:
            self.observations.push_goal(self.tensor(measurement.position),torch.tensor([measurement.timestamp]),
                torch.tensor([measurement.valid]),torch.tensor([measurement.confidence]))
        observation=self.observations.build(state,self.action,now)
        if hold:
            if not self.holding:
                self.action=((state.joint_pos-self.pd.default_pos)/self.pd.action_scale).clamp(-1,1)
        else:
            self.action=self.policy(observation).clamp(-1,1)
        self.holding=hold
        self.pd.command(self.action)
        for _ in range(self.spec.decimation):
            torque=self.pd.torque(self.tensor(self.data.qpos[self.q_indices]),self.tensor(self.data.qvel[self.v_indices]))
            self.torque=torque[0].numpy()
            self.data.ctrl[self.motor_indices]=self.torque
            mujoco.mj_step(self.model,self.data)
            if self.physics_callback is not None:
                self.physics_callback(self)
        self.observations.push_state(self.state())
        return observation.numpy()[0],self.action.numpy()[0]

    def evaluate(self,trajectory:Trajectory,output:Path):
        self.reset(trajectory.positions[0])
        rows={key:[] for key in ("times","errors","base","tcp","goal","torques","velocities","actions","observations")}
        fallen=False
        steps=int((trajectory.timestamps[-1]-trajectory.timestamps[0])/.02)
        for _ in range(steps):
            goal=trajectory.sample(self.data.time)
            obs,action=self.step(goal)
            state=self.state()
            # Score the target at the same time as the resulting state.
            target=trajectory.sample(self.data.time)
            error=np.linalg.norm(state.tcp_pos_w.numpy()[0]-target)
            values=(self.data.time,error,state.base_pos_w.numpy()[0],state.tcp_pos_w.numpy()[0],target,
                    self.torque.copy(),self.data.qvel[self.v_indices].copy(),action,obs)
            for key,value in zip(rows,values):
                rows[key].append(value)
            base=self.data.body("base_link")
            fallen=base.xpos[2]<.2 or base.xmat.reshape(3,3)[2,2]<.35
            if fallen:
                break
        result=episode_metrics(rows["times"],rows["errors"],rows["base"],rows["torques"],rows["velocities"],fallen)
        result.update(engine="MuJoCo",trajectory=trajectory.metadata,bundle_hash=self.bundle["policy_sha256"])
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/"trace.npz",**{key:np.asarray(value) for key,value in rows.items()})
        (output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset",type=Path,required=True)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--trajectory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    runner=MujocoRunner(args.asset,args.bundle)
    print(json.dumps(runner.evaluate(Trajectory.load(args.trajectory),args.output),indent=2))

if __name__=="__main__":
    main()
