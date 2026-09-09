"""Goal-only, 18-action vector task on Isaac Lab 3 + PhysX. Hardware gates are mandatory."""
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .assets.build import verify_asset
from .assets.model import RobotTree,numbers
from .contracts import ActuatorSpec,RobotState,JOINT_NAMES,FOOT_NAMES,named_indices
from .control import JointPD
from .observations import ObservationBuilder
from .math import quat_apply,quat_apply_inverse
from .task import GoalBank,reward_terms
from .isaac_robot import create_scene

class WholeBodyEnv:
    def __init__(self,asset:Path,config:dict,num_envs=1024,device="cuda:0",seed=0):
        self.manifest=verify_asset(asset)
        self.spec=ActuatorSpec.load(asset/"actuators.json")
        conversion=json.loads((asset/"usd/conversion.json").read_text())
        if conversion["source_asset_hash"]!=self.manifest["asset_hash"]:
            raise ValueError("USD was converted from a different canonical asset")
        self.config,self.device,self.num_envs=config,device,num_envs
        self.num_actions=18
        self.max_episode_length=round(config["episode_seconds"]/.02)
        torch.manual_seed(seed)
        self.sim,self.scene=create_scene(asset,num_envs,device,self.spec,contacts=True)
        self.robot=self.scene["robot"]
        self.joint_ids=named_indices(self.robot.joint_names,JOINT_NAMES)
        self.foot_ids=named_indices(self.robot.body_names,FOOT_NAMES)
        self.gripper_id=self.robot.body_names.index("arm_gripper_base")
        tree=RobotTree.load(asset/"robot.urdf")
        self.tcp_offset=torch.tensor(numbers(tree.joints["tcp_mount"].find("origin").get("xyz")),device=device,dtype=torch.float32)
        self.pd=JointPD(self.spec,num_envs,device)
        self.observations=ObservationBuilder(num_envs,self.pd.default_pos)
        self.episode_length_buf=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.reference=GoalBank(num_envs,self.max_episode_length,device,seed,
            config.get("curriculum_stage",0),config["adaptive_sampling"],config.get("demonstrations",()))
        self.previous_action=torch.zeros(num_envs,18,device=device)
        self.previous_qd=torch.zeros_like(self.previous_action)
        self.previous_tcp=torch.zeros(num_envs,3,device=device)
        self.previous_error=torch.zeros(num_envs,device=device)
        self.episode_error=torch.zeros(num_envs,device=device)
        self.torque=torch.zeros_like(self.previous_action)
        self.contacts=torch.zeros(num_envs,len(self.robot.body_names),device=device)
        self.nominal_mass=self.robot.data.body_mass.torch.clone()
        self.nominal_inertia=self.robot.data.body_inertia.torch.clone()
        self.nominal_com=self.robot.data.body_com_pose_b.torch.clone()
        self.mass_factor=torch.ones(num_envs,1,device=device)
        self.com_offset=torch.zeros(num_envs,3,device=device)
        self.friction=torch.full((num_envs,1),.8,device=device)
        self.materials=self.robot.root_view.get_material_properties().numpy().copy()
        self.materials[:,:,:2]=.8
        self.materials[:,:,2]=0.
        self.set_materials(torch.arange(num_envs,device=device))
        self.reset(torch.arange(num_envs,device=device))

    def state(self):
        data=self.robot.data
        # Isaac Lab 3 Warp data is XYZW; the portable contract is WXYZ.
        pose=data.root_link_pose_w.torch
        gripper=data.body_link_pose_w.torch[:,self.gripper_id]
        tcp=gripper[:,:3]+quat_apply(gripper[:,[6,3,4,5]],self.tcp_offset.expand(self.num_envs,-1))
        return RobotState(data.joint_pos.torch[:,self.joint_ids],data.joint_vel.torch[:,self.joint_ids],
            pose[:,:3],pose[:,[6,3,4,5]],data.root_link_ang_vel_b.torch,tcp,data.root_link_lin_vel_b.torch)

    def reset(self,ids):
        if not len(ids):
            return
        root=self.robot.data.default_root_pose.torch[ids].clone()
        root[:,:3]+=self.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root,env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids),6,device=self.device),env_ids=ids)
        q=self.robot.data.default_joint_pos.torch[ids].clone()
        self.robot.write_joint_position_to_sim_index(position=q,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q),env_ids=ids)
        self.scene.reset(ids)
        self.pd.reset(ids)
        self.episode_length_buf[ids]=0
        self.previous_action[ids]=0
        self.previous_qd[ids]=0
        self.episode_error[ids]=0
        if self.config["domain_randomization"]:
            self.randomize(ids)
        self.sim.forward()
        self.scene.update(.002)
        state=self.state()
        self.reference.reset(ids,state.tcp_pos_w[ids])
        goal=self.reference.current(self.episode_length_buf)
        self.observations.reset(ids,state,goal,self.episode_length_buf.float()*.02)
        self.previous_tcp[ids]=state.tcp_pos_w[ids]
        self.previous_error[ids]=(goal-state.tcp_pos_w).norm(dim=-1)[ids]

    def randomize(self,ids):
        cfg=self.config["randomization"]
        def uniform(bounds,shape):
            return torch.empty(shape,device=self.device).uniform_(*bounds)
        self.pd.strength[ids]=uniform(cfg["motor_strength"],(len(ids),18))
        self.pd.kp_factor[ids]=uniform(cfg["gain_multiplier"],(len(ids),18))
        self.pd.kd_factor[ids]=uniform(cfg["gain_multiplier"],(len(ids),18))
        low,high=cfg["control_delay_physics_steps"]
        if high>=self.pd.capacity:
            raise ValueError("Randomized delay exceeds shared PD queue")
        self.pd.delays[ids]=torch.randint(low,high+1,(len(ids),18),device=self.device)
        factors=uniform(cfg["mass_multiplier"],(len(ids),1))
        self.mass_factor[ids]=factors
        self.robot.set_masses_index(masses=(self.nominal_mass[ids]*factors).contiguous(),env_ids=ids)
        self.robot.set_inertias_index(inertias=(self.nominal_inertia[ids]*factors[:,:,None]).contiguous(),env_ids=ids)
        offset=uniform([-cfg["com_offset_m"],cfg["com_offset_m"]],(len(ids),3))
        self.com_offset[ids]=offset
        com=self.nominal_com[ids].clone()
        base_index=self.robot.body_names.index("base_link")
        com[:,base_index,:3]+=offset
        self.robot.set_coms_index(coms=com.contiguous(),env_ids=ids)
        friction=uniform(cfg["friction"],(len(ids),1))
        self.friction[ids]=friction
        self.materials[ids.cpu().numpy(),:,:2]=friction.cpu().numpy()[:,:,None]
        self.set_materials(ids)

    def set_materials(self,ids):
        import warp as wp
        self.robot.root_view.set_material_properties(wp.from_numpy(self.materials,dtype=wp.float32,device="cpu"),
            wp.from_numpy(ids.cpu().numpy().astype(np.uint32),dtype=wp.uint32,device="cpu"))

    def get_observations(self):
        state=self.state()
        now=self.episode_length_buf.float()*.02
        policy=self.observations.build(state,self.previous_action,now)
        if self.config["domain_randomization"]:
            policy=policy.clone()
            policy[:,:210]+=torch.randn_like(policy[:,:210])*self.config["randomization"]["observation_noise"]
        future,valid=self.reference.future(self.episode_length_buf)
        current=self.reference.current(self.episode_length_buf)
        # Predict displacements in CURRENT base orientation, never future moving base coordinates.
        label=quat_apply_inverse(state.base_quat_w[:,None,:].expand(-1,4,-1),future-current[:,None,:]).flatten(1)
        privileged=torch.cat((policy,state.base_lin_vel_b,self.contacts/100.,self.mass_factor,self.com_offset,self.friction,
            self.pd.strength,self.pd.kp_factor,self.pd.kd_factor,self.pd.delays.float()/10.),dim=-1)
        return TensorDict({"policy":policy,"critic":privileged,"velocity_label":state.base_lin_vel_b.clone(),
            "future_label":label,"future_valid":valid.flatten(1).float()},batch_size=[self.num_envs])

    def step(self,actions,*,auto_reset=True):
        self.pd.command(actions)
        if self.config["domain_randomization"]:
            ids=((self.episode_length_buf>0)&(self.episode_length_buf%200==0)).nonzero(as_tuple=False).flatten()
            if len(ids):
                velocity=self.robot.data.root_link_vel_w.torch[ids].clone()
                bound=self.config["randomization"]["push_velocity_m_s"]
                velocity[:,:2]+=torch.empty(len(ids),2,device=self.device).uniform_(-bound,bound)
                self.robot.write_root_link_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
        for _ in range(self.spec.decimation):
            data=self.robot.data
            self.torque=self.pd.torque(data.joint_pos.torch[:,self.joint_ids],data.joint_vel.torch[:,self.joint_ids])
            self.robot.set_joint_effort_target_index(target=self.torque.contiguous(),joint_ids=self.joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(.002)
        self.episode_length_buf+=1
        state=self.state()
        goal=self.reference.current(self.episode_length_buf)
        last_goal=self.reference.current(self.episode_length_buf-1)
        self.observations.push_state(state)
        self.observations.push_goal(goal,self.episode_length_buf.float()*.02,
            torch.ones(self.num_envs,device=self.device,dtype=torch.bool),torch.ones(self.num_envs,device=self.device))
        for i,name in enumerate(self.robot.body_names):
            key="contact_"+name
            if key in self.scene.sensors:
                self.contacts[:,i]=self.scene[key].data.net_forces_w.torch[:,0].norm(dim=-1)
        down=torch.zeros_like(state.base_pos_w); down[:,2]=-1.
        gravity=quat_apply_inverse(state.base_quat_w,down)
        fallen=(state.base_pos_w[:,2]-self.scene.env_origins[:,2]<.2)|(gravity[:,2]>-.35)
        nonfeet=[i for i in range(len(self.robot.body_names)) if i not in self.foot_ids]
        collision=(self.contacts[:,nonfeet]>5.).any(dim=-1)
        error=goal-state.tcp_pos_w
        terms=reward_terms(error=error,previous_error=self.previous_error,
            tcp_velocity=(state.tcp_pos_w-self.previous_tcp)/.02,goal_velocity=(goal-last_goal)/.02,
            action=self.pd.last_action,previous_action=self.previous_action,torque=self.torque,
            effort=self.pd.effort,q=state.joint_pos,qd=state.joint_vel,previous_qd=self.previous_qd,
            lower=self.pd.lower,upper=self.pd.upper,gravity_b=gravity,
            foot_velocity=self.robot.data.body_link_lin_vel_w.torch[:,self.foot_ids],
            foot_contact=self.contacts[:,self.foot_ids]>1.,collision=collision,fallen=fallen)
        reward=sum(self.config["reward_weights"][name]*value for name,value in terms.items() if name!="termination")*.02
        reward+=self.config["reward_weights"]["termination"]*fallen
        timeout=self.episode_length_buf>=self.max_episode_length
        done=fallen|timeout
        self.previous_action.copy_(self.pd.last_action)
        self.previous_tcp.copy_(state.tcp_pos_w)
        self.previous_qd.copy_(state.joint_vel)
        self.previous_error.copy_(error.norm(dim=-1))
        self.episode_error+=self.previous_error
        ids=done.nonzero(as_tuple=False).flatten()
        for index in ids.cpu().tolist():
            family=self.reference.sampler.families[self.reference.family[index]]
            score=float(self.episode_error[index]/self.episode_length_buf[index])/.08+float(fallen[index])
            self.reference.sampler.update(family,score)
        extras={"time_outs":timeout&~fallen,"tracking_error_m":self.previous_error.mean().item(),
                "fall_fraction":fallen.float().mean().item()}
        if auto_reset:
            self.reset(ids)
        return self.get_observations(),reward,done,extras

    def close(self):
        self.sim.stop()
