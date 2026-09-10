"""Goal-only, 18-action vector task on Isaac Lab 3 + PhysX. Hardware gates are mandatory."""
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .assets.model import RobotTree,numbers
from .contracts import RobotState,JOINT_NAMES,FOOT_NAMES,named_indices
from .control import JointPD
from .observations import ObservationBuilder
from .math import quat_apply,quat_apply_inverse,quat_mul,quat_angle_error,rpy_quat
from .task import GoalBank,reward_terms,sum_reward_terms,UmiPoseReward,termination_config,fall_causes
from .training_inputs import training_inputs

def _foot_sphere_radii_m(tree):
    """The optional height cost uses link origins only for centered spheres."""
    radii=[]
    for name in FOOT_NAMES:
        collisions=tree.links[name].findall("collision")
        if len(collisions)!=1 or collisions[0].find("geometry/sphere") is None:
            raise ValueError(f"foot_above_thigh requires one collision sphere for {name}")
        collision=collisions[0]
        pose=collision.find("origin")
        if pose is not None and any(float(v)!=0 for key in ("xyz","rpy")
                                    for v in pose.get(key,"0 0 0").split()):
            raise ValueError(f"foot_above_thigh requires zero collision origin for {name}")
        radius=float(collision.find("geometry/sphere").get("radius"))
        if not np.isfinite(radius) or radius<=0:
            raise ValueError(f"foot_above_thigh requires a finite positive sphere radius for {name}")
        radii.append(radius)
    return radii

class WholeBodyEnv:
    def __init__(self,asset:Path,config:dict,num_envs=1024,device="cuda:0",seed=0,*,diagnostic=False,provisional_spec=None):
        from .isaac_robot import create_scene
        self.manifest,self.spec,_=training_inputs(asset,diagnostic=diagnostic,provisional_spec=provisional_spec)
        self.diagnostic=diagnostic
        conversion=json.loads((asset/"usd/conversion.json").read_text())
        if conversion["source_asset_hash"]!=self.manifest["asset_hash"]:
            raise ValueError("USD was converted from a different canonical asset")
        self.config,self.device,self.num_envs=config,device,num_envs
        self.termination=termination_config(config)
        self.umi_pose_reward=UmiPoseReward(device) if config.get("umi_pose_reward",False) else None
        self.num_actions=18
        self.max_episode_length=round(config["episode_seconds"]/.02)
        torch.manual_seed(seed)
        self.sim,self.scene=create_scene(asset,num_envs,device,self.spec,contacts=True,
            initial_base_height_m=config.get("initial_base_height_m",.5))
        self.robot=self.scene["robot"]
        if diagnostic:
            for name,sensor in self.scene.sensors.items():
                if name.startswith("contact_"):
                    if sensor.num_sensors!=1 or sensor.body_names!=[name.removeprefix("contact_")]:
                        raise ValueError(f"Contact sensor maps unexpected bodies: {name} {sensor.body_names}")
        self.joint_ids=named_indices(self.robot.joint_names,JOINT_NAMES)
        if diagnostic:
            self.loaded_armature=self.robot.root_view.get_dof_armatures().numpy()[:,self.joint_ids]
            if not np.allclose(self.loaded_armature,np.asarray(self.spec.armature)[None,:],rtol=1e-6,atol=1e-8):
                raise ValueError("PhysX named armature differs from the supplied diagnostic spec")
        self.foot_ids=named_indices(self.robot.body_names,FOOT_NAMES)
        self.gripper_id=self.robot.body_names.index("arm_gripper_base")
        tree=RobotTree.load(asset/"robot.urdf")
        if any(config.get("reward_weights",{}).get(name,0.)!=0
               for name in ("feet_under_hips","foot_above_thigh")):
            self.thigh_ids=named_indices(self.robot.body_names,tuple(name.replace("_foot","_thigh") for name in FOOT_NAMES))
        if config.get("reward_weights",{}).get("foot_above_thigh",0.)!=0:
            self.foot_sphere_radii_m=torch.tensor(_foot_sphere_radii_m(tree),device=device,dtype=torch.float32)
        self.tcp_offset=torch.tensor(numbers(tree.joints["tcp_mount"].find("origin").get("xyz")),device=device,dtype=torch.float32)
        self.tcp_rotation=torch.tensor(rpy_quat(numbers(tree.joints["tcp_mount"].find("origin").get("rpy","0 0 0"))),
            device=device,dtype=torch.float32)
        self.pd=JointPD(self.spec,num_envs,device)
        self.observations=ObservationBuilder(num_envs,self.pd.default_pos)
        self.episode_length_buf=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.reference=GoalBank(num_envs,self.max_episode_length,device,seed,
            config.get("curriculum_stage",0),config["adaptive_sampling"],config.get("demonstrations",()),
            static_goal_offsets_m=config.get("static_goal_offsets_m"),
            demonstrations_only=config.get("demonstrations_only",False),
            demonstration_group_weights=config.get("demonstration_group_weights"))
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
        gripper_quat=gripper[:,[6,3,4,5]]
        tcp=gripper[:,:3]+quat_apply(gripper_quat,self.tcp_offset.expand(self.num_envs,-1))
        return RobotState(joint_pos=data.joint_pos.torch[:,self.joint_ids],
            joint_vel=data.joint_vel.torch[:,self.joint_ids],base_pos_w=pose[:,:3],
            base_quat_w=pose[:,[6,3,4,5]],base_ang_vel_b=data.root_link_ang_vel_b.torch,
            tcp_pos_w=tcp,tcp_quat_w=quat_mul(gripper_quat,self.tcp_rotation.expand(self.num_envs,-1)),
            base_lin_vel_b=data.root_link_lin_vel_b.torch)

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
        self.reference.reset(ids,state.tcp_pos_w[ids],state.tcp_quat_w[ids])
        goal=self.reference.current(self.episode_length_buf)
        self.observations.reset(ids,state,goal,self.episode_length_buf.float()*.02,
            goal_quat_w=self.reference.current_orientation(self.episode_length_buf))
        self.previous_tcp[ids]=state.tcp_pos_w[ids]
        self.previous_error[ids]=(goal-state.tcp_pos_w).norm(dim=-1)[ids]
        if self.umi_pose_reward is not None:
            self.umi_pose_reward.on_reset()

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
        saturated=0
        for _ in range(self.spec.decimation):
            data=self.robot.data
            self.torque=self.pd.torque(data.joint_pos.torch[:,self.joint_ids],data.joint_vel.torch[:,self.joint_ids])
            if self.diagnostic:
                if not torch.isfinite(self.torque).all() or not torch.isfinite(self.pd.raw_torque).all():
                    raise FloatingPointError("Non-finite physics torque")
                saturated+=int((self.pd.raw_torque.abs()>=self.pd.effort).sum().item())
            self.robot.set_joint_effort_target_index(target=self.torque.contiguous(),joint_ids=self.joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(.002)
            if self.diagnostic:
                for value in vars(self.state()).values():
                    if value is not None and not torch.isfinite(value).all():
                        raise FloatingPointError("Non-finite physics state")
        self.episode_length_buf+=1
        state=self.state()
        goal=self.reference.current(self.episode_length_buf)
        last_goal=self.reference.current(self.episode_length_buf-1)
        self.observations.push_state(state)
        self.observations.push_goal(goal,self.episode_length_buf.float()*.02,
            torch.ones(self.num_envs,device=self.device,dtype=torch.bool),torch.ones(self.num_envs,device=self.device),
            orientation_wxyz=self.reference.current_orientation(self.episode_length_buf))
        for i,name in enumerate(self.robot.body_names):
            key="contact_"+name
            if key in self.scene.sensors:
                self.contacts[:,i]=self.scene[key].data.net_forces_w.torch[:,0].norm(dim=-1)
        down=torch.zeros_like(state.base_pos_w); down[:,2]=-1.
        gravity=quat_apply_inverse(state.base_quat_w,down)
        base_up_z=-gravity[:,2]
        fall_height,fall_tilt=fall_causes(state.base_pos_w[:,2]-self.scene.env_origins[:,2],base_up_z,self.termination)
        fallen=fall_height|fall_tilt
        nonfeet=[i for i in range(len(self.robot.body_names)) if i not in self.foot_ids]
        collision=(self.contacts[:,nonfeet]>5.).any(dim=-1)
        error=goal-state.tcp_pos_w
        orientation_error=quat_angle_error(state.tcp_quat_w,self.reference.current_orientation(self.episode_length_buf))
        terms=reward_terms(error=error,orientation_error=orientation_error,previous_error=self.previous_error,
            tracking_width=self.config.get("tracking_width_m",.15),
            orientation_tracking_width_rad=self.config.get("orientation_tracking_width_rad",.5),
            joint_limit_margin_fraction=self.config.get("joint_limit_margin_fraction"),
            # Flat-ground mapping: existing sensors explicitly supply world-frame force.
            foot_force_z=(torch.stack([self.scene["contact_"+name].data.net_forces_w.torch[:,0,2]
                for name in FOOT_NAMES],dim=-1)
                if self.config.get("reward_weights",{}).get("even_mass_distribution",0.)!=0 else None),
            foot_hip_delta_xy=(self.robot.data.body_link_pose_w.torch[:,self.foot_ids,:2]
                -self.robot.data.body_link_pose_w.torch[:,self.thigh_ids,:2]
                if self.config.get("reward_weights",{}).get("feet_under_hips",0.)!=0 else None),
            feet_under_hips_distance_sigma=self.config.get("feet_under_hips_distance_sigma",.5),
            foot_bottom_minus_thigh_z=(self.robot.data.body_link_pose_w.torch[:,self.foot_ids,2]
                -self.foot_sphere_radii_m-self.robot.data.body_link_pose_w.torch[:,self.thigh_ids,2]
                if self.config.get("reward_weights",{}).get("foot_above_thigh",0.)!=0 else None),
            tcp_velocity=(state.tcp_pos_w-self.previous_tcp)/.02,goal_velocity=(goal-last_goal)/.02,
            action=self.pd.last_action,previous_action=self.previous_action,torque=self.torque,
            effort=self.pd.effort,q=state.joint_pos,qd=state.joint_vel,previous_qd=self.previous_qd,
            lower=self.pd.lower,upper=self.pd.upper,gravity_b=gravity,
            foot_velocity=self.robot.data.body_link_lin_vel_w.torch[:,self.foot_ids],
            foot_contact=self.contacts[:,self.foot_ids]>1.,collision=collision,fallen=fallen)
        if self.config.get("umi_pose_reward",False):
            reward_diagnostics=self.config.get("reward_diagnostics",False)
            result=self.umi_pose_reward.nonterminal_reward(terms,self.config["reward_weights"],
                error.norm(dim=-1),orientation_error,return_terms=reward_diagnostics)
            if reward_diagnostics:
                nonterminal,weighted_terms=result
            else:
                nonterminal=result
            clip_nonnegative=self.config.get("umi_clip_nonnegative",True)
            postclip=nonterminal.clamp_min(0) if clip_nonnegative else nonterminal
            reward=postclip*.02
            self.umi_pose_reward.update(error.norm(dim=-1),orientation_error)
        else:
            reward=sum_reward_terms(terms,self.config["reward_weights"],
                coupled_pose=self.config.get("coupled_pose_reward",False))*.02
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
                "orientation_error_rad":orientation_error.mean().item(),"fall_fraction":fallen.float().mean().item(),
                "fall_height":fall_height,"fall_tilt":fall_tilt,"base_up_z":base_up_z}
        if self.config.get("umi_pose_reward",False):
            extras.update(umi_nonterminal_clipped_fraction=(nonterminal<0).float().mean().item() if clip_nonnegative else 0.,
                umi_nonterminal_preclip_mean=nonterminal.mean().item())
            if reward_diagnostics:
                # Current-step tensors only: the collector owns any aggregation.
                # All values precede the .02 control dt and the fall penalty.
                extras["reward_diagnostics"]={"weighted_nonterminal_terms":weighted_terms,
                    "nonterminal_preclip":nonterminal,"nonterminal_postclip":postclip}
        if self.config.get("demonstration_group_weights") is not None:
            # Capture the task identity before auto-reset selects another file.
            group_stats={}
            selected=self.reference.demonstration_index
            for group in self.config["demonstration_group_weights"]:
                members=[i for i,t in enumerate(self.reference.demonstrations)
                         if t.metadata["training_group"]==group]
                mask=torch.as_tensor((selected>=0)&np.isin(selected,members),device=self.device)
                ended=mask&done
                row={"transitions":int(mask.sum()),
                     "position_error_sum_m":float(self.previous_error[mask].sum()),
                     "orientation_error_sum_rad":float(orientation_error[mask].sum()),
                     "falls":int((mask&fallen).sum()),
                     "resets":int(ended.sum()) if auto_reset else 0,
                     "timeouts":int((mask&timeout&~fallen).sum()),
                     "ended_episodes":int(ended.sum()),
                     "ended_episode_seconds_sum":float(self.episode_length_buf[ended].sum())*.02}
                if self.umi_pose_reward is not None:
                    row.update(nonterminal_preclip_sum=float(nonterminal[mask].sum()),
                               nonterminal_postclip_sum=float(postclip[mask].sum()))
                group_stats[group]=row
            extras["demonstration_group_stats"]=group_stats
        if self.diagnostic:
            # One boolean per environment at 50 Hz; not substep contact pairs.
            extras.update(falls=int(fallen.sum().item()),resets=len(ids) if auto_reset else 0,
                collision_control_samples=int(collision.sum().item()),
                torque_saturated=saturated,torque_samples=self.num_envs*18*self.spec.decimation,
                tracking_error_max_m=float(error.norm(dim=-1).max()),
                orientation_error_max_rad=float(orientation_error.max()),
                base_height_min_m=float((state.base_pos_w[:,2]-self.scene.env_origins[:,2]).min()))
        if auto_reset:
            self.reset(ids)
        return self.get_observations(),reward,done,extras

    def close(self):
        self.sim.stop()
