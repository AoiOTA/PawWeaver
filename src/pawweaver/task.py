"""Simulator-neutral task references, reward terms and fixed evaluation metrics."""
import numpy as np
import torch
from .trajectories import FAMILIES,synthetic,AdaptiveSampler,Trajectory

class UmiPoseReward:
    """Opt-in engineering recipe, UMI d75c9c182d8044dadf53043612da2ffbf1936a97.

    Global error EMA at 50 Hz preserves the upstream approximately 16 s time
    scale. Resets select shared widths but never clear the running errors.
    """
    position_schedule=((100.,2.),(1.,1.),(.8,.5),(.5,.1),(.4,.05),(.2,.01),(.1,.005))
    orientation_schedule=((100.,8.),(1.,4.),(.8,2.),(.6,1.),(.2,.5))
    alpha=.00125

    def __init__(self,device="cpu"):
        self.position_ema_m=torch.tensor(1.,device=device,dtype=torch.float64)
        self.orientation_ema_rad=torch.tensor(1.,device=device,dtype=torch.float64)
        self.position_sigma_m2=2.
        self.orientation_sigma_rad=8.

    def update(self,position_error,orientation_error):
        self.position_ema_m.mul_(1-self.alpha).add_(position_error.double().mean(),alpha=self.alpha)
        self.orientation_ema_rad.mul_(1-self.alpha).add_(orientation_error.double().mean(),alpha=self.alpha)

    def on_reset(self):
        for threshold,sigma in self.position_schedule:
            if self.position_ema_m.item()<threshold:
                self.position_sigma_m2=sigma
        for threshold,sigma in self.orientation_schedule:
            if self.orientation_ema_rad.item()<threshold:
                self.orientation_sigma_rad=sigma

    def nonterminal_reward(self,terms,weights,position_error,orientation_error):
        pose=4*torch.exp(-position_error.square()/self.position_sigma_m2)*torch.exp(
            -orientation_error/self.orientation_sigma_rad)
        return pose+sum(weights[name]*value for name,value in terms.items()
                        if name not in ("tracking","orientation_tracking","termination"))

    def state_dict(self):
        return {"position_ema_m":self.position_ema_m.item(),
                "orientation_ema_rad":self.orientation_ema_rad.item(),
                "position_sigma_m2":self.position_sigma_m2,
                "orientation_sigma_rad":self.orientation_sigma_rad}

    def load_state_dict(self,state):
        required=("position_ema_m","orientation_ema_rad","position_sigma_m2","orientation_sigma_rad")
        if not isinstance(state,dict) or any(key not in state for key in required):
            raise ValueError("UMI pose reward resume requires saved EMA and sigma state")
        values={key:float(state[key]) for key in required}
        if any(not np.isfinite(v) or v<0 for v in values.values()):
            raise ValueError("UMI pose reward state must be finite and nonnegative")
        if (values["position_sigma_m2"] not in dict(self.position_schedule).values()
                or values["orientation_sigma_rad"] not in dict(self.orientation_schedule).values()):
            raise ValueError("UMI pose reward state has unknown sigma")
        self.position_ema_m.fill_(values["position_ema_m"])
        self.orientation_ema_rad.fill_(values["orientation_ema_rad"])
        self.position_sigma_m2=values["position_sigma_m2"]
        self.orientation_sigma_rad=values["orientation_sigma_rad"]


class GoalBank:
    def __init__(self,batch,steps,device,seed=0,stage=0,adaptive=False,demonstrations=(),static_goal_offsets_m=None):
        self.batch,self.steps,self.device=batch,steps,device
        self.rng=np.random.default_rng(seed)
        self.stage,self.adaptive=stage,adaptive
        self.static_goal_offsets_m=None
        if static_goal_offsets_m is not None:
            offsets=np.asarray(static_goal_offsets_m,dtype=np.float32)
            if offsets.ndim!=2 or offsets.shape[0]==0 or offsets.shape[1]!=3 or not np.isfinite(offsets).all():
                raise ValueError("Static goal offsets must be a nonempty finite [N,3] array in meters")
            self.static_goal_offsets_m=offsets.copy()
        self.demonstrations=[Trajectory.load(path) for path in demonstrations]
        if any(t.metadata.get("split")!="train" for t in self.demonstrations):
            raise ValueError("Only pre-split training demonstrations can enter the training GoalBank")
        families=FAMILIES+(('fastumi',) if self.demonstrations else ())
        self.sampler=AdaptiveSampler(families)
        self.positions=torch.zeros(batch,steps+5,3,device=device)
        self.orientations_wxyz=torch.zeros(batch,steps+5,4,device=device)
        self.family=np.zeros(batch,dtype=int)

    def reset(self,ids,start_w,start_quat_w):
        starts=start_w.detach().cpu().numpy()
        start_orientations=start_quat_w.detach().cpu().numpy().astype(np.float64)
        if starts.shape!=(len(ids),3) or not np.isfinite(starts).all():
            raise ValueError("Goal reset requires finite start positions for every selected environment")
        if start_orientations.shape!=(len(ids),4) or not np.isfinite(start_orientations).all():
            raise ValueError("Goal reset requires finite WXYZ start orientations for every selected environment")
        scale=np.max(np.abs(start_orientations),axis=-1,keepdims=True)
        if (scale==0).any():
            raise ValueError("Goal reset orientation must be nonzero")
        start_orientations=start_orientations/scale
        start_orientations/=np.linalg.norm(start_orientations,axis=-1,keepdims=True)
        if self.static_goal_offsets_m is not None:
            indices=ids.cpu().numpy()
            targets=starts+self.static_goal_offsets_m[indices%len(self.static_goal_offsets_m)]
            self.positions[ids]=torch.as_tensor(targets,device=self.device,dtype=torch.float32)[:,None,:]
            self.orientations_wxyz[ids]=torch.as_tensor(start_orientations,device=self.device,dtype=torch.float32)[:,None,:]
            self.family[indices]=0
            return
        for index,start,start_orientation in zip(ids.cpu().tolist(),starts,start_orientations):
            if self.stage==0:
                family=int(self.rng.choice([0,1]))
            else:
                family=int(self.sampler.sample(self.rng,1)[0] if self.adaptive else self.rng.integers(len(self.sampler.families)))
            self.family[index]=family
            # Synthetic positions explicitly hold the actual reset TCP world
            # orientation. Combining them does not establish pose reachability.
            orientations=np.repeat(start_orientation[None,:],self.steps+5,axis=0)
            if family==len(FAMILIES):
                trajectory=self.demonstrations[int(self.rng.integers(len(self.demonstrations)))].align(start)
                positions=trajectory.sample(np.arange(self.steps+5)*.02)
                orientations=trajectory.sample_orientation(np.arange(self.steps+5)*.02)
            elif family==0:
                maximum=(.15,.6,2.,2.)[self.stage]
                delta=self.rng.uniform(-maximum,maximum,3)
                delta[2]=self.rng.uniform(-.08,.08)
                positions=np.repeat((start+delta)[None],self.steps+5,axis=0)
            else:
                amplitude=(.08,.2,.6,.8)[self.stage]
                speed=(.04,.1,.2,.3)[self.stage]
                trajectory=synthetic(FAMILIES[family],int(self.rng.integers(2**31)),
                    duration=(self.steps+4)*.02,amplitude=amplitude,speed=speed,
                    orientation_wxyz=start_orientation)
                # Yaw augments only the synthetic position path; its final
                # orientation target is the explicit reset-hold rule above.
                positions=trajectory.align(start,float(self.rng.uniform(-np.pi,np.pi))).positions
            self.positions[index]=torch.as_tensor(positions[:self.steps+5],device=self.device,dtype=torch.float32)
            self.orientations_wxyz[index]=torch.as_tensor(orientations,device=self.device,dtype=torch.float32)

    def current(self,indices):
        return self.positions[torch.arange(self.batch,device=self.device),indices.clamp(0,self.steps+4)]

    def current_orientation(self,indices):
        return self.orientations_wxyz[torch.arange(self.batch,device=self.device),indices.clamp(0,self.steps+4)]

    def future(self,indices):
        steps=indices[:,None]+torch.arange(1,5,device=self.device)
        valid=(steps<=self.steps).unsqueeze(-1).expand(-1,-1,3)
        positions=self.positions[torch.arange(self.batch,device=self.device)[:,None],steps.clamp(max=self.steps+4)]
        return positions,valid

def reward_terms(*,error,orientation_error,previous_error,tcp_velocity,goal_velocity,action,previous_action,
                 torque,effort,q,qd,previous_qd,lower,upper,gravity_b,foot_velocity,foot_contact,
                 collision,fallen,tracking_width=.15,orientation_tracking_width_rad=.5):
    if not np.isfinite(tracking_width) or tracking_width<=0:
        raise ValueError("Position reward width must be finite and positive")
    if not np.isfinite(orientation_tracking_width_rad) or orientation_tracking_width_rad<=0:
        raise ValueError("Orientation reward width must be finite and positive")
    error_norm=error.norm(dim=-1)
    return {
        "tracking":torch.exp(-error_norm.square()/tracking_width**2),
        "orientation_tracking":torch.exp(-orientation_error.square()/orientation_tracking_width_rad**2),
        "progress":(previous_error-error_norm).clamp(-.1,.1)/.02,
        "velocity_tracking":torch.exp(-(tcp_velocity-goal_velocity).square().sum(-1)/.25**2),
        "action_rate":(action-previous_action).square().sum(-1),
        "normalized_torque":(torque/effort).square().sum(-1),
        "joint_acceleration":((qd-previous_qd)/.02).square().sum(-1),
        "joint_limit":((lower+.02-q).clamp_min(0)+(q-upper+.02).clamp_min(0)).sum(-1),
        "saturation":(torque.abs()>=.98*effort).float().mean(-1),
        "foot_slip":(foot_velocity[:,:,:2].square().sum(-1)*foot_contact).sum(-1),
        "body_tilt":gravity_b[:,:2].square().sum(-1),
        "collision":collision.float(),"termination":fallen.float()}

def sum_reward_terms(terms,weights,*,coupled_pose=False):
    """Combine control-rate rewards; termination remains a separate event cost."""
    if not coupled_pose:
        return sum(weights[name]*value for name,value in terms.items() if name!="termination")
    pose=(weights["tracking"]+weights["orientation_tracking"])*terms["tracking"]*terms["orientation_tracking"]
    return pose+sum(weights[name]*value for name,value in terms.items()
                    if name not in ("tracking","orientation_tracking","termination"))


def episode_metrics(times,errors,base_positions,torques,velocities,fallen,transient_s=2.,*,orientation_errors):
    times,errors=np.asarray(times),np.asarray(errors)
    orientation_errors=np.asarray(orientation_errors,dtype=float)
    if times.ndim!=1 or errors.shape!=times.shape or len(times)<1:
        raise ValueError("Metrics require a complete time series")
    if (orientation_errors.shape!=times.shape or not np.isfinite(orientation_errors).all()
            or (orientation_errors<0).any() or (orientation_errors>np.pi+1e-6).any()):
        raise ValueError("Metrics require one finite shortest orientation angle in radians per sample")
    selection=times>=transient_s
    if not selection.any():
        selection=np.ones(len(times),dtype=bool)
    selected=errors[selection]
    selected_orientation=orientation_errors[selection]
    hold_start=None; reach_time=None
    for t,error in zip(times,errors):
        if error<=.05:
            hold_start=t if hold_start is None else hold_start
            if t-hold_start>=1. and hold_start<=10.:
                reach_time=hold_start; break
        else:
            hold_start=None
    energy=float(np.trapezoid(np.abs(np.asarray(torques)*velocities).sum(-1),times))
    return {"rmse_m":float(np.sqrt(np.mean(selected**2))),"p95_m":float(np.quantile(selected,.95)),
        "orientation_rmse_rad":float(np.sqrt(np.mean(selected_orientation**2))),
        "orientation_p95_rad":float(np.quantile(selected_orientation,.95)),
        "reached":reach_time is not None and not fallen,"reach_time_s":None if reach_time is None else float(reach_time),
        "fallen":bool(fallen),"absolute_joint_work_j":energy,
        "base_displacement_m":float(np.linalg.norm(np.asarray(base_positions)[-1,:2]-np.asarray(base_positions)[0,:2]))}
