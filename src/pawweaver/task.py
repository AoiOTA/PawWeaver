"""Simulator-neutral task references, reward terms and fixed evaluation metrics."""
import numpy as np
import torch
from .trajectories import FAMILIES,synthetic,AdaptiveSampler,Trajectory

class GoalBank:
    def __init__(self,batch,steps,device,seed=0,stage=0,adaptive=False,demonstrations=()):
        self.batch,self.steps,self.device=batch,steps,device
        self.rng=np.random.default_rng(seed)
        self.stage,self.adaptive=stage,adaptive
        self.demonstrations=[Trajectory.load(path) for path in demonstrations]
        if any(t.metadata.get("split")!="train" for t in self.demonstrations):
            raise ValueError("Only pre-split training demonstrations can enter the training GoalBank")
        families=FAMILIES+(('fastumi',) if self.demonstrations else ())
        self.sampler=AdaptiveSampler(families)
        self.positions=torch.zeros(batch,steps+5,3,device=device)
        self.family=np.zeros(batch,dtype=int)

    def reset(self,ids,start_w):
        starts=start_w.detach().cpu().numpy()
        for index,start in zip(ids.cpu().tolist(),starts):
            if self.stage==0:
                family=int(self.rng.choice([0,1]))
            else:
                family=int(self.sampler.sample(self.rng,1)[0] if self.adaptive else self.rng.integers(len(self.sampler.families)))
            self.family[index]=family
            if family==len(FAMILIES):
                trajectory=self.demonstrations[int(self.rng.integers(len(self.demonstrations)))].align(start)
                positions=trajectory.sample(np.arange(self.steps+5)*.02)
            elif family==0:
                maximum=(.15,.6,2.,2.)[self.stage]
                delta=self.rng.uniform(-maximum,maximum,3)
                delta[2]=self.rng.uniform(-.08,.08)
                positions=np.repeat((start+delta)[None],self.steps+5,axis=0)
            else:
                amplitude=(.08,.2,.6,.8)[self.stage]
                speed=(.04,.1,.2,.3)[self.stage]
                trajectory=synthetic(FAMILIES[family],int(self.rng.integers(2**31)),
                    duration=(self.steps+4)*.02,amplitude=amplitude,speed=speed)
                positions=trajectory.align(start,float(self.rng.uniform(-np.pi,np.pi))).positions
            self.positions[index]=torch.as_tensor(positions[:self.steps+5],device=self.device,dtype=torch.float32)

    def current(self,indices):
        return self.positions[torch.arange(self.batch,device=self.device),indices.clamp(0,self.steps+4)]

    def future(self,indices):
        steps=indices[:,None]+torch.arange(1,5,device=self.device)
        valid=(steps<=self.steps).unsqueeze(-1).expand(-1,-1,3)
        positions=self.positions[torch.arange(self.batch,device=self.device)[:,None],steps.clamp(max=self.steps+4)]
        return positions,valid

def reward_terms(*,error,previous_error,tcp_velocity,goal_velocity,action,previous_action,
                 torque,effort,q,qd,previous_qd,lower,upper,gravity_b,foot_velocity,foot_contact,
                 collision,fallen,tracking_width=.15):
    error_norm=error.norm(dim=-1)
    return {
        "tracking":torch.exp(-error_norm.square()/tracking_width**2),
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

def episode_metrics(times,errors,base_positions,torques,velocities,fallen,transient_s=2.):
    times,errors=np.asarray(times),np.asarray(errors)
    if times.ndim!=1 or errors.shape!=times.shape or len(times)<1:
        raise ValueError("Metrics require a complete time series")
    selected=errors[times>=transient_s]
    if not len(selected):
        selected=errors
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
        "reached":reach_time is not None and not fallen,"reach_time_s":None if reach_time is None else float(reach_time),
        "fallen":bool(fallen),"absolute_joint_work_j":energy,
        "base_displacement_m":float(np.linalg.norm(np.asarray(base_positions)[-1,:2]-np.asarray(base_positions)[0,:2]))}
