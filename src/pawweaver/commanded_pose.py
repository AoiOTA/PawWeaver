"""Independent velocity-command + yaw-following task-frame pose references.

T has origin (base world X, base world Y, ground Z) and world-from-T rotation
equal to base yaw only. These references are never world Trajectory objects.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from .math import normalize_quat, quat_apply, quat_apply_inverse, quat_mul
from .trajectories import AdaptiveSampler

COMMAND_SCHEMA_VERSION=1
COMMAND_TASK_KIND="velocity_ee_pose"
COMMAND_TASK_FRAME="base_xy_yaw_ground_z"
COMMAND_UNITS={"time":"s","position":"m","orientation":"wxyz",
               "linear_velocity":"m/s","yaw_rate":"rad/s"}


def _metadata(metadata):
    if not isinstance(metadata,dict):
        raise ValueError("Commanded pose metadata must be a dictionary")
    if "schema_version" in metadata and metadata["schema_version"]!=COMMAND_SCHEMA_VERSION:
        raise ValueError("Commanded pose metadata schema_version differs")
    if metadata.get("task_kind")!=COMMAND_TASK_KIND or metadata.get("task_frame")!=COMMAND_TASK_FRAME:
        raise ValueError("Commanded pose requires explicit velocity_ee_pose task kind and yaw/ground task frame")
    if metadata.get("units")!=COMMAND_UNITS:
        raise ValueError("Commanded pose units must explicitly match seconds/meters/WXYZ/m/s/rad/s")
    if metadata.get("split") not in ("train","test","validation"):
        raise ValueError("Commanded pose requires an explicit train/test/validation split")
    for key in ("case_id","source_id","command_source"):
        if not isinstance(metadata.get(key),str) or not metadata[key].strip():
            raise ValueError(f"Commanded pose requires a nonempty {key}")
    return deepcopy(metadata)


@dataclass
class CommandedPoseTrajectory:
    timestamps: np.ndarray
    positions_task: np.ndarray
    orientations_task_wxyz: np.ndarray
    velocity_commands_yaw: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.metadata=_metadata(self.metadata)
        self.timestamps=np.asarray(self.timestamps,dtype=np.float64)
        self.positions_task=np.asarray(self.positions_task,dtype=np.float64)
        self.orientations_task_wxyz=np.asarray(self.orientations_task_wxyz,dtype=np.float64)
        self.velocity_commands_yaw=np.asarray(self.velocity_commands_yaw,dtype=np.float64)
        if self.timestamps.ndim!=1 or len(self.timestamps)<2:
            raise ValueError("Commanded pose requires at least two timestamps")
        if not np.isfinite(self.timestamps).all() or self.timestamps[0]!=0. or (np.diff(self.timestamps)<=0).any():
            raise ValueError("Commanded pose time must be finite, start at zero and increase strictly")
        n=len(self.timestamps)
        for name,width in (("positions_task",3),("orientations_task_wxyz",4),("velocity_commands_yaw",3)):
            values=getattr(self,name)
            if values.shape!=(n,width) or not np.isfinite(values).all():
                raise ValueError(f"Commanded pose {name} requires finite Nx{width} data")
        scale=np.abs(self.orientations_task_wxyz).max(axis=-1,keepdims=True)
        if (scale==0).any():
            raise ValueError("Commanded pose orientations must be nonzero WXYZ quaternions")
        scaled=self.orientations_task_wxyz/scale
        self.orientations_task_wxyz=scaled/np.linalg.norm(scaled,axis=-1,keepdims=True)

    @staticmethod
    def _times(times):
        query=np.asarray(times,dtype=np.float64)
        if not np.isfinite(query).all():
            raise ValueError("Commanded pose sample times must be finite")
        return query

    def sample(self,times):
        query=self._times(times)
        return np.stack([np.interp(query,self.timestamps,self.positions_task[:,axis]) for axis in range(3)],axis=-1)

    def sample_orientation(self,times):
        query=self._times(times)
        rotations=Rotation.from_quat(self.orientations_task_wxyz[:,[1,2,3,0]])
        sampled=Slerp(self.timestamps,rotations)(np.clip(query.reshape(-1),self.timestamps[0],self.timestamps[-1]))
        return sampled.as_quat()[:,[3,0,1,2]].reshape(query.shape+(4,))

    def sample_command(self,times):
        query=self._times(times)
        return np.stack([np.interp(query,self.timestamps,self.velocity_commands_yaw[:,axis]) for axis in range(3)],axis=-1)

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,schema_version=COMMAND_SCHEMA_VERSION,task_kind=COMMAND_TASK_KIND,
            timestamps=self.timestamps,positions_task=self.positions_task,orientations_task_wxyz=self.orientations_task_wxyz,
            velocity_commands_yaw=self.velocity_commands_yaw,metadata=json.dumps(self.metadata,sort_keys=True,allow_nan=False))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            fields={"schema_version","task_kind","timestamps","positions_task","orientations_task_wxyz","velocity_commands_yaw","metadata"}
            if set(data.files)!=fields:
                raise ValueError("Commanded pose requires its independent schema fields; legacy world trajectories are not accepted")
            schema=np.asarray(data["schema_version"])
            if schema.shape!=() or schema.dtype.kind not in "iu" or schema.item()!=COMMAND_SCHEMA_VERSION:
                raise ValueError("Commanded pose requires schema_version=1")
            kind=np.asarray(data["task_kind"])
            if kind.shape!=() or kind.item()!=COMMAND_TASK_KIND:
                raise ValueError("Commanded pose requires task_kind=velocity_ee_pose")
            if np.asarray(data["metadata"]).shape!=():
                raise ValueError("Commanded pose metadata must be scalar JSON")
            return cls(data["timestamps"],data["positions_task"],data["orientations_task_wxyz"],
                       data["velocity_commands_yaw"],json.loads(str(data["metadata"])))


def load_command_suite(path):
    """Load manifest cases by case_id/path without relabeling split or source."""
    path=Path(path)
    manifest=json.loads((path/"manifest.json").read_text())
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"]!=COMMAND_SCHEMA_VERSION or manifest.get("task_kind")!=COMMAND_TASK_KIND:
        raise ValueError("Command suite requires schema_version=1 and task_kind=velocity_ee_pose")
    if manifest.get("units")!=COMMAND_UNITS:
        raise ValueError("Command suite requires explicit matching units")
    if "task_frame" in manifest and manifest["task_frame"]!=COMMAND_TASK_FRAME:
        raise ValueError("Command suite task frame differs")
    cases=manifest.get("cases")
    if not isinstance(cases,list) or not cases:
        raise ValueError("Command suite requires nonempty cases")
    trajectories=[];ids=set()
    for entry in cases:
        if not isinstance(entry,dict) or any(not isinstance(entry.get(key),str) or not entry[key].strip() for key in ("case_id","path")):
            raise ValueError("Command suite entries require case_id/path strings")
        if entry["case_id"] in ids:
            raise ValueError("Command suite has duplicate case_id")
        ids.add(entry["case_id"])
        source=path/entry["path"]
        if "sha256" in entry and hashlib.sha256(source.read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError(f"Changed command suite input: {source}")
        trajectory=CommandedPoseTrajectory.load(source)
        if trajectory.metadata["case_id"]!=entry["case_id"]:
            raise ValueError("Command suite case_id differs from trajectory identity")
        if "split" in manifest and trajectory.metadata["split"]!=manifest["split"]:
            raise ValueError("Command suite split differs from trajectory identity")
        for key in ("split","source_id"):
            if key in entry and entry[key]!=trajectory.metadata[key]:
                raise ValueError(f"Command suite {key} differs from trajectory identity")
        trajectories.append(trajectory)
    return trajectories


def yaw_angle(quat_wxyz):
    """World-Z Euler yaw of a WXYZ rotation, in radians."""
    w,x,y,z=normalize_quat(quat_wxyz).unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def yaw_quaternion(quat_wxyz):
    angle=yaw_angle(quat_wxyz)*.5
    zero=torch.zeros_like(angle)
    return torch.stack((torch.cos(angle),zero,zero,torch.sin(angle)),dim=-1)


def task_pose_to_world(position_t,quat_t,base_pos_w,base_quat_w,ground_z):
    """Map N poses or NxH history from T with the current base XY/yaw only."""
    if position_t.ndim not in (2,3) or position_t.shape[-1]!=3 or quat_t.shape!=position_t.shape[:-1]+(4,):
        raise ValueError("Task pose requires matching Nx3/Nx4 or NxHx3/NxHx4 tensors")
    batch=position_t.shape[0]
    if base_pos_w.shape!=(batch,3) or base_quat_w.shape!=(batch,4):
        raise ValueError("Task pose conversion requires one base position/quaternion per environment")
    ground=torch.as_tensor(ground_z,dtype=base_pos_w.dtype,device=base_pos_w.device)
    if ground.ndim==0:
        ground=ground.expand(batch)
    if ground.shape!=(batch,) or not bool(torch.isfinite(ground).all()):
        raise ValueError("ground_z must be finite scalar or N-vector")
    if not bool(torch.isfinite(position_t).all() and torch.isfinite(base_pos_w).all()):
        raise ValueError("Task/base positions must be finite")
    origin=torch.cat((base_pos_w[:,:2],ground[:,None]),dim=-1)
    yaw=yaw_quaternion(base_quat_w)
    if position_t.ndim==3:
        origin=origin[:,None,:]
        yaw=yaw[:,None,:]
    return origin+quat_apply(yaw,position_t),normalize_quat(quat_mul(yaw,normalize_quat(quat_t)))


def yaw_linear_velocity(base_lin_vel_b,base_quat_w):
    """Body linear velocity -> world -> yaw coordinates; returns all XYZ m/s."""
    if base_lin_vel_b.ndim!=2 or base_lin_vel_b.shape[-1]!=3 or base_quat_w.shape!=(base_lin_vel_b.shape[0],4):
        raise ValueError("Yaw linear velocity requires Nx3 body velocity and Nx4 base quaternion")
    if not bool(torch.isfinite(base_lin_vel_b).all()):
        raise ValueError("Body linear velocity must be finite")
    world=quat_apply(base_quat_w,base_lin_vel_b)
    return quat_apply_inverse(yaw_quaternion(base_quat_w),world)


def yaw_rate(previous_quat,current_quat,dt):
    """Wrapped finite difference of world yaw, not body angular-velocity Z."""
    previous,current=yaw_angle(previous_quat),yaw_angle(current_quat)
    if previous.shape!=current.shape:
        raise ValueError("Yaw-rate quaternion batches must match")
    period=torch.as_tensor(dt,dtype=current.dtype,device=current.device)
    if (period.ndim!=0 and period.shape!=current.shape) or not bool(torch.isfinite(period).all() and (period>0).all()):
        raise ValueError("Yaw-rate dt must be finite and positive")
    delta=current-previous
    return torch.atan2(torch.sin(delta),torch.cos(delta))/period


class CommandedPoseBank:
    """Fixed file/group sampling of T-pose + velocity commands at 50 Hz.

    The single-family sampler records episode errors for checkpoint consumers;
    its adaptive weights never change the fixed demonstration/group draws.
    Call reset(ids) before consuming those environments.
    """
    def __init__(self,batch,steps,device,seed=0,demonstrations=(),demonstration_group_weights=None):
        if type(batch) is not int or type(steps) is not int or min(batch,steps)<1:
            raise ValueError("Commanded pose bank requires positive integer batch/steps")
        self.batch,self.steps,self.device=batch,steps,device
        self.rng=np.random.default_rng(seed)
        self.demonstrations=[CommandedPoseTrajectory.load(path) for path in demonstrations]
        if not self.demonstrations or any(t.metadata["split"]!="train" for t in self.demonstrations):
            raise ValueError("Commanded pose bank requires nonempty pre-split train demonstrations only")
        self.demonstration_groups=None
        if demonstration_group_weights is not None:
            if not isinstance(demonstration_group_weights,dict) or not demonstration_group_weights:
                raise ValueError("Commanded pose group weights require a nonempty dictionary")
            groups={}
            for index,trajectory in enumerate(self.demonstrations):
                group=trajectory.metadata.get("training_group")
                if not isinstance(group,str) or not group.strip():
                    raise ValueError("Grouped command demonstrations require training_group")
                groups.setdefault(group,[]).append(index)
            active=[];masses=[]
            for group,weight in demonstration_group_weights.items():
                if (not isinstance(group,str) or not group.strip() or isinstance(weight,bool) or
                        not isinstance(weight,(int,float)) or not np.isfinite(weight) or weight<0):
                    raise ValueError("Commanded pose group weights must be named, finite and nonnegative")
                if weight>0:
                    if group not in groups:
                        raise ValueError(f"Positive command group {group!r} has no demonstrations")
                    active.append(groups[group]);masses.append(weight)
            if not masses:
                raise ValueError("Commanded pose group weights need positive total weight")
            masses=np.asarray(masses,dtype=float)
            masses/=masses.max()
            self.demonstration_groups=active
            self.demonstration_group_probabilities=masses/masses.sum()
        self.sampler=AdaptiveSampler(families=("velocity_pose",))
        self.family=np.zeros(batch,dtype=int)
        self.demonstration_index=np.full(batch,-1,dtype=int)
        self.positions=torch.zeros(batch,steps+5,3,device=device)
        self.orientations_wxyz=torch.zeros(batch,steps+5,4,device=device)
        self.velocity_commands_yaw=torch.zeros(batch,steps+5,3,device=device)
        times=np.arange(steps+5)*.02
        # Sample once per file; resets copy the same T data without transforms.
        self._positions=torch.as_tensor(np.stack([t.sample(times) for t in self.demonstrations]),device=device,dtype=torch.float32)
        self._orientations=torch.as_tensor(np.stack([t.sample_orientation(times) for t in self.demonstrations]),device=device,dtype=torch.float32)
        self._commands=torch.as_tensor(np.stack([t.sample_command(times) for t in self.demonstrations]),device=device,dtype=torch.float32)

    def reset(self,ids):
        if not isinstance(ids,torch.Tensor) or ids.ndim!=1 or ids.dtype not in (torch.int32,torch.int64):
            raise ValueError("Commanded pose reset ids must be a 1D integer tensor")
        selected_ids=ids.detach().cpu().numpy()
        if np.any(selected_ids<0) or np.any(selected_ids>=self.batch) or len(np.unique(selected_ids))!=len(selected_ids):
            raise ValueError("Commanded pose reset ids must be distinct valid environments")
        if self.demonstration_groups is None:
            selected=self.rng.integers(len(self.demonstrations),size=len(selected_ids))
        else:
            selected=[]
            for _ in selected_ids:
                group=self.demonstration_groups[int(self.rng.choice(len(self.demonstration_groups),p=self.demonstration_group_probabilities))]
                selected.append(group[int(self.rng.integers(len(group)))])
            selected=np.asarray(selected,dtype=int)
        source=torch.as_tensor(selected,device=self.device,dtype=torch.long)
        target=ids.to(device=self.device,dtype=torch.long)
        self.positions[target]=self._positions[source]
        self.orientations_wxyz[target]=self._orientations[source]
        self.velocity_commands_yaw[target]=self._commands[source]
        self.demonstration_index[selected_ids]=selected
        self.family[selected_ids]=0

    def current(self,indices):
        return self.positions[torch.arange(self.batch,device=self.device),indices.clamp(0,self.steps+4)]

    def current_orientation(self,indices):
        return self.orientations_wxyz[torch.arange(self.batch,device=self.device),indices.clamp(0,self.steps+4)]

    def current_command(self,indices):
        return self.velocity_commands_yaw[torch.arange(self.batch,device=self.device),indices.clamp(0,self.steps+4)]

    def future(self,indices):
        steps=indices[:,None]+torch.arange(1,5,device=self.device)
        valid=((steps>=0)&(steps<=self.steps)).unsqueeze(-1).expand(-1,-1,3)
        positions=self.positions[torch.arange(self.batch,device=self.device)[:,None],steps.clamp(0,self.steps+4)]
        return positions,valid
