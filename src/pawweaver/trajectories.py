"""Fixed-task-frame trajectories and performance-driven sampling (no simulator imports)."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FAMILIES = ("reach", "line", "arc", "eight", "smooth")

@dataclass
class Trajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    orientations_wxyz: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.timestamps = np.asarray(self.timestamps,dtype=np.float64)
        self.positions = np.asarray(self.positions,dtype=np.float64)
        self.orientations_wxyz = np.asarray(self.orientations_wxyz,dtype=np.float64)
        if self.timestamps.ndim != 1 or len(self.timestamps) < 2 or self.positions.shape != (len(self.timestamps),3):
            raise ValueError("Trajectory requires N increasing timestamps and Nx3 positions")
        if not np.isfinite(self.timestamps).all() or not np.isfinite(self.positions).all() or (np.diff(self.timestamps)<=0).any():
            raise ValueError("Trajectory has invalid time or position data")
        if self.orientations_wxyz.shape != (len(self.timestamps),4) or not np.isfinite(self.orientations_wxyz).all():
            raise ValueError("Pose trajectory requires finite Nx4 WXYZ orientations")
        # Scale first so finite, very large/small quaternion coefficients normalize safely.
        scale = np.max(np.abs(self.orientations_wxyz),axis=-1,keepdims=True)
        if (scale==0).any():
            raise ValueError("Pose trajectory orientations must be nonzero quaternions")
        scaled = self.orientations_wxyz/scale
        self.orientations_wxyz = scaled/np.linalg.norm(scaled,axis=-1,keepdims=True)

    def sample(self, times):
        return np.stack([np.interp(times,self.timestamps,self.positions[:,axis]) for axis in range(3)],axis=-1)

    def sample_orientation(self, times):
        """Shortest-path SLERP, with the same endpoint holding as position sampling."""
        query = np.asarray(times,dtype=np.float64)
        if not np.isfinite(query).all():
            raise ValueError("Orientation sample times must be finite")
        rotations = Rotation.from_quat(self.orientations_wxyz[:,[1,2,3,0]])
        sampled = Slerp(self.timestamps,rotations)(np.clip(query.reshape(-1),self.timestamps[0],self.timestamps[-1]))
        return sampled.as_quat()[:,[3,0,1,2]].reshape(query.shape+(4,))

    def align(self, start, yaw=0., start_orientation_wxyz=None):
        """Apply one fixed rigid transform, never reanchor to the moving base.

        With a supplied start orientation, the first pose matches it before the
        optional additional world-yaw rotation. With yaw=0 it matches exactly.
        """
        start = np.asarray(start,dtype=float)
        if start.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(yaw):
            raise ValueError("Alignment requires a finite position and yaw")
        rotation = Rotation.from_rotvec([0.,0.,yaw])
        source = Rotation.from_quat(self.orientations_wxyz[:,[1,2,3,0]])
        metadata = dict(self.metadata,alignment="episode_start_only",yaw=float(yaw))
        if start_orientation_wxyz is not None:
            q = np.asarray(start_orientation_wxyz,dtype=float)
            if q.shape != (4,) or not np.isfinite(q).all() or not np.any(q):
                raise ValueError("Alignment requires a finite nonzero WXYZ start orientation")
            q = q/np.max(np.abs(q))
            rotation = rotation*Rotation.from_quat(q[[1,2,3,0]])*source[0].inv()
            metadata["start_orientation_wxyz_before_yaw"] = (q/np.linalg.norm(q)).tolist()
        return Trajectory(self.timestamps.copy(),rotation.apply(self.positions-self.positions[0])+start,
                          (rotation*source).as_quat()[:,[3,0,1,2]],metadata)

    def save(self,path:Path):
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,schema_version=2,timestamps=self.timestamps,positions=self.positions,
                            orientations_wxyz=self.orientations_wxyz,
                            metadata=json.dumps(self.metadata,sort_keys=True))

    @classmethod
    def load(cls,path:Path):
        with np.load(path,allow_pickle=False) as data:
            if "orientations_wxyz" not in data:
                raise ValueError("Position-only trajectory is missing required pose orientations_wxyz")
            if "schema_version" not in data or np.asarray(data["schema_version"]).shape != () or data["schema_version"].item()!=2:
                raise ValueError("Pose trajectory requires schema_version=2")
            return cls(data["timestamps"],data["positions"],data["orientations_wxyz"],json.loads(str(data["metadata"])))

def synthetic(family:str, seed:int, duration=60., dt=.02, amplitude=.3, speed=.3, *, orientation_wxyz):
    if family not in FAMILIES or min(duration,dt,amplitude,speed)<=0:
        raise ValueError("Invalid synthetic trajectory settings")
    rng = np.random.default_rng(seed)
    t = np.arange(0,duration+dt*.5,dt)
    omega = speed/(3*amplitude)
    u = omega*t
    p = np.zeros((len(t),3))
    if family == "reach":
        p[:] = [rng.uniform(.3,2),rng.uniform(-.7,.7),rng.uniform(.5,.95)]
    elif family == "line":
        # Smooth reciprocal motion; never discontinuously wraps at the endpoint.
        p[:,0] = amplitude*np.sin(u)
    elif family == "arc":
        p[:,:2] = amplitude*np.stack((np.cos(u)-1,np.sin(u)),axis=-1)
    elif family == "eight":
        p[:,:2] = amplitude*np.stack((np.sin(u),.5*np.sin(2*u)),axis=-1)
    else:
        phases = rng.uniform(-np.pi,np.pi,(3,3))
        for axis in range(3):
            for k in range(1,4):
                p[:,axis] += amplitude/(k*k*3)*np.sin(k*u+phases[axis,k-1])
    if family != "reach":
        p += [.45,0,.7]
    orientation = np.asarray(orientation_wxyz,dtype=float)
    if orientation.shape != (4,):
        raise ValueError("Synthetic orientation must be an explicit WXYZ quaternion")
    return Trajectory(t,p,np.repeat(orientation[None,:],len(t),axis=0),
                      {"family":family,"seed":seed,"source":"synthetic","max_requested_speed":speed,
                       "orientation_generation":"explicit_constant_wxyz",
                       "orientation_wxyz":orientation.tolist(),"reachability":"not robot-validated"})

def split_for_source(source_id:str,validation_fraction=.2):
    # Source identity is assigned BEFORE cropping, alignment, speed scaling or augmentation.
    value = int(hashlib.sha256(source_id.encode()).hexdigest()[:16],16)/2**64
    return "validation" if value < validation_fraction else "train"

class AdaptiveSampler:
    def __init__(self,families=FAMILIES,floor=.05,ema=.1):
        self.families = tuple(families)
        if not 0<=floor<1/len(families) or not 0<ema<=1:
            raise ValueError("Invalid adaptive sampler parameters")
        self.floor,self.ema = floor,ema
        self.errors = np.ones(len(families),dtype=float)
        self.counts = np.zeros(len(families),dtype=int)

    def update(self,family,normalized_error):
        if not np.isfinite(normalized_error) or normalized_error<0:
            raise ValueError("Sampling error must be finite and nonnegative")
        i = self.families.index(family)
        self.errors[i] = (1-self.ema)*self.errors[i]+self.ema*normalized_error
        self.counts[i] += 1

    @property
    def probabilities(self):
        weights = np.sqrt(np.clip(self.errors,1e-4,100))
        return self.floor+(1-len(weights)*self.floor)*weights/weights.sum()

    def sample(self,rng,size):
        return rng.choice(len(self.families),size=size,p=self.probabilities)

    def state_dict(self):
        return {"families":self.families,"floor":self.floor,"ema":self.ema,
                "errors":self.errors.tolist(),"counts":self.counts.tolist()}

    def load_state_dict(self,state):
        if tuple(state["families"])!=self.families or state["floor"]!=self.floor or state["ema"]!=self.ema:
            raise ValueError("Sampler configuration differs from saved checkpoint")
        self.errors=np.asarray(state["errors"],float)
        self.counts=np.asarray(state["counts"],int)
        if self.errors.shape!=(len(self.families),) or not np.isfinite(self.errors).all() or (self.errors<0).any():
            raise ValueError("Invalid saved sampler weights")
