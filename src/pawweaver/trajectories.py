"""Fixed-task-frame trajectories and performance-driven sampling (no simulator imports)."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

FAMILIES = ("reach", "line", "arc", "eight", "smooth")

@dataclass
class Trajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.timestamps = np.asarray(self.timestamps,dtype=np.float64)
        self.positions = np.asarray(self.positions,dtype=np.float64)
        if self.timestamps.ndim != 1 or len(self.timestamps) < 2 or self.positions.shape != (len(self.timestamps),3):
            raise ValueError("Trajectory requires N increasing timestamps and Nx3 positions")
        if not np.isfinite(self.timestamps).all() or not np.isfinite(self.positions).all() or (np.diff(self.timestamps)<=0).any():
            raise ValueError("Trajectory has invalid time or position data")

    def sample(self, times):
        return np.stack([np.interp(times,self.timestamps,self.positions[:,axis]) for axis in range(3)],axis=-1)

    def align(self, start, yaw=0.):
        c,s = np.cos(yaw),np.sin(yaw)
        rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        return Trajectory(self.timestamps.copy(),(self.positions-self.positions[0])@rotation.T+start,
                          dict(self.metadata,alignment="episode_start_only",yaw=float(yaw)))

    def save(self,path:Path):
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,timestamps=self.timestamps,positions=self.positions,
                            metadata=json.dumps(self.metadata,sort_keys=True))

    @classmethod
    def load(cls,path:Path):
        with np.load(path,allow_pickle=False) as data:
            return cls(data["timestamps"],data["positions"],json.loads(str(data["metadata"])))

def synthetic(family:str, seed:int, duration=60., dt=.02, amplitude=.3, speed=.3):
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
    return Trajectory(t,p,{"family":family,"seed":seed,"source":"synthetic","max_requested_speed":speed})

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
