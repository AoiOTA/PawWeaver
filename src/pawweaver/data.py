"""Read only FastUMI pose arrays; explicit time, units and pose-frame calibration are required."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp
from .trajectories import Trajectory, split_for_source

def rigid_transform(value):
    matrix = np.asarray(value,dtype=float)
    if matrix.shape != (4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1]):
        raise ValueError("Calibration must be a finite homogeneous 4x4 transform")
    r = matrix[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(r),1,atol=1e-6):
        raise ValueError("Calibration rotation must be proper orthonormal")
    return matrix

def convert(source:Path, timestamps:np.ndarray, config:dict, source_id:str):
    # Official qpos: XYZ QX QY QZ QW, optionally followed by gripper opening.
    # Raw T265 and processed TCP files share the same key; shape cannot distinguish them.
    pose_kind = config["pose_kind"]
    if pose_kind not in ("tcp","sensor"):
        raise ValueError("Declare pose_kind tcp or sensor explicitly")
    with h5py.File(source,"r") as file:
        q = file[config.get("pose_key","observations/qpos")][:]
    if q.ndim != 2 or q.shape[1] not in (7,8) or not np.isfinite(q).all():
        raise ValueError("Expected FastUMI XYZ + XYZW pose rows; joint trajectories are not accepted")
    times = np.asarray(timestamps,dtype=float)*float(config["timestamp_scale_s"])
    xyz = q[:,:3]*float(config["position_scale_m"])
    norms = np.linalg.norm(q[:,3:7],axis=-1)
    if not np.allclose(norms,1,atol=.01):
        raise ValueError("Quaternion norm invalid; verify XYZW convention")
    source_rotations = Rotation.from_quat(q[:,3:7])
    rotations = source_rotations.as_matrix()
    extrinsic = rigid_transform(config["sensor_to_tcp"]) if pose_kind == "sensor" else np.eye(4)
    xyz = xyz+np.einsum("nij,j->ni",rotations,extrinsic[:3,3])
    task = rigid_transform(config["source_to_task"])
    xyz = xyz@task[:3,:3].T+task[:3,3]
    goal_rotations = Rotation.from_matrix(task[:3,:3]) * source_rotations * Rotation.from_matrix(extrinsic[:3,:3])
    orientations = goal_rotations.as_quat()[:,[3,0,1,2]]
    Trajectory(times,xyz,orientations,{})
    times = times-times[0]
    max_gap = float(config["max_gap_s"])
    if np.max(np.diff(times))>max_gap:
        raise ValueError("Timestamp gap exceeds configured threshold; segment source episode first")
    dt = float(config.get("dt",.02))
    max_speed,max_acc = float(config["max_speed_m_s"]),float(config["max_acceleration_m_s2"])
    if min(dt,max_speed,max_acc)<=0:
        raise ValueError("Sampling period and motion bounds must be positive")
    spline = PchipInterpolator(times,xyz,axis=0)
    probe = np.linspace(0,times[-1],max(len(times)*10,100))
    scale = max(1.,np.linalg.norm(spline(probe,1),axis=-1).max()/max_speed,
                np.sqrt(np.linalg.norm(spline(probe,2),axis=-1).max()/max_acc))
    new_t = np.arange(0,times[-1]*scale+1e-9,dt)
    positions = spline(new_t/scale)
    orientations = Slerp(times,goal_rotations)(new_t/scale).as_quat()[:,[3,0,1,2]]
    lower,upper = np.asarray(config["task_bounds_m"],float)
    if (positions<lower).any() or (positions>upper).any():
        raise ValueError("Trajectory is outside the configured task workspace")
    metadata = {"family":"fastumi","source_id":source_id,"source_pose_sha256":hashlib.sha256(q.tobytes()).hexdigest(),
                "split":split_for_source(source_id),"time_scale":float(scale),"conversion":config,
                "reachability":"task bounds only; robot-specific validation required"}
    return Trajectory(new_t,positions,orientations,metadata)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--timestamps",type=Path,required=True,help="N-element .npy with verified per-frame timestamps")
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--source-id",required=True,help="Stable session/demo ID shared by all crops and variants")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = convert(args.source,np.load(args.timestamps,allow_pickle=False),json.loads(args.config.read_text()),args.source_id)
    result.save(args.output)
    print(json.dumps(result.metadata,indent=2))

if __name__ == "__main__":
    main()
