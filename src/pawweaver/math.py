"""Explicit frame conventions used in both simulation backends."""
import numpy as np
import torch


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    """Normalize finite nonzero WXYZ quaternions; never invent an orientation."""
    if q.dim() < 1 or q.size(-1) != 4:
        raise ValueError("Expected WXYZ quaternions with final dimension 4")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("Quaternion must be finite")
    scale = q.abs().amax(dim=-1,keepdim=True)
    if bool((scale == 0).any()):
        raise ValueError("Quaternion must be nonzero")
    scaled = q / scale
    return scaled / torch.linalg.vector_norm(scaled,dim=-1,keepdim=True)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1],-q[..., 1:]),dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Broadcasting WXYZ Hamilton product a*b; normalize separately if needed."""
    a,b = torch.broadcast_tensors(a,b)
    aw,ax,ay,az = a.unbind(-1)
    bw,bx,by,bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw),dim=-1)


def quat_angle_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest SO(3) angular distance in radians, invariant to quaternion sign."""
    relative = quat_mul(quat_conjugate(normalize_quat(a)),normalize_quat(b))
    return 2*torch.atan2(torch.linalg.vector_norm(relative[...,1:],dim=-1),relative[...,0].abs())


def quat_to_rotation_6d(q: torch.Tensor) -> torch.Tensor:
    """Rotation columns concatenated: [R00,R10,R20,R01,R11,R21]."""
    w,x,y,z = normalize_quat(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y),
                        2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate [...,3] vectors using normalized scalar-first quaternions [...,4]."""
    q = normalize_quat(q)
    xyz,v = torch.broadcast_tensors(q[...,1:],v)
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def rpy_matrix(rpy) -> np.ndarray:
    """URDF extrinsic XYZ: Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def rpy_quat(rpy) -> np.ndarray:
    r, p, y = np.asarray(rpy) * 0.5
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def transform(xyz=(0., 0., 0.), rpy=(0., 0., 0.)) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rpy_matrix(rpy), xyz
    return result


def axis_angle_matrix(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*skew + (1.0-np.cos(angle))*(skew @ skew)
