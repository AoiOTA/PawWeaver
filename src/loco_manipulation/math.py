"""Explicit frame conventions used in both simulation backends."""
import numpy as np
import torch


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate [...,3] vectors using normalized scalar-first quaternions [...,4]."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(torch.cat((q[..., :1], -q[..., 1:]), dim=-1), v)


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

