"""Versioned simulator-independent I/O. Quaternions use scalar-first WXYZ."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FR", "FL", "RR", "RL")
                    for joint in ("hip", "thigh", "calf")) + tuple(f"arm_joint{i}" for i in range(1, 7))
FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
SCHEMA_VERSION = 1


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class GoalSample:
    """Position [m] in a fixed task frame, monotonic timestamp [s]."""
    timestamp: float
    position: tuple[float, float, float]
    valid: bool = True
    confidence: float = 1.0

    def __post_init__(self):
        if not np.isfinite([self.timestamp, *self.position, self.confidence]).all():
            raise ValueError("GoalSample must contain finite values")
        if len(self.position) != 3 or not 0 <= self.confidence <= 1:
            raise ValueError("GoalSample expects XYZ and confidence in [0,1]")


@dataclass
class RobotState:
    """Batched tensors; positions [m/rad], velocities [m/s, rad/s], base quaternion WXYZ."""
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    base_pos_w: torch.Tensor
    base_quat_w: torch.Tensor
    base_ang_vel_b: torch.Tensor
    tcp_pos_w: torch.Tensor
    base_lin_vel_b: torch.Tensor | None = None  # supervision/critic only


@dataclass(frozen=True)
class ActuatorSpec:
    """Ordered 18-joint contract, angles [rad], torque [N m], gains [N m/rad, N m s/rad]."""
    joint_names: tuple[str, ...]
    default_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    effort: tuple[float, ...]
    velocity: tuple[float, ...]
    armature: tuple[float, ...]
    damping: tuple[float, ...]
    frictionloss: tuple[float, ...]
    delay_steps: tuple[int, ...]
    physics_dt: float = 0.002
    decimation: int = 10

    def __post_init__(self):
        if tuple(self.joint_names) != JOINT_NAMES:
            raise ValueError("Joint contract must match FR,FL,RR,RL hip/thigh/calf then arm_joint1..6")
        vectors = (self.default_pos, self.action_scale, self.kp, self.kd, self.lower, self.upper,
                   self.effort, self.velocity, self.armature, self.damping, self.frictionloss, self.delay_steps)
        if any(len(values) != 18 or not np.isfinite(values).all() for values in vectors):
            raise ValueError("Every actuator vector must contain 18 finite entries")
        for values in (self.action_scale, self.kp, self.effort, self.velocity):
            if np.any(np.asarray(values) <= 0):
                raise ValueError("Action scales, Kp, effort and velocity limits must be positive")
        for values in (self.kd, self.armature, self.damping, self.frictionloss, self.delay_steps):
            if np.any(np.asarray(values) < 0):
                raise ValueError("Damping, friction, armature and delays must be nonnegative")
        if any(int(value) != value for value in self.delay_steps):
            raise ValueError("Delays are integer physics steps")
        if np.any(np.asarray(self.lower) >= np.asarray(self.upper)):
            raise ValueError("Joint lower limits must be below upper limits")
        q0 = np.asarray(self.default_pos)
        if np.any(q0 < self.lower) or np.any(q0 > self.upper):
            raise ValueError("Default pose exceeds joint limits")
        if self.physics_dt != 0.002 or self.decimation != 10:
            raise ValueError("v1 requires 2 ms physics and 50 Hz control")

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ActuatorSpec":
        return cls(**{key: tuple(value) if isinstance(value, list) else value for key, value in values.items()})

    @classmethod
    def load(cls, path: Path) -> "ActuatorSpec":
        return cls.from_dict(json.loads(path.read_text()))


def named_indices(available: list[str] | tuple[str, ...], required: tuple[str, ...]) -> list[int]:
    if len(set(available)) != len(available):
        raise ValueError("Duplicate names in simulator interface")
    missing = set(required) - set(available)
    if missing:
        raise ValueError(f"Missing required joints/bodies: {sorted(missing)}")
    return [available.index(name) for name in required]

