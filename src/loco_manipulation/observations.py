"""One causal observation builder for Isaac Lab, MuJoCo and policy export."""
from dataclasses import asdict, dataclass
import torch

from .contracts import RobotState
from .math import quat_apply_inverse


@dataclass(frozen=True)
class ObservationSpec:
    history_frames: int = 5
    goal_frames: int = 4
    proprio_dim: int = 42
    action_dim: int = 18
    joint_velocity_scale: float = 0.05
    angular_velocity_scale: float = 0.25
    max_goal_age: float = 0.5

    @property
    def size(self) -> int:
        # proprio history, previous action, TCP position, relative goal history, valid/confidence/age
        return self.history_frames*self.proprio_dim + self.action_dim + 3 + self.goal_frames*3 + 3

    def to_dict(self) -> dict:
        return asdict(self)


class ObservationBuilder:
    def __init__(self, batch: int, default_pos: torch.Tensor, spec: ObservationSpec | None = None):
        self.spec = spec or ObservationSpec()
        self.batch, self.device = batch, default_pos.device
        self.default_pos = default_pos
        self.proprio = torch.zeros(batch, self.spec.history_frames, 42, device=self.device)
        self.goals_w = torch.zeros(batch, self.spec.goal_frames, 3, device=self.device)
        self.stamp = torch.full((batch,), -torch.inf, device=self.device)
        self.valid = torch.zeros(batch, device=self.device, dtype=torch.bool)
        self.confidence = torch.zeros(batch, device=self.device)
        self.initialized = torch.zeros(batch, device=self.device, dtype=torch.bool)

    def _proprio(self, state: RobotState) -> torch.Tensor:
        down = torch.zeros_like(state.base_pos_w)
        down[:, 2] = -1
        gravity = quat_apply_inverse(state.base_quat_w, down)
        return torch.cat((state.joint_pos-self.default_pos,
                          state.joint_vel*self.spec.joint_velocity_scale,
                          state.base_ang_vel_b*self.spec.angular_velocity_scale, gravity), dim=-1)

    def reset(self, ids: torch.Tensor, state: RobotState, goal_w: torch.Tensor, timestamp: torch.Tensor):
        current = self._proprio(state)
        self.proprio[ids] = current[ids, None, :]
        self.goals_w[ids] = goal_w[ids, None, :]
        self.stamp[ids] = timestamp[ids]
        self.valid[ids] = True
        self.confidence[ids] = 1
        self.initialized[ids] = True

    def push_state(self, state: RobotState, ids: torch.Tensor | None = None):
        ids = torch.arange(self.batch, device=self.device) if ids is None else ids
        self.proprio[ids, :-1] = self.proprio[ids, 1:].clone()
        self.proprio[ids, -1] = self._proprio(state)[ids]

    def push_goal(self, position_w: torch.Tensor, timestamp: torch.Tensor,
                  valid: torch.Tensor, confidence: torch.Tensor):
        finite = torch.isfinite(position_w).all(-1) & torch.isfinite(timestamp) & torch.isfinite(confidence)
        newer = timestamp > self.stamp
        accepted = finite & valid & newer & (confidence > 0)
        ids = accepted.nonzero(as_tuple=False).flatten()
        self.goals_w[ids, :-1] = self.goals_w[ids, 1:].clone()
        self.goals_w[ids, -1] = position_w[ids]
        self.stamp[ids] = timestamp[ids]
        self.confidence[ids] = confidence[ids].clamp(0, 1)
        # Out-of-order measurements cannot invalidate a more recent accepted sample.
        new_or_equal = timestamp >= self.stamp
        self.valid[new_or_equal] = (finite & valid & (confidence > 0))[new_or_equal]

    def build(self, state: RobotState, previous_action: torch.Tensor, now: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized.all()):
            raise RuntimeError("Reset every observation history before inference")
        # All historical world targets use the CURRENT base frame.
        relative = self.goals_w-state.base_pos_w[:, None, :]
        quats = state.base_quat_w[:, None, :].expand(-1, self.spec.goal_frames, -1)
        goal_b = quat_apply_inverse(quats, relative)
        tcp_b = quat_apply_inverse(state.base_quat_w, state.tcp_pos_w-state.base_pos_w)
        age = (now-self.stamp).clamp_min(0)
        valid = self.valid & (age <= self.spec.max_goal_age)
        quality = torch.stack((valid.float(), self.confidence*valid, age.clamp(max=10.0)), dim=-1)
        result = torch.cat((self.proprio.flatten(1), previous_action, tcp_b, goal_b.flatten(1), quality), -1)
        if result.shape[-1] != self.spec.size or not torch.isfinite(result).all():
            raise ValueError("Non-finite or inconsistent observation")
        return result
