"""Shared explicit PD; delays are evaluated at every physics step, not policy step."""
import torch
from .contracts import ActuatorSpec


class JointPD:
    def __init__(self, spec: ActuatorSpec, batch: int, device: str = "cpu"):
        self.spec = spec
        self.batch = batch
        self.device = device
        for name in ("default_pos", "action_scale", "kp", "kd", "lower", "upper", "effort"):
            setattr(self, name, torch.tensor(getattr(spec, name), device=device, dtype=torch.float32))
        self.delays = torch.tensor(spec.delay_steps, device=device, dtype=torch.long).repeat(batch, 1)
        self.capacity = max(max(spec.delay_steps), 10) + 1
        self.queue = self.default_pos.repeat(self.capacity, batch, 1)
        self.cursor = 0
        self.target = self.default_pos.repeat(batch, 1)
        self.last_action = torch.zeros(batch, 18, device=device)
        self.strength = torch.ones(batch, 18, device=device)
        self.kp_factor = torch.ones(batch, 18, device=device)
        self.kd_factor = torch.ones(batch, 18, device=device)
        self.raw_torque = torch.zeros(batch, 18, device=device)

    def reset(self, ids: torch.Tensor, position: torch.Tensor | None = None):
        position = self.default_pos.expand(len(ids), -1) if position is None else position
        self.target[ids] = position
        self.queue[:, ids] = position
        self.last_action[ids] = 0
        self.raw_torque[ids] = 0

    def command(self, action: torch.Tensor):
        if action.shape != (self.batch, 18) or not torch.isfinite(action).all():
            raise ValueError("Expected finite [batch,18] action")
        action = action.clamp(-1, 1)
        self.last_action.copy_(action)
        self.target = torch.clamp(self.default_pos + self.action_scale * action, self.lower, self.upper)

    def torque(self, position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        self.queue[self.cursor] = self.target
        queue_index = (self.cursor - self.delays) % self.capacity
        batch_index = torch.arange(self.batch, device=self.device).unsqueeze(1)
        joint_index = torch.arange(18, device=self.device).unsqueeze(0)
        delayed = self.queue[queue_index, batch_index, joint_index]
        self.cursor = (self.cursor + 1) % self.capacity
        self.raw_torque = self.kp*self.kp_factor*(delayed-position) - self.kd*self.kd_factor*velocity
        # Strength changes the available motor torque; do not exceed the randomized limit.
        limit = self.effort * self.strength
        return torch.clamp(self.raw_torque * self.strength, -limit, limit)

