"""One causal observation builder for Isaac Lab, MuJoCo and policy export."""
from dataclasses import asdict, dataclass
import torch

from .contracts import RobotState
from .math import normalize_quat,quat_apply_inverse,quat_conjugate,quat_mul,quat_to_rotation_6d


@dataclass(frozen=True)
class ObservationSpec:
    schema_version: int = 2
    history_frames: int = 5
    goal_frames: int = 4
    proprio_dim: int = 42
    action_dim: int = 18
    joint_velocity_scale: float = 0.05
    angular_velocity_scale: float = 0.25
    max_goal_age: float = 0.5
    control_dt: float = 0.02
    orientation_encoding: str = "rotation_columns_0_xyz_then_1_xyz"

    @property
    def size(self) -> int:
        # Preserve the 246-position prefix, then TCP rotation6D and four goal rotations6D.
        return self.history_frames*self.proprio_dim + self.action_dim + 3 + self.goal_frames*3 + 3 + 6 + self.goal_frames*6

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CommandObservationSpec(ObservationSpec):
    schema_version: int = 3
    task_mode: str = "velocity_ee_pose"
    target_frame: str = "current_base_xy_and_yaw_with_fixed_ground_z; excludes_base_height_roll_pitch"
    goal_history_frame: str = "task_frame; all_history_transformed_with_current_base_xy_yaw"
    velocity_command_order: str = "vx_yaw_m_s,vy_yaw_m_s,yaw_rate_rad_s"
    velocity_command_dim: int = 3

    @property
    def size(self) -> int:
        return super().size+self.velocity_command_dim


def observation_spec(config):
    mode=config.get("task_mode","world_ee_pose")
    if mode=="world_ee_pose":
        return ObservationSpec()
    if mode=="velocity_ee_pose":
        return CommandObservationSpec()
    raise ValueError(f"Unsupported task_mode: {mode!r}")


class _ObservationHistory:
    def __init__(self, batch: int, default_pos: torch.Tensor, spec: ObservationSpec | None = None):
        self.spec = spec or ObservationSpec()
        self.batch, self.device = batch, default_pos.device
        self.default_pos = default_pos
        self.proprio = torch.zeros(batch, self.spec.history_frames, 42, device=self.device)
        self.goals = torch.zeros(batch, self.spec.goal_frames, 3, device=self.device)
        self.goal_quats = torch.zeros(batch, self.spec.goal_frames, 4, device=self.device)
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

    def _reset(self, ids: torch.Tensor, state: RobotState, position: torch.Tensor, timestamp: torch.Tensor,
               orientation: torch.Tensor):
        if position.shape != (self.batch,3) or orientation.shape != (self.batch,4):
            raise ValueError("Reset requires batched goal XYZ and WXYZ")
        orientation = normalize_quat(orientation[ids])
        if not bool(torch.isfinite(position[ids]).all() and torch.isfinite(timestamp[ids]).all()):
            raise ValueError("Reset goal position and timestamp must be finite")
        current = self._proprio(state)
        self.proprio[ids] = current[ids, None, :]
        self.goals[ids] = position[ids, None, :]
        self.goal_quats[ids] = orientation[:,None,:]
        self.stamp[ids] = timestamp[ids]
        self.valid[ids] = True
        self.confidence[ids] = 1
        self.initialized[ids] = True

    def push_state(self, state: RobotState, ids: torch.Tensor | None = None):
        ids = torch.arange(self.batch, device=self.device) if ids is None else ids
        self.proprio[ids, :-1] = self.proprio[ids, 1:].clone()
        self.proprio[ids, -1] = self._proprio(state)[ids]
        # Histories are sampled at 50 Hz even when camera samples arrive at 30 Hz.
        self.goals[ids, :-1] = self.goals[ids, 1:].clone()
        self.goal_quats[ids, :-1] = self.goal_quats[ids, 1:].clone()

    def _push_goal(self, position: torch.Tensor, timestamp: torch.Tensor,
                   valid: torch.Tensor, confidence: torch.Tensor, orientation: torch.Tensor):
        if position.shape != (self.batch,3) or orientation.shape != (self.batch,4):
            raise ValueError("Goal updates require batched XYZ and WXYZ")
        finite = (torch.isfinite(position).all(-1) & torch.isfinite(timestamp) & torch.isfinite(confidence)
                  & torch.isfinite(orientation).all(-1) & (orientation.abs().amax(-1)>0))
        newer = timestamp > self.stamp
        accepted = finite & valid & newer & (confidence > 0)
        ids = accepted.nonzero(as_tuple=False).flatten()
        self.goals[ids, -1] = position[ids]
        self.goal_quats[ids, -1] = normalize_quat(orientation[ids])
        self.stamp[ids] = timestamp[ids]
        self.confidence[ids] = confidence[ids].clamp(0, 1)
        # Out-of-order measurements cannot invalidate a more recent accepted sample.
        new_or_equal = timestamp >= self.stamp
        self.valid[new_or_equal] = (finite & valid & (confidence > 0))[new_or_equal]

    def _build_from_world(self, state: RobotState, previous_action: torch.Tensor, now: torch.Tensor,
                          goals_w: torch.Tensor,goal_quats_w: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized.all()):
            raise RuntimeError("Reset every observation history before inference")
        # All historical world targets use the CURRENT base frame.
        relative = goals_w-state.base_pos_w[:, None, :]
        quats = state.base_quat_w[:, None, :].expand(-1, self.spec.goal_frames, -1)
        goal_b = quat_apply_inverse(quats, relative)
        tcp_b = quat_apply_inverse(state.base_quat_w, state.tcp_pos_w-state.base_pos_w)
        age = (now-self.stamp).clamp_min(0)
        valid = self.valid & (age <= self.spec.max_goal_age)
        quality = torch.stack((valid.float(), self.confidence*valid, age.clamp(max=10.0)), dim=-1)
        base_inverse = quat_conjugate(normalize_quat(state.base_quat_w))
        tcp_rotation = quat_to_rotation_6d(quat_mul(base_inverse,normalize_quat(state.tcp_quat_w)))
        goal_rotation = quat_to_rotation_6d(quat_mul(base_inverse[:,None,:],goal_quats_w))
        result = torch.cat((self.proprio.flatten(1), previous_action, tcp_b, goal_b.flatten(1), quality,
                            tcp_rotation,goal_rotation.flatten(1)), -1)
        return result

    def _validate(self,result):
        if result.shape[-1] != self.spec.size or not torch.isfinite(result).all():
            raise ValueError("Non-finite or inconsistent observation")
        return result


class ObservationBuilder(_ObservationHistory):
    """Original world-fixed EE-only pose observation interface."""
    def __init__(self,batch,default_pos,spec=None):
        super().__init__(batch,default_pos,spec)
        self.goals_w,self.goal_quats_w=self.goals,self.goal_quats

    def reset(self,ids,state,goal_w,timestamp,*,goal_quat_w):
        self._reset(ids,state,goal_w,timestamp,goal_quat_w)

    def push_goal(self,position_w,timestamp,valid,confidence,*,orientation_wxyz):
        self._push_goal(position_w,timestamp,valid,confidence,orientation_wxyz)

    def build(self,state,previous_action,now):
        return self._validate(self._build_from_world(state,previous_action,now,self.goals_w,self.goal_quats_w))


class CommandObservationBuilder(_ObservationHistory):
    """Keep causal goals in T; apply the same current T→world map to all frames."""
    def __init__(self,batch,default_pos,spec=None):
        spec=spec or CommandObservationSpec()
        if spec.to_dict()!=CommandObservationSpec().to_dict():
            raise ValueError("Command observations require the 279-dimensional task-frame contract")
        super().__init__(batch,default_pos,spec)
        self.goals_t,self.goal_quats_t=self.goals,self.goal_quats

    def reset(self,ids,state,goal_t,timestamp,*,goal_quat_t):
        self._reset(ids,state,goal_t,timestamp,goal_quat_t)

    def push_goal(self,position_t,timestamp,valid,confidence,*,orientation_t):
        self._push_goal(position_t,timestamp,valid,confidence,orientation_t)

    def build(self,state,previous_action,now,*,velocity_command,ground_z):
        from .commanded_pose import task_pose_to_world
        if velocity_command.shape!=(self.batch,3) or not torch.isfinite(velocity_command).all():
            raise ValueError("Velocity command requires finite batched yaw-frame vx, vy and yaw rate")
        goals_w,quats_w=task_pose_to_world(self.goals_t,self.goal_quats_t,
            state.base_pos_w,state.base_quat_w,ground_z)
        pose=self._build_from_world(state,previous_action,now,goals_w,quats_w)
        return self._validate(torch.cat((pose,velocity_command),dim=-1))
