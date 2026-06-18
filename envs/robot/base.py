"""Robot abstraction — the central seam between tasks and embodiments.

A ``Robot`` owns the articulation, exposes a uniform state/control interface,
and delegates motion planning to a pluggable ``Planner``.

Subclasses implement embodiment-specific behaviour (prismatic vs angular
gripper, PD position vs velocity bang-bang, body-pose vs FK end-effector).

Tasks call *only* the methods defined here — they never branch on robot type.
"""

from __future__ import annotations

import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg

from .planner import Planner, PlannerResult
from ..utils.transforms import Pose
from .._global import EMBODIMENTS_ROOT

if TYPE_CHECKING:
    from .._base_task import BaseTask
    from ..sensors.tactile import TactileCfg


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RobotConfig:
    """Flat configuration for a robot embodiment.

    Created by factory functions in ``robot_cfg.py`` and consumed by
    ``Robot.__init__``.  Tasks never construct this directly.
    """

    robot: ArticulationCfg
    tactiles: list  # list[TactileCfg] — deferred import to avoid circular dep
    gripper_offset: float = 0.131  # metres, ee → gripper-centre
    gripper_max_qpos: float = 0.039  # metres (prismatic) or rad (angular)
    tactile_far_plane: float = 30.0  # mm, resting gel depth
    adaptive_grasp_depth_threshold: float | None = None  # mm
    contact_threshold: tuple[float, float] = (27.5, 28.0)  # (min, max) mm
    planner_time_dilation_factor: float = 1.0


# ---------------------------------------------------------------------------
# Abstract Robot
# ---------------------------------------------------------------------------

class Robot(ABC):
    """Uniform interface for all robot embodiments.

    Lifecycle
    ---------
    1. ``__init__(cfg, task)`` — load articulation, map joint indices.
    2. ``setup()`` — initialise planner, calibrate offsets, set joint limits.
    3. The task uses state/control/planning methods every step.

    Subclasses MUST set these class-level constants::

        hand_name: str
        arm_joint_names: list[str]
        gripper_joint_names: list[str]
        yaml_path: str              # cuRobo config
    """

    # ── Set by subclasses ──
    hand_name: str = ""
    arm_joint_names: list[str] = []
    gripper_joint_names: list[str] = []
    yaml_path: str = ""

    def __init__(self, cfg: RobotConfig, task: "BaseTask"):
        self.cfg = cfg
        self.task = task
        self.device = task.device

        # Load articulation (lightweight — scene not fully set up yet)
        self.articulation = Articulation(cfg.robot)
        task.scene.articulations["robot"] = self.articulation

        # Joint IDs mapped lazily in setup() — not available during __init__
        self._arm_ids: torch.Tensor | None = None
        self._gripper_ids: torch.Tensor | None = None

        # Offset from EE (panda_hand) to gripper-centre / grasp TCP
        offset = cfg.gripper_offset
        self._offset = Pose(p=[0, 0, -offset], q=[1, 0, 0, 0])
        self._offset_pos = torch.tensor(
            [0.0, 0.0, offset], device=self.device
        ).repeat(task.num_envs, 1)
        self._offset_rot = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(task.num_envs, 1)

        self.root_pose: Pose | None = None
        self.origin_pose: Pose | None = None
        self.planner: Planner | None = None
        self._last_arm_velocity: torch.Tensor | None = None
        self._last_gripper_velocity: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Abstract — every embodiment implements these differently
    # ------------------------------------------------------------------

    def _map_joint_ids(self) -> None:
        """Map joint names → indices.  Must be called during setup() — NOT __init__."""
        joint_names = self.articulation.joint_names
        name_to_id = {n: i for i, n in enumerate(joint_names)}
        self._arm_ids = torch.tensor(
            [name_to_id[n] for n in self.arm_joint_names], device=self.device
        )
        self._gripper_ids = torch.tensor(
            [name_to_id[n] for n in self.gripper_joint_names], device=self.device
        )

    @abstractmethod
    def setup(self) -> None:
        """Initialise planner, body indices, joint limits.  Called once after __init__."""
        ...

    @abstractmethod
    def get_ee_pose(self, env_ids: slice | None = None) -> Pose:
        """Current end-effector pose in robot-root frame."""
        ...

    @abstractmethod
    def set_arm(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor | None = None,
        env_ids: slice | None = None,
        force: bool = True,
    ) -> None:
        """Command the arm joints."""
        ...

    @abstractmethod
    def set_gripper(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor | None = None,
        env_ids: slice | None = None,
        force: bool = True,
    ) -> None:
        """Command the gripper."""
        ...

    @abstractmethod
    def plan_gripper(self, pos: float, type: str = "percent") -> dict:
        """Plan a gripper trajectory.  Returns {status, num_steps, position, velocity}."""
        ...

    @abstractmethod
    def is_overpressed(self, min_depth: float) -> bool:
        """Return True if tactile indicates the gel is over-compressed."""
        ...

    # ------------------------------------------------------------------
    # Concrete — shared implementation
    # ------------------------------------------------------------------

    def get_qpos(self) -> torch.Tensor:
        return self.articulation.data.joint_pos.clone().cpu()

    def get_gripper_qpos(self) -> float:
        return float(self.get_qpos()[0, self._gripper_ids[0]].item())

    def gripper_percent_to_qpos(self, percentage: float) -> float:
        return self.gripper_max_qpos * percentage

    def gripper_qpos_to_percent(self, qpos: float | None = None) -> float:
        if qpos is None:
            qpos = self.get_gripper_qpos()
        return qpos / self.gripper_max_qpos

    def ee_to_gripper_center(self, ee_pose: Pose) -> Pose:
        """EE (panda_hand) → grasp TCP."""
        return ee_pose.add_offset(self._offset.inv())

    def gripper_center_to_ee(self, gripper_center_pose: Pose) -> Pose:
        """Grasp TCP → EE (panda_hand) for planning."""
        return gripper_center_pose.add_offset(self._offset)

    def get_gripper_center_pose(self, env_ids: slice | None = None) -> Pose:
        return self.ee_to_gripper_center(self.get_ee_pose(env_ids))

    def get_inhand_pose(self, actor) -> Pose:
        from ..utils.actor import Actor

        return actor.get_pose().rebase(self.get_gripper_center_pose())

    def plan_arm(
        self,
        target_pose: Pose,
        constraint_pose=None,
        pre_dis: float | None = None,
        time_dilation_factor: float | None = None,
    ) -> dict:
        """Plan arm trajectory via the configured Planner."""
        if time_dilation_factor is None:
            time_dilation_factor = self.cfg.planner_time_dilation_factor
        if self.planner is None:
            return {"status": "Fail", "num_steps": 0, "position": None, "velocity": None}

        # Actual plan_arm delegates to cuRobo — subclasses keep the concrete
        # implementation since the call signature depends on cuRobo internals.
        raise NotImplementedError("Subclass must implement plan_arm with cuRobo binding")

    def get_observations(self, data_type: list[str] | None = None) -> dict:
        obs: dict = {}
        if data_type is None:
            data_type = ["joint", "ee"]
        if "ee" in data_type:
            obs["ee"] = self.get_ee_pose().totensor(device=self.device)
        if "joint" in data_type:
            obs["joint"] = self.articulation.data.joint_pos.squeeze(0)
        return obs

    def get_grasp_perfect_direction(self) -> str:
        return "top_down"

    @property
    def adaptive_grasp_step_coarse(self) -> float:
        """Coarse step size for adaptive grasping (robot-native units)."""
        return 0.0005

    @property
    def adaptive_grasp_step_fine(self) -> float:
        """Fine (contact) step size for adaptive grasping (robot-native units)."""
        return 0.00005

    @property
    def needs_post_settle(self) -> bool:
        """Whether the arm needs extra settle steps after a trajectory."""
        return False

    @property
    def grasp_open_axis_index(self) -> int:
        """Which column of the rotation matrix points along the gripper open direction."""
        return 0

    @property
    def grasp_height_clearance(self) -> float:
        """Extra Z clearance (m) needed above an object for safe approach.

        Grippers with longer fingers (ZX Hand) need more clearance to avoid
        hitting the table when approaching a low object from above.
        """
        return 0.0

    @property
    def grasp_pre_dis_extra(self) -> float:
        """Extra pre-grasp distance (m) for this embodiment beyond the task's
        standard pre-grasp offset.  Longer fingers need to start higher."""
        return 0.0

    def build_grasp_pose(
        self,
        target_position: np.ndarray,
        approach_direction: np.ndarray,
        object_x_axis: np.ndarray | None = None,
    ) -> "Pose":
        """Build a grasp pose corrected for this embodiment's gripper geometry.

        Embodiment-specific adjustments applied automatically:

        * ``grasp_height_clearance`` — extra Z offset so the gripper centre
          lands at the correct height after the fingers rotate closed.
        * ``grasp_open_axis_index`` — rotates the grasp frame so the fingers
          open along the correct local axis (object x for GelSight, object y
          for ZX Hand).

        Tasks call this instead of ``construct_grasp_pose`` directly.

        Parameters
        ----------
        target_position: (3,) — where the gripper centre should be at contact.
        approach_direction: (3,) — direction the gripper approaches from.
        object_x_axis: (3,) — object's local x-axis, used to determine gripper
            open direction.  If None, defaults to [1, 0, 0].
        """
        from ..utils.transforms import construct_grasp_pose

        if object_x_axis is None:
            object_x_axis = np.array([1.0, 0.0, 0.0])

        # Apply embodiment-specific height clearance.
        # ZX fingers rotate closed, shifting the gel-midpoint downward.
        adjusted_position = np.asarray(target_position, dtype=float).copy()
        adjusted_position[2] += self.grasp_height_clearance

        # Each embodiment's fingers open along a different local axis of the
        # grasp rotation matrix.  GelSight (idx=0) opens along column 0 (x),
        # ZX Hand (idx=1) opens along column 1 (y).
        #
        # For idx=0: camera_up = object_x_axis        → column 0 = object_x
        # For idx=1: camera_up = cross(approach, obj_x) → column 1 = object_x
        #   (solving cross(-approach, camera_up) = object_x → camera_up = cross(approach, object_x))
        idx = self.grasp_open_axis_index
        if idx == 0:
            up_axis = object_x_axis
        else:
            up_axis = np.cross(approach_direction, object_x_axis)
            up_norm = np.linalg.norm(up_axis)
            if up_norm < 1e-8:
                # Degenerate: approach is parallel to object_x.
                # Pick any unit vector perpendicular to approach.
                if abs(approach_direction[2]) < 0.9:
                    up_axis = np.cross(approach_direction, np.array([0.0, 0.0, 1.0]))
                else:
                    up_axis = np.cross(approach_direction, np.array([1.0, 0.0, 0.0]))
                up_norm = np.linalg.norm(up_axis)
                if up_norm < 1e-8:
                    up_axis = np.array([0.0, 1.0, 0.0])
                else:
                    up_axis = up_axis / up_norm
            else:
                up_axis = up_axis / up_norm

        return construct_grasp_pose(adjusted_position, approach_direction, up_axis)

    @property
    def uses_adaptive_grasp(self) -> bool:
        """Whether this embodiment uses tactile-adaptive grasping by default."""
        return True

    def close_gripper_action(self, atom, rng=None) -> list:
        """Return the correct close-gripper Action list for this embodiment.

        Embodiments that need non-adaptive close with specific qpos ranges
        (e.g. GF225) override this.  Tasks call this instead of
        ``atom.close_gripper()`` directly.
        """
        return atom.close_gripper()

    def randomize_grasp_threshold(self, rng) -> float | None:
        """Return a randomized adaptive-grasp depth threshold, or None."""
        return None

    def _body_pose_in_root(self, body_idx: int) -> Pose:
        pw = self.articulation.data.body_link_pos_w[:, body_idx]
        qw = self.articulation.data.body_link_quat_w[:, body_idx]
        rp = self.articulation.data.root_link_pos_w
        rq = self.articulation.data.root_link_quat_w
        pb, qb = math_utils.subtract_frame_transforms(rp, rq, pw, qw)
        return Pose(pb[0].cpu().numpy(), qb[0].cpu().numpy())

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        if not hasattr(self, "origin_pose") or self.origin_pose is None:
            self.setup()
        joint_pos = self.articulation.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        if self.planner is not None:
            self.planner.reset()
        self.articulation.set_joint_position_target(joint_pos)
        self.articulation.write_joint_state_to_sim(joint_pos, joint_vel)
