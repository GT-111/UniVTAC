"""Franka + GelSight Mini / GF225 — UIPC soft-body tactile simulation.

Prismatic gripper (0 – 0.039 m), PD position control, end-effector pose
from Isaac Sim body state.
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Robot, RobotConfig
from .registry import register_robot
from .planner import PlannerResult
from ..utils.transforms import Pose
from .._global import EMBODIMENTS_ROOT

if TYPE_CHECKING:
    from curobo.wrap.reacher.motion_gen import MotionGenResult


@register_robot("franka_gsmini")
@register_robot("franka_gf225")
class FrankaGelSight(Robot):
    """Franka arm with GelSight Mini or GF225 gripper."""

    hand_name = "panda_hand"
    arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    yaml_path = str(EMBODIMENTS_ROOT / "franka" / "curobo.yml")

    def __init__(self, cfg: RobotConfig, task):
        super().__init__(cfg, task)
        self.gripper_max_qpos = cfg.gripper_max_qpos
        # Distinguish gsmini from gf225 for close_gripper_action
        self._sensor_variant = task.cfg.tactile_sensor_type  # 'gsmini' or 'gf225'

    @property
    def uses_adaptive_grasp(self) -> bool:
        return self._sensor_variant != "gf225"

    def close_gripper_action(self, atom, rng=None) -> list:
        if self._sensor_variant == "gf225":
            qpos = rng.uniform(0.0118, 0.013) / self.gripper_max_qpos
            return atom.close_gripper(pos=qpos, depth_threshold=None)
        return atom.close_gripper()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        self._map_joint_ids()

        # Map body index for panda_hand
        body_ids, body_names = self.articulation.find_bodies(self.hand_name)
        self._body_idx: int = body_ids[0]
        self._body_name: str = body_names[0]
        self._jacobi_body_idx: int = self._body_idx - 1

        # Store root pose
        self.root_pose = Pose.from_list(
            self.articulation.data.root_link_pos_w[0]
        )

        # Initialise cuRobo planner
        from .curobo_planner import CuroboPlanner, CuroboPlannerCfg

        planner_cfg = CuroboPlannerCfg(
            dt=self.task.cfg.sim.dt,
            all_joints_name=self.articulation.joint_names,
            active_joints_name=self.arm_joint_names,
            robot_prime_path=self.cfg.robot.prim_path,
            yaml_path=self.yaml_path,
        )
        self.planner = CuroboPlanner(
            task=self.task,
            cfg=planner_cfg,
            robot_origin_pose=self.root_pose,
        )
        self.origin_pose = self.get_gripper_center_pose()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_ee_pose(self, env_ids: slice | None = None) -> Pose:
        """EE from Isaac Sim body state (fast, no FK)."""
        if env_ids is None:
            env_ids = [0]
        return self._body_pose_in_root(self._body_idx)

    # ------------------------------------------------------------------
    # Low-level control
    # ------------------------------------------------------------------

    def set_arm(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor | None = None,
        env_ids: slice | None = None,
        force: bool = True,
    ) -> None:
        self.articulation.set_joint_position_target(pos, joint_ids=self._arm_ids, env_ids=env_ids)
        if vel is not None:
            self.articulation.set_joint_velocity_target(vel, joint_ids=self._arm_ids, env_ids=env_ids)
        if force:
            self.articulation.root_physx_view.set_dof_positions(
                self.articulation._data.joint_pos_target,
                self.articulation._ALL_INDICES,
            )

    def set_gripper(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor | None = None,
        env_ids: slice | None = None,
        force: bool = True,
    ) -> None:
        self.articulation.set_joint_position_target(pos, joint_ids=self._gripper_ids, env_ids=env_ids)
        if vel is not None:
            self.articulation.set_joint_velocity_target(vel, joint_ids=self._gripper_ids, env_ids=env_ids)
        if force:
            self.articulation.root_physx_view.set_dof_positions(
                self.articulation._data.joint_pos_target,
                self.articulation._ALL_INDICES,
            )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan_arm(
        self,
        target_pose: Pose,
        constraint_pose=None,
        pre_dis: float | None = None,
        time_dilation_factor: float | None = None,
    ) -> dict:
        if time_dilation_factor is None:
            time_dilation_factor = self.cfg.planner_time_dilation_factor
        result: "MotionGenResult" = self.planner.plan_path(
            curr_joint_pos=self.articulation.data.joint_pos[
                0, : self.articulation.num_joints - 2
            ],
            curr_joint_vel=self.articulation.data.joint_vel[
                0, : self.articulation.num_joints - 2
            ],
            target_ee_pose=target_pose,
            real_robot_pose=self.root_pose,
            pre_dis=pre_dis,
            constraint_pose=constraint_pose,
            time_dilation_factor=time_dilation_factor,
        )
        if result.success.item():
            return {
                "status": "Success",
                "num_steps": result.interpolated_plan.position.shape[0],
                "position": result.interpolated_plan.position.detach(),
                "velocity": result.interpolated_plan.velocity.detach(),
            }
        return {"status": "Fail", "num_steps": 0, "position": None, "velocity": None}

    def plan_gripper(self, pos: float, type: str = "percent") -> dict:
        if type == "percent":
            target_pos = self.gripper_percent_to_qpos(pos)
        else:
            target_pos = pos

        gripper_pos = self.articulation.data.joint_pos[0, self._gripper_ids][0]
        step = 0.0005
        num_steps = int(np.ceil(abs(target_pos - gripper_pos.cpu().item()) / step))
        position = torch.linspace(gripper_pos, target_pos, num_steps, device=self.device)
        velocity = torch.clip(
            (position - gripper_pos) / self.task.cfg.sim.dt, -0.0001, 0.0001
        )
        return {
            "status": "Success",
            "num_steps": num_steps,
            "position": position.detach(),
            "velocity": velocity.detach(),
        }

    # ------------------------------------------------------------------
    # Tactile-aware
    # ------------------------------------------------------------------

    def is_overpressed(self, min_depth: float) -> bool:
        return min_depth < self.cfg.contact_threshold[0]
