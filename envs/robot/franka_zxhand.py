"""Franka + Xense ZX Hand — rigid-finger tactile with xensim FemSensor.

Angular gripper (0 – 0.99 rad), velocity bang-bang control, end-effector
pose from cuRobo forward kinematics (no panda_hand body exists in the ZX USD).
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from .base import Robot, RobotConfig
from .registry import register_robot
from ..utils.transforms import Pose
from .._global import EMBODIMENTS_ROOT

if TYPE_CHECKING:
    from curobo.wrap.reacher.motion_gen import MotionGenResult


@register_robot("franka_zxhand")
class FrankaZXHand(Robot):
    """Franka arm with Xense ZX Hand gripper."""

    hand_name = "panda_link8"
    arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    gripper_joint_names = ["right_Left_1_Joint"]
    yaml_path = str(EMBODIMENTS_ROOT / "franka" / "curobo.yml")
    _ZX_GRIPPER_SPEED = 2.0  # rad/s (constant velocity drive — matches pre-refactor behavior)

    # Finger-camera (gel surface) local mounts, from zx_official LEFT/RIGHT_CAM
    _LF_CAM_LOCAL = np.array([0.0, 0.003, 0.037])
    _RF_CAM_LOCAL = np.array([0.0, -0.003, 0.037])

    def __init__(self, cfg: RobotConfig, task):
        super().__init__(cfg, task)
        self.gripper_max_qpos = cfg.gripper_max_qpos  # rad
        self._lf_idx: int = 0
        self._rf_idx: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        self._map_joint_ids()

        # Map body index for panda_link8
        body_ids, body_names = self.articulation.find_bodies(self.hand_name)
        self._body_idx: int = body_ids[0]
        self._body_name: str = body_names[0]
        self._jacobi_body_idx: int = self._body_idx - 1

        # Map finger body indices (for gel midpoint calibration)
        lf_ids, _ = self.articulation.find_bodies("xense_leftfinger")
        rf_ids, _ = self.articulation.find_bodies("xense_rightfinger")
        self._lf_idx = int(lf_ids[0])
        self._rf_idx = int(rf_ids[0])

        # Set joint limits and armature for the driven finger joint
        n_grip = self._gripper_ids.numel()
        limits = torch.tensor([0.0, 1.0], device=self.device)
        limits = limits.view(1, 1, 2).repeat(self.task.num_envs, n_grip, 1)
        self.articulation.write_joint_position_limit_to_sim(limits, joint_ids=self._gripper_ids)
        self.articulation.write_joint_armature_to_sim(0.05, joint_ids=self._gripper_ids)

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

        # Calibrate ZX-specific gripper offset
        self._offset = self._calibrate_zx_gripper_offset()
        self.origin_pose = self.get_gripper_center_pose()

        # NOTE: _ensure_finger_collision() is DISABLED — adding CollisionAPI
        # to finger meshes with enabled_self_collisions=True on the 4-bar
        # closed-chain linkage creates complex contact pairs that hang PhysX.
        # The ZX USD already has collision geometry on finger bodies; if
        # fingers pass through objects, the issue is elsewhere (UIPC sync,
        # solver_velocity_iterations, or position teleport).
        # self._ensure_finger_collision()

    @staticmethod
    def _iter_descendants(prim):
        """Recursively yield all descendant prims (pxr.Usd has no GetDescendants)."""
        for child in prim.GetChildren():
            yield child
            yield from FrankaZXHand._iter_descendants(child)

    def _ensure_finger_collision(self):
        """Apply CollisionAPI to every Mesh prim under finger body prims."""
        from pxr import UsdPhysics
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
        except Exception:
            return  # Kit not running

        # prim_path may be a glob (e.g. "env_.*") — fix to concrete env_0
        # to match OfficialZXTactileSensor.from_cfg pattern.
        base_path = self.cfg.robot.prim_path
        if "env_.*" in base_path:
            base_path = base_path.replace("env_.*", "env_0")

        for finger_name in ("xense_leftfinger", "xense_rightfinger"):
            finger_prim_path = f"{base_path}/{finger_name}"
            finger_prim = stage.GetPrimAtPath(finger_prim_path)
            if not finger_prim.IsValid():
                continue
            for mesh_prim in self._iter_descendants(finger_prim):
                if mesh_prim.GetTypeName() != "Mesh":
                    continue
                if not UsdPhysics.CollisionAPI(mesh_prim):
                    UsdPhysics.CollisionAPI.Apply(mesh_prim)

    # ------------------------------------------------------------------
    # cuRobo FK helpers (ZX has no panda_hand body)
    # ------------------------------------------------------------------

    def _curobo_ee_in_root(self) -> Pose:
        """cuRobo FK of panda_hand in articulation root frame."""
        q = self.articulation.data.joint_pos[0, self._arm_ids].unsqueeze(0).float()
        state = self.planner.motion_gen.kinematics.get_state(q)
        p = state.ee_position[0].detach().cpu().numpy()
        quat = state.ee_quaternion[0].detach().cpu().numpy()
        return Pose(p, quat)

    def _gel_point(self, body_idx: int, local: np.ndarray) -> np.ndarray:
        pose = self._body_pose_in_root(body_idx)
        R = pose.to_transformation_matrix()[:3, :3]
        return pose.p + R @ np.asarray(local)

    def _gel_midpoint(self) -> np.ndarray:
        """Midpoint of the two gel contact surfaces (between finger pads)."""
        gl = self._gel_point(self._lf_idx, self._LF_CAM_LOCAL)
        gr = self._gel_point(self._rf_idx, self._RF_CAM_LOCAL)
        return (gl + gr) / 2.0

    def _calibrate_zx_gripper_offset(self) -> Pose:
        """Calibrate ee(panda_hand) → grasp TCP at the gel contact midpoint."""
        hand = self._curobo_ee_in_root()
        R_hand = hand.to_transformation_matrix()[:3, :3]
        offset_local = R_hand.T @ (self._gel_midpoint() - hand.p)
        return Pose(p=(-offset_local).tolist(), q=[1.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_ee_pose(self, env_ids: slice | None = None) -> Pose:
        """EE from cuRobo FK — no panda_hand body in ZX USD."""
        return self._curobo_ee_in_root()

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
            # Arm-only teleport: snap ONLY the 7 arm joints.  Whole-articulation
            # set_dof_positions would also reset closed-chain gripper DOFs.
            arm_pos = self.articulation._data.joint_pos_target[:, self._arm_ids]
            arm_vel = self.articulation.data.joint_vel[:, self._arm_ids]
            self.articulation.write_joint_state_to_sim(
                arm_pos, arm_vel, joint_ids=self._arm_ids
            )

    def set_gripper(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor | None = None,
        env_ids: slice | None = None,
        force: bool = True,
    ) -> None:
        """ZX gripper: pure velocity bang-bang drive.

        When ``vel`` is non-zero, its sign determines the direction (negative
        → close, positive → open).  This properly handles adaptive grasping
        where the *position* target starts HIGH (near 0.99, the open pose)
        even while closing — the old intent-based threshold would then
        misinterpret ``close`` as ``open`` and push the wrong way.

        When ``vel`` is zero or None, falls back to intent-based threshold
        (position <= 40 % of max → close; otherwise open).

        ``force`` is accepted for API compatibility but not used.
        """
        n_grip = self._gripper_ids.numel()
        p = torch.as_tensor(pos, device=self.device).reshape(-1)
        if p.numel() == 1:
            p = p.repeat(n_grip)
        elif p.numel() > n_grip:
            p = p[:n_grip]
        p = torch.clamp(p, 0.0, self.gripper_max_qpos)

        # Determine close/open direction: prefer velocity sign when available.
        if vel is not None and torch.is_tensor(vel) and vel.numel() > 0:
            v_mean = vel.reshape(-1).float().mean().item()
            if abs(v_mean) > 1e-8:
                speed = self._ZX_GRIPPER_SPEED
                if v_mean < 0:
                    speed = -speed
                v = torch.full(
                    (self.task.num_envs, n_grip), speed, device=self.device,
                )
                self.articulation.set_joint_velocity_target(
                    v, joint_ids=self._gripper_ids, env_ids=env_ids
                )
                return

        # Fallback: intent-based bang-bang from position threshold.
        target_val = float(p.reshape(-1)[0].item())
        speed = self._ZX_GRIPPER_SPEED
        if target_val <= 0.4 * self.gripper_max_qpos:
            speed = -speed

        v = torch.full(
            (self.task.num_envs, n_grip), speed, device=self.device,
        )
        self.articulation.set_joint_velocity_target(
            v, joint_ids=self._gripper_ids, env_ids=env_ids
        )
        return

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

        # Velocity-controlled gripper: feed a constant target for a fixed window
        # long enough to fully open/close and stall on the object.
        # At 2.0 rad/s and 120 Hz, traversing 0→0.99 rad takes ~60 steps;
        # 120 steps gives ~2x margin to push through mechanical resistance.
        num_steps = 120
        position = torch.full((num_steps,), float(target_pos), device=self.device)
        velocity = torch.zeros(num_steps, device=self.device)
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
        # ZX processed camera depth: rest ≈ +1 mm, full indentation ≈ -4 mm
        return min_depth < -3.8

    # ------------------------------------------------------------------
    # Embodiment-specific properties
    # ------------------------------------------------------------------

    @property
    def adaptive_grasp_step_coarse(self) -> float:
        return 0.016  # rad (~2.0 rad/s at 120Hz, matches velocity drive speed)

    @property
    def adaptive_grasp_step_fine(self) -> float:
        return 0.002  # rad

    @property
    def needs_post_settle(self) -> bool:
        return True

    @property
    def uses_adaptive_grasp(self) -> bool:
        # ZX velocity drive at 1.0 rad/s moves ~0.008 rad/step.  The adaptive
        # generator's step (0.003 rad) is smaller than this, causing overshoot
        # and velocity oscillation.  Use plan_gripper (fixed-duration velocity
        # ramp) instead of the adaptive position-based generator.
        return False

    @property
    def grasp_open_axis_index(self) -> int:
        # ZX hand fingers open along a different axis than gsmini panda fingers
        return 1

    @property
    def grasp_height_clearance(self) -> float:
        # ZX fingers rotate closed → the gel-midpoint (effective grasp point)
        # shifts DOWNWARD relative to the open-planning pose.  The grasp
        # target must be raised to compensate so the gel pads land on the
        # object AFTER the close, not before.
        return 0.02

    @property
    def grasp_pre_dis_extra(self) -> float:
        # Longer pre-grasp distance to start the approach from higher up.
        return 0.01
