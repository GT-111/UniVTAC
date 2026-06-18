import yaml
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass

from ..utils.transforms import *
from ..utils.atom import GRASP_DIRECTION_DIC
from .robot_cfg import RobotCfg
from .curobo_planner import CuroboPlanner, CuroboPlannerCfg
from .._global import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from curobo.wrap.reacher.motion_gen import MotionGenResult
    from .._base_task import BaseTask


class RobotManager:
    def __init__(self, robot_cfg:RobotCfg, task:'BaseTask', planner_time_dilation_factor:float=1.0):
        self.cfg = robot_cfg
        self.task = task
        self.device = task.device
        self.sensor_type = task.cfg.tactile_sensor_type
        if self.sensor_type in ['gsmini', 'gf225']:
            self.robot_type = 'franka_panda'
        elif self.sensor_type == 'zxhand':
            self.robot_type = 'franka_zx_hand'
        else:
            raise ValueError(f'Unknown tactile sensor type: {self.sensor_type}')

        self.robot = Articulation(self.cfg.robot)
        self.task.scene.articulations['robot'] = self.robot
        self.planner_time_dilation_factor = planner_time_dilation_factor

        self.gripper_max_qpos = 0.039
        self.last_arm_velocity = None
        self.last_gripper_velocity = None

        if self.robot_type == 'franka_panda':
            self.hand_name = 'panda_hand'
            self._arm_joint_names = [
                'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7'
            ]
            self._gripper_joint_names = [
                'panda_finger_joint1', 'panda_finger_joint2'
            ]
            self.gripper_max_qpos = self.cfg.gripper_max_qpos
            self.yaml_path = str(EMBODIMENTS_ROOT / 'franka' / 'curobo.yml')
            offset = self.cfg.gripper_offset
        elif self.robot_type == 'franka_zx_hand':
            # The ZX USD has no panda_hand link, so the ee pose comes from cuRobo FK
            # of panda_hand (see get_ee_pose); panda_link8 is only used to look up
            # the arm's last body index for the Jacobian.
            self.hand_name = 'panda_link8'
            self._arm_joint_names = [
                'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7'
            ]
            # Official example drives ONLY `right_Left_1_Joint`; the opposite
            # finger + link joints follow the closed-chain loop.
            self._gripper_joint_names = [
                'right_Left_1_Joint',
            ]
            self.gripper_max_qpos = 0.99  # rad, open pose (just under USD 1.0 cap)
            self.yaml_path = str(EMBODIMENTS_ROOT / 'franka' / 'curobo.yml')
            offset = self.cfg.gripper_offset  # placeholder; calibrated in setup()
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} not implemented.")
 
        # offset from ee (panda_hand) to gripper-center / grasp TCP
        self._offset = Pose(p=[0, 0, -offset], q=[1, 0, 0, 0])
        self._offset_pos = torch.tensor([0.0, 0.0, offset], device=self.device).repeat(self.task.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.task.num_envs, 1)

    def setup(self):
        """设置机器人属性"""
        body_ids, body_names = self.robot.find_bodies(self.hand_name)
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1

        joint_names = self.robot.joint_names
        self.joint_name_to_id = {name: i for i, name in enumerate(joint_names)}

        self._arm_ids = torch.tensor([
            self.joint_name_to_id[n] for n in self._arm_joint_names
        ], device=self.device)
        self._gripper_ids = torch.tensor([
            self.joint_name_to_id[n] for n in self._gripper_joint_names
        ], device=self.device)
        self._all_ids = torch.cat([self._arm_ids, self._gripper_ids], dim=0)

        if self.robot_type == 'franka_zx_hand':
            lf_ids, _ = self.robot.find_bodies('xense_leftfinger')
            rf_ids, _ = self.robot.find_bodies('xense_rightfinger')
            self._lf_idx = lf_ids[0]
            self._rf_idx = rf_ids[0]

            n_grip = self._gripper_ids.numel()
            # Match the USD joint limit [0, 1.0] rad for the driven finger joint.
            limits = torch.tensor([0.0, 1.0], device=self.device)
            limits = limits.view(1, 1, 2).repeat(self.task.num_envs, n_grip, 1)
            self.robot.write_joint_position_limit_to_sim(limits, joint_ids=self._gripper_ids)
            self.robot.write_joint_armature_to_sim(0.05, joint_ids=self._gripper_ids)

            # Ensure finger/linkage mesh prims have CollisionAPI so that
            # PhysX contact with UIPC objects is detected.
            self._ensure_finger_collision()

        self.root_pose = Pose.from_list(self.robot.data.root_link_pos_w[0])
        planner_cfg = CuroboPlannerCfg(
            dt=self.task.cfg.sim.dt,
            all_joints_name=self.robot.joint_names,
            active_joints_name=self._arm_joint_names,
            robot_prime_path=self.cfg.robot.prim_path,
            yaml_path=self.yaml_path
        )
        self.planner = CuroboPlanner(
            task=self.task,
            cfg=planner_cfg,
            robot_origin_pose=self.root_pose,
        )

        if self.robot_type == 'franka_zx_hand':
            self._offset = self._calibrate_zx_gripper_offset()

        self.origin_pose = self.get_gripper_center_pose()
    
    def _curobo_ee_in_root(self) -> Pose:
        """cuRobo FK panda_hand in articulation root frame (matches plan_arm targets)."""
        q = self.robot.data.joint_pos[0, self._arm_ids].unsqueeze(0).float()
        state = self.planner.motion_gen.kinematics.get_state(q)
        p = state.ee_position[0].detach().cpu().numpy()
        quat = state.ee_quaternion[0].detach().cpu().numpy()
        return Pose(p, quat)

    def _calibrate_zx_gripper_offset(self) -> Pose:
        """Calibrate ee(panda_hand) -> grasp TCP at the GEL CONTACT midpoint.

        The TCP must be the point that lands between the two gel pads, not the
        finger-body midpoint: the gel/camera surface is mounted ~3.7 cm off the
        finger body (ZX_CAM_LOCAL), so targeting the body midpoint leaves the
        object outside the gel faces (it closes on empty space). Measure the full
        3D offset (in the cuRobo panda_hand frame) at the open/grasp pose.
        """
        hand = self._curobo_ee_in_root()
        R_hand = hand.to_transformation_matrix()[:3, :3]
        offset_local = R_hand.T @ (self._gel_midpoint() - hand.p)
        return Pose(p=(-offset_local).tolist(), q=[1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _iter_descendants(prim):
        """Recursively yield all descendant prims (pxr.Usd has no GetDescendants)."""
        for child in prim.GetChildren():
            yield child
            yield from RobotManager._iter_descendants(child)

    def _ensure_finger_collision(self):
        """Apply CollisionAPI to every Mesh prim under finger body prims.

        The ZX USD may not include CollisionAPI on all meshes, and
        UsdFileCfg.collision_props only *modifies* existing collision — it
        never creates new CollisionAPI. Without collision geometry on the
        fingers, PhysX won't detect contact with UIPC objects.
        """
        from pxr import UsdPhysics
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
        except Exception:
            return  # Kit not running

        # prim_path may be a glob (e.g. "env_.*") — fix to concrete env_0
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

    def ee_to_gripper_center(self, ee_pose:Pose) -> Pose:
        """EE (panda_hand) -> grasp TCP."""
        return ee_pose.add_offset(self._offset.inv())

    def gripper_center_to_ee(self, gripper_center_pose:Pose) -> Pose:
        """Grasp TCP -> EE (panda_hand) for cuRobo."""
        return gripper_center_pose.add_offset(self._offset)
    
    def get_gripper_center_pose(self, env_ids:slice=None) -> Pose:
        """获取当前夹爪中心位姿"""
        return self.ee_to_gripper_center(self.get_ee_pose())
    
    def get_inhand_pose(self, actor:'Actor') -> Pose:
        return actor.get_pose().rebase(self.get_gripper_center_pose())
    
    def _body_pose_in_root(self, body_idx: int) -> Pose:
        pw = self.robot.data.body_link_pos_w[:, body_idx]
        qw = self.robot.data.body_link_quat_w[:, body_idx]
        rp = self.robot.data.root_link_pos_w
        rq = self.robot.data.root_link_quat_w
        pb, qb = math_utils.subtract_frame_transforms(rp, rq, pw, qw)
        return Pose(pb[0].cpu().numpy(), qb[0].cpu().numpy())

    # Finger-camera (gel surface) local mounts, from zx_official LEFT/RIGHT_CAM.
    _LF_CAM_LOCAL = np.array([0.0, 0.003, 0.037])
    _RF_CAM_LOCAL = np.array([0.0, -0.003, 0.037])

    def _gel_point(self, body_idx: int, local: np.ndarray) -> np.ndarray:
        pose = self._body_pose_in_root(body_idx)
        R = pose.to_transformation_matrix()[:3, :3]
        return pose.p + R @ np.asarray(local)

    def _gel_midpoint(self) -> np.ndarray:
        """Midpoint of the two gel contact surfaces (between the finger pads)."""
        gl = self._gel_point(self._lf_idx, self._LF_CAM_LOCAL)
        gr = self._gel_point(self._rf_idx, self._RF_CAM_LOCAL)
        return (gl + gr) / 2.0

    def get_ee_pose(self, env_ids:slice=None) -> Pose:
        """EE pose = cuRobo panda_hand (gsmini body / ZX cuRobo FK)."""
        if env_ids is None:
            env_ids = [0]
        if self.robot_type == 'franka_zx_hand':
            return self._curobo_ee_in_root()
        return self._body_pose_in_root(self._body_idx)

    def get_qpos(self):
        return self.robot.data.joint_pos.clone().cpu()
    
    def get_gripper_qpos(self):
        return self.get_qpos()[0, self._gripper_ids[0]].clone().cpu().item()
    def get_gripper_percentage(self):
        return self.get_gripper_qpos().item() / self.gripper_max_qpos

    def set_arm(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''设置目标位姿'''
        self.robot.set_joint_position_target(pos, joint_ids=self._arm_ids, env_ids=env_ids)
        if vel is not None:
            self.robot.set_joint_velocity_target(vel, joint_ids=self._arm_ids, env_ids=env_ids)
        if force:
            if self.robot_type == 'franka_zx_hand':
                # Arm-only teleport: snap ONLY the 7 arm joints to target via
                # write_joint_state_to_sim(joint_ids=arm). The whole-articulation
                # set_dof_positions would also reset the closed-chain gripper DOFs
                # every step and make the fingers go limp; pure PD on the arm
                # undershoots by several cm, so teleport just the arm.
                arm_pos = self.robot._data.joint_pos_target[:, self._arm_ids]
                arm_vel = self.robot.data.joint_vel[:, self._arm_ids]
                self.robot.write_joint_state_to_sim(
                    arm_pos, arm_vel, joint_ids=self._arm_ids
                )
                return
            self.robot.root_physx_view.set_dof_positions(
                self.robot._data.joint_pos_target,
                self.robot._ALL_INDICES
            )

    # ZX gripper velocity controller (official cust_gripper.ParallelGripper uses
    # joint-velocity actions). Pure bang-bang: constant +/-speed toward the
    # target angle, NEVER zeroed. The motor keeps pushing and stalls at the joint
    # limit (open) or on the object (closed); because the velocity target
    # persists across the following arm-only moves, this also HOLDS the gripper
    # open during the reach and HOLDS the grasp during the lift. Position-PD
    # cannot do this: the closed-chain loop equilibrium is "closed" and drags any
    # held position target shut.
    _ZX_GRIPPER_SPEED = 2.0   # rad/s

    def set_gripper(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''设置目标位姿'''
        if self.robot_type == 'franka_zx_hand':
            # Velocity sign takes priority when available: adaptive grasping
            # yields HIGH position targets (near 0.99, the open pose) while
            # still closing, so the old intent-based threshold would wrongly
            # push OPEN instead of CLOSE.  When vel is zero/None, fall back
            # to the intent-based position threshold.
            if vel is not None and torch.is_tensor(vel) and vel.numel() > 0:
                v_mean = vel.reshape(-1).float().mean().item()
                if abs(v_mean) > 1e-8:
                    speed = self._ZX_GRIPPER_SPEED
                    if v_mean < 0:
                        speed = -speed
                    v = torch.full(
                        (self.task.num_envs, self._gripper_ids.numel()),
                        speed, device=self.device,
                    )
                    self.robot.set_joint_velocity_target(v, joint_ids=self._gripper_ids, env_ids=env_ids)
                    return
            # Fallback: intent-based bang-bang from position threshold.
            target_val = float(torch.as_tensor(pos).reshape(-1)[0])
            speed = self._ZX_GRIPPER_SPEED
            if target_val <= 0.4 * self.gripper_max_qpos:
                speed = -speed
            v = torch.full(
                (self.task.num_envs, self._gripper_ids.numel()),
                speed, device=self.device,
            )
            self.robot.set_joint_velocity_target(v, joint_ids=self._gripper_ids, env_ids=env_ids)
            return
        self.robot.set_joint_position_target(pos, joint_ids=self._gripper_ids, env_ids=env_ids)
        if vel is not None:
            self.robot.set_joint_velocity_target(vel, joint_ids=self._gripper_ids, env_ids=env_ids)
        if force:
            self.robot.root_physx_view.set_dof_positions(
                self.robot._data.joint_pos_target,
                self.robot._ALL_INDICES
            )

    def plan_arm(self, target_pose:Pose, constraint_pose=None, pre_dis=None, time_dilation_factor=None):
        if time_dilation_factor is None:
            time_dilation_factor = self.planner_time_dilation_factor
        result:MotionGenResult = self.planner.plan_path(
            curr_joint_pos=self.robot.data.joint_pos[0, :self.robot.num_joints-2],
            curr_joint_vel=self.robot.data.joint_vel[0, :self.robot.num_joints-2],
            target_ee_pose=target_pose,
            real_robot_pose=self.root_pose,
            pre_dis=pre_dis,
            constraint_pose=constraint_pose,
            time_dilation_factor=time_dilation_factor,
        )
        
        if result.success.item():
            return {
                'status': 'Success',
                'num_steps': result.interpolated_plan.position.shape[0],
                'position': result.interpolated_plan.position.detach(),
                'velocity': result.interpolated_plan.velocity.detach()
            }
        else:
            return {'status': 'Fail', 'num_steps': 0, 'position': None, 'velocity': None}

    def gripper_percent2qpos(self, percentage:float):
        gripper_range = [0, self.gripper_max_qpos]
        target_pos = gripper_range[0] + (gripper_range[1] - gripper_range[0]) * percentage
        return target_pos

    def plan_gripper(self, pos:float, type:Literal['percent', 'qpos'] = 'percent'):
        if type == 'percent':
            target_pos = self.gripper_percent2qpos(pos)
        else:
            target_pos = pos
        gripper_pos = self.robot.data.joint_pos[0, self._gripper_ids][0]
        if self.robot_type == 'franka_zx_hand':
            # Velocity-controlled gripper (set_gripper reads the CONSTANT target to
            # pick open vs close direction). A linspace ramp would make the target
            # cross the open/close threshold mid-motion and flip direction, so
            # feed a constant target for a fixed window long enough to fully
            # open/close and stall on the object.
            num_steps = 80
            position = torch.full((num_steps,), float(target_pos), device=self.device)
            velocity = torch.zeros(num_steps, device=self.device)
            return {
                'status': 'Success',
                'num_steps': num_steps,
                'position': position.detach(),
                'velocity': velocity.detach(),
            }
        step = 0.0005
        num_steps = np.ceil(abs(target_pos - gripper_pos.cpu().item()) / step).astype(int)
        position = torch.linspace(gripper_pos, target_pos, num_steps, device=self.device)
        velocity = torch.clip((position - gripper_pos)/self.task.cfg.sim.dt, -0.0001, 0.0001)

        return {
            'status': 'Success',
            'num_steps': num_steps,
            'position': position.detach(),
            'velocity': velocity.detach()
        }

    def _reset_idx(self, env_ids: torch.Tensor | None=None):
        """重置环境"""
        if not hasattr(self, 'origin_pose'):
            self._setup_robot_properties()
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        
        self.planner.reset()
        self.robot.set_joint_position_target(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    def get_observations(self, data_type:list[str]=['joint', 'ee']) -> dict:
        obs = {}
        if 'ee' in data_type:
            obs['ee'] = self.get_ee_pose().totensor(device=self.device)
        if 'joint' in data_type:
            obs['joint'] = self.robot.data.joint_pos.squeeze(0)
        return obs
    
    def get_grasp_perfect_direction(self):
        return 'top_down'
