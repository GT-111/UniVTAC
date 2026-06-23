"""
PlaceSphere task — adapted from ManiSkill ``PlaceSphere-v1``.

Task: Pick up a sphere and place it into a shallow bin.
The sphere must rest on the bin surface and the gripper must release.

Reference: https://github.com/haosulab/ManiSkill (Apache 2.0)
"""

from ._base_task import *
from ._registry import register_task
import numpy as np

SPHERE_RADIUS = 0.02       # ManiSkill radius
BIN_HALF_XY = 0.025        # half-extent of the bin (5 cm square)
BIN_HALF_Z = 0.005         # half-height of the bin (1 cm thick)


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 50
    adaptive_grasp_depth_threshold = 27.0
    manipulated_actor_name: str | None = "sphere"


@register_task("place_sphere")
class Task(BaseTask):
    cfg: TaskCfg

    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect',
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        stash = Pose([-1.0, 0.0, SPHERE_RADIUS + 0.001], [1, 0, 0, 0])
        self.sphere = self._actor_manager.add_from_usd_file(
            name="sphere", asset_path="Sphere.usda", pose=stash.clone(),
        )
        stash.p[1] = 1.0
        self.bin = self._actor_manager.add_from_usd_file(
            name="bin", asset_path="Bin.usda", pose=stash.clone(),
        )

    # ==================================================================
    #  Episode — ManiSkill PlaceSphere-v1 randomisation
    # ==================================================================

    def _reset_actors(self):
        rng = self.rng

        # ManiSkill sphere: x∈[-0.1,-0.05], y∈[-0.1,0.1]
        sphere_x = rng.uniform(-0.1, -0.05)
        sphere_y = rng.uniform(-0.1, 0.1)
        sphere_pose = Pose(
            [0.5 + sphere_x, sphere_y, SPHERE_RADIUS + 0.001],
            [1, 0, 0, 0],
        )
        self.sphere.set_pose(sphere_pose)

        # ManiSkill bin: x∈[0,0.1], y∈[-0.1,0.1]
        bin_x = rng.uniform(0.0, 0.1)
        bin_y = rng.uniform(-0.1, 0.1)
        bin_pose = Pose(
            [0.5 + bin_x, bin_y, BIN_HALF_Z + 0.001],
            [1, 0, 0, 0],
        )
        self.bin.set_pose(bin_pose)

        self._last_sphere_pose: Pose | None = None
        self._sphere_static_counter: int = 0
        self.metadata["sphere_radius"] = SPHERE_RADIUS

    # ==================================================================
    #  Expert demo — grasp sphere, place in bin, release
    # ==================================================================

    def pre_move(self):
        self.delay(10)

        # --- 1. Open gripper ------------------------------------------------
        self.move(self.atom.open_gripper(1.0))

        # --- 2. Top-down grasp at sphere centre ----------------------------
        # Contact at sphere centre (same as insert_hole: grasp at actor centre).
        target_pose = self.sphere.get_pose()  # centre of the sphere

        camera_up = np.array([0, 1, 0]) if self.is_zxhand else np.array([1, 0, 0])
        cpose = construct_grasp_pose(
            target_pose.p,
            np.array([0, 0, 1]),          # top-down
            camera_up,
        )
        self.grasp_noise = self.create_noise(
            euler=[0, [-np.pi / 12, np.pi / 12], 0],
        )
        self.metadata["grasp_noise"] = self.grasp_noise.tolist()
        cpose = cpose.add_offset(self.grasp_noise)
        cid = self.sphere.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.sphere,
            pre_dis=0.0, dis=0.0,
            contact_point_id=cid,
        ))
        self.origin_inhand_pose = self.sphere.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        self.origin_inhand_pose = self._robot_manager.get_inhand_pose(self.sphere)

    def _play_once(self):
        # --- 3. Lift --------------------------------------------------------
        self.move(self.atom.move_by_displacement(z=0.10))

        # --- 4. Place sphere on bin -----------------------------------------
        bin_pose = self.bin.get_pose()
        place_z = BIN_HALF_Z + SPHERE_RADIUS + 0.001
        place_pose = Pose([bin_pose[0], bin_pose[1], place_z], [1, 0, 0, 0])
        self.move(self.atom.place_actor(
            self.sphere, place_pose,
            pre_dis=0.05, dis=0.003,
        ))

        # --- 5. Release -----------------------------------------------------
        self.move(self.atom.open_gripper(1.0))
        self.move(self.atom.move_by_displacement(z=0.12))
        self.delay(30, is_save=False)

    # ==================================================================
    #  Success — ManiSkill evaluate() semantics
    # ==================================================================

    def check_success(self):
        """ManiSkill: sphere on bin AND static AND not grasped."""
        posS = self.sphere.get_pose().p
        posB = self.bin.get_pose().p
        offset = posS - posB

        # --- on bin: xy centred, z at radius + bin_half_z (±5 mm) -----------
        xy_ok = float(np.linalg.norm(offset[:2])) <= 0.005
        z_ok = abs(offset[2] - SPHERE_RADIUS - BIN_HALF_Z) <= 0.005
        on_bin = xy_ok and z_ok

        if not on_bin:
            self._sphere_static_counter = 0
            return False

        # --- static (single observation, no hysteresis) --------------------
        cur_pose = self.sphere.get_pose()
        is_static = True
        if self._last_sphere_pose is not None:
            lin_vel = float(np.linalg.norm(cur_pose.p - self._last_sphere_pose.p))
            q_dot = float(np.abs(np.dot(cur_pose.q, self._last_sphere_pose.q)))
            ang_vel = 2.0 * np.arccos(min(q_dot, 1.0))
            is_static = lin_vel < 0.01 and ang_vel < 0.5
        self._last_sphere_pose = cur_pose

        if not is_static:
            return False

        # --- not grasped ----------------------------------------------------
        gripper_open = self._robot_manager.get_gripper_percentage() > 0.7
        inhand = self._robot_manager.get_inhand_pose(self.sphere)
        inhand_dis = float(np.linalg.norm(
            inhand.p - self.origin_inhand_pose.p,
        ))
        return gripper_open or inhand_dis > 0.03

    def check_early_stop(self):
        if self.tactile_overpressed():
            self.metadata["early_stop"] = True
            self.metadata["min_depth"] = float(
                torch.min(self._tactile_manager.get_min_depth()).item(),
            )
            return True
        return False
