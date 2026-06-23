"""
PokeCube task — adapted from ManiSkill ``PokeCube-v1``.

Task: Pick up a peg (tool) and use its far end to poke a cube onto
a target region.  Tool-mediated pushing — the peg acts as an extension.

Reference: https://github.com/haosulab/ManiSkill (Apache 2.0)
"""

from ._base_task import *
from ._registry import register_task
import numpy as np

HALF_SIZE = 0.02        # ManiSkill cube_half_size
PEG_HALF_LENGTH = 0.12  # ManiSkill peg_half_length
PEG_HALF_WIDTH = 0.025  # ManiSkill peg_half_width
GOAL_RADIUS = 0.05      # ManiSkill goal_radius


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 50
    adaptive_grasp_depth_threshold = 27.0
    manipulated_actor_name: str | None = "peg"


@register_task("poke_cube")
class Task(BaseTask):
    cfg: TaskCfg

    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect',
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        stash = Pose([-1.0, 0.0, HALF_SIZE + 0.001], [1, 0, 0, 0])
        self.cube = self._actor_manager.add_from_usd_file(
            name="cube", asset_path="Cube_Red.usda", pose=stash.clone(),
        )
        stash.p[1] = 1.0
        self.peg = self._actor_manager.add_from_usd_file(
            name="peg", asset_path="Peg.usda", pose=stash.clone(),
        )

    # ==================================================================
    #  Episode — ManiSkill PokeCube-v1 randomisation
    # ==================================================================

    def _reset_actors(self):
        rng = self.rng

        # Peg xy: uniform in [-0.1, 0.1]
        peg_xy = rng.uniform(-0.1, 0.1, size=2)
        # Peg in world frame (plate at x=0.5, plus 0.05 offset to keep out of
        # the way — ManiSkill uses world origin directly, we adjust for plate)
        peg_x = 0.45 + peg_xy[0]
        peg_y = peg_xy[1]

        peg_pose = Pose([peg_x, peg_y, PEG_HALF_WIDTH + 0.001], [1, 0, 0, 0])
        self.peg.set_pose(peg_pose)

        # Cube x = peg_x + PEG_HALF_LENGTH + 0.1; y = uniform in [-0.1, 0.1]
        cube_x = peg_x + PEG_HALF_LENGTH + 0.1
        cube_y = rng.uniform(-0.1, 0.1)
        cube_pose = Pose([cube_x, cube_y, HALF_SIZE + 0.001], [1, 0, 0, 0])
        cube_pose.q = t3d.euler.euler2quat(0, 0, rng.uniform(-np.pi / 6, np.pi / 6))
        self.cube.set_pose(cube_pose)

        # ManiSkill: target = cube_xy + [0.05 + goal_radius, 0]  (right of cube)
        self._target_xy = np.array([cube_x + 0.05 + GOAL_RADIUS, cube_y])

        self._last_cube_pose: Pose | None = None
        self._cube_static_counter: int = 0
        self.metadata["goal_radius"] = GOAL_RADIUS
        self.metadata["peg_half_length"] = PEG_HALF_LENGTH
        self.metadata["target_xy"] = self._target_xy.tolist()

        BaseTask._restore_primvar_color(self, self.cube, (0.9, 0.1, 0.1))

    # ==================================================================
    #  Expert demo — grasp peg, poke cube, release
    # ==================================================================

    def pre_move(self):
        self.delay(10)

        # --- 1. Open gripper ------------------------------------------------
        self.move(self.atom.open_gripper(1.0))

        # --- 2. Top-down grasp of peg at centre -----------------------------
        peg_pose = self.peg.get_pose()
        target_pose = peg_pose.add_bias([0, 0, PEG_HALF_WIDTH])

        camera_up = np.array([0, 1, 0]) if self.is_zxhand else np.array([1, 0, 0])
        cpose = construct_grasp_pose(
            target_pose.p,
            np.array([0, 0, 1]),          # top-down
            camera_up,
        )
        self.grasp_noise = self.create_noise(
            euler=[0, [-np.pi / 12, np.pi / 12], 0],
        )
        cpose = cpose.add_offset(self.grasp_noise)
        cid = self.peg.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.peg, contact_point_id=cid,
            pre_dis=0.0, dis=0.0, is_close=True,
        ))

    def _play_once(self):
        # --- 3. Lift to working height -------------------------------------
        self.move(self.atom.move_by_displacement(z=0.06))

        # --- 4. Position peg head just LEFT of the cube (relative, keeps orientation) -
        cube_xy = self.cube.get_pose().p[:2]
        ee = self._robot_manager.get_ee_pose()
        self.move(self.atom.move_by_displacement(
            x=cube_xy[0] - HALF_SIZE - 0.02 - PEG_HALF_LENGTH - ee[0],
            y=cube_xy[1] - ee[1],
            z=(PEG_HALF_WIDTH + 0.06) - ee[2],
        ))

        # --- 5. Poke: push forward so peg head drives cube to target --------
        self.move(self.atom.move_by_displacement(
            x=self._target_xy[0] + 0.02 - PEG_HALF_LENGTH
              - (cube_xy[0] - HALF_SIZE - 0.02 - PEG_HALF_LENGTH),
        ), time_dilation_factor=0.5)

        # --- 6. Release -----------------------------------------------------
        self.move(self.atom.open_gripper(1.0))
        self.move(self.atom.move_by_displacement(z=0.12))
        self.delay(30, is_save=False)

    # ==================================================================
    #  Success / early-stop — ManiSkill evaluate() semantics
    # ==================================================================

    def check_success(self):
        pos = self.cube.get_pose().p
        dist = float(np.linalg.norm(pos[:2] - self._target_xy))

        # ManiSkill: cube on target AND robot static
        xy_ok = dist < GOAL_RADIUS
        on_table = pos[2] < HALF_SIZE + 0.005
        return xy_ok and on_table

    def check_early_stop(self):
        if self.tactile_overpressed():
            self.metadata["early_stop"] = True
            self.metadata["min_depth"] = float(
                torch.min(self._tactile_manager.get_min_depth()).item(),
            )
            return True
        return False
