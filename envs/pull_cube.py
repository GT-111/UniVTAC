"""
PullCube task — adapted from ManiSkill ``PullCube-v1``.

Task: Pull a cube onto a target region. Non-prehensile — closes gripper
on the far side of the cube and pulls toward the robot (-x direction).

Reference: https://github.com/haosulab/ManiSkill (Apache 2.0)
"""

from ._base_task import *
from ._registry import register_task
import numpy as np

HALF_SIZE = 0.02   # ManiSkill cube_half_size
GOAL_RADIUS = 0.1  # ManiSkill goal_radius


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 50
    adaptive_grasp_depth_threshold = 27.0
    manipulated_actor_name: str | None = "cube"


@register_task("pull_cube")
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

    # ==================================================================
    #  Episode — ManiSkill PullCube-v1 randomisation
    # ==================================================================

    def _reset_actors(self):
        rng = self.rng

        # ManiSkill: xy = rand(2)*0.2-0.1
        xy = rng.uniform(-0.1, 0.1, size=2)
        # Cube in world frame (plate at x=0.5)
        cube_x = 0.5 + xy[0]
        cube_y = xy[1]

        # ManiSkill: target = cube_xy - [0.1 + goal_radius, 0]  (to the left)
        self._target_xy = np.array([cube_x - 0.1 - GOAL_RADIUS, cube_y])

        pose = Pose([cube_x, cube_y, HALF_SIZE + 0.001], [1, 0, 0, 0])
        self.cube.set_pose(pose)

        self.metadata["goal_radius"] = GOAL_RADIUS
        self.metadata["target_xy"] = self._target_xy.tolist()

        BaseTask._restore_primvar_color(self, self.cube, (0.9, 0.1, 0.1))

    # ==================================================================
    #  Expert demo — move to far side, close, pull toward robot
    # ==================================================================

    def pre_move(self):
        self.delay(10)

        # Move to the far side of the cube (right side, away from robot)
        # ManiSkill: tcp_pull_pos = cube.p + [half_size + 2*0.005, 0, 0]
        cube_xy = self.cube.get_pose().p[:2]
        far_side = Pose(
            [cube_xy[0] + HALF_SIZE + 0.03, cube_xy[1], HALF_SIZE + 0.04],
            [1, 0, 0, 0],
        )
        self.move(self.atom.move_to_pose(far_side))

        # Close gripper partially to form a barrier behind the cube
        self.move(self.atom.close_gripper(0.3))

    def _play_once(self):
        # Pull backward toward the target (-x direction)
        pull_pose = Pose(
            [self._target_xy[0] - 0.03, self._target_xy[1], HALF_SIZE + 0.04],
            [1, 0, 0, 0],
        )
        self.move(self.atom.move_to_pose(pull_pose), time_dilation_factor=0.5)

        self.move(self.atom.open_gripper(1.0))
        self.move(self.atom.move_by_displacement(z=0.12))
        self.delay(30, is_save=False)

    # ==================================================================
    #  Success — ManiSkill evaluate() semantics
    # ==================================================================

    def check_success(self):
        pos = self.cube.get_pose().p
        dist = float(np.linalg.norm(pos[:2] - self._target_xy))

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
