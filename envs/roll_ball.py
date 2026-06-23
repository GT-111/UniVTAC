"""
RollBall task — adapted from ManiSkill ``RollBall-v1``.

Task: Push a ball across the table to a goal region at the far end.
Dynamic rolling contact — the ball can roll freely after being pushed.

Reference: https://github.com/haosulab/ManiSkill (Apache 2.0)
"""

from ._base_task import *
from ._registry import register_task
import numpy as np

BALL_RADIUS = 0.035   # ManiSkill ball_radius
GOAL_RADIUS = 0.1     # ManiSkill goal_radius


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 80
    adaptive_grasp_depth_threshold = 27.0
    manipulated_actor_name: str | None = "ball"


@register_task("roll_ball")
class Task(BaseTask):
    cfg: TaskCfg

    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect',
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        stash = Pose([-1.0, 0.0, BALL_RADIUS + 0.001], [1, 0, 0, 0])
        self.ball = self._actor_manager.add_from_usd_file(
            name="ball", asset_path="Ball.usda", pose=stash.clone(),
        )

    # ==================================================================
    #  Episode — ManiSkill RollBall-v1 randomisation
    # ==================================================================

    def _reset_actors(self):
        rng = self.rng

        # ManiSkill ball: x∈[-0.4,0.2], y∈[0.5,0.7] — near robot side
        ball_x = rng.uniform(-0.4, 0.2)
        ball_y = rng.uniform(0.5, 0.7)
        # Map to UniVTAC world frame (plate x-offset 0.5)
        ball_pose = Pose(
            [0.5 + ball_x, ball_y, BALL_RADIUS + 0.001],
            [1, 0, 0, 0],
        )
        self.ball.set_pose(ball_pose)

        # ManiSkill target: x∈[-0.4,0.2], y∈[-0.9,-0.7] — far side
        target_x = rng.uniform(-0.4, 0.2)
        target_y = rng.uniform(-0.9, -0.7)
        self._target_xy = np.array([0.5 + target_x, target_y])

        self.metadata["ball_radius"] = BALL_RADIUS
        self.metadata["goal_radius"] = GOAL_RADIUS
        self.metadata["target_xy"] = self._target_xy.tolist()

    # ==================================================================
    #  Expert demo — approach behind ball, push toward target
    # ==================================================================

    def pre_move(self):
        self.delay(10)

        # Close gripper partially to form a pushing surface
        self.move(self.atom.close_gripper(0.3))

        # Approach from behind the ball (from the side closer to robot),
        # pushing in the direction of the target.
        ball_xy = self.ball.get_pose().p[:2]
        # Unit vector from ball toward target — push from opposite side
        to_target = self._target_xy - ball_xy
        to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-8)
        behind_xy = ball_xy - to_target_norm * (BALL_RADIUS + 0.05)

        behind_pose = Pose(
            [behind_xy[0], behind_xy[1], BALL_RADIUS + 0.02],
            [1, 0, 0, 0],
        )
        self.move(self.atom.move_to_pose(behind_pose))

    def _play_once(self):
        # Push ball toward the target
        to_target = self._target_xy - self.ball.get_pose().p[:2]
        to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-8)
        push_xy = self._target_xy + to_target_norm * 0.03

        push_pose = Pose(
            [push_xy[0], push_xy[1], BALL_RADIUS + 0.02],
            [1, 0, 0, 0],
        )
        self.move(self.atom.move_to_pose(push_pose), time_dilation_factor=0.3)

        # Release
        self.move(self.atom.open_gripper(1.0))
        self.move(self.atom.move_by_displacement(z=0.12))
        self.delay(40, is_save=False)

    # ==================================================================
    #  Success — ManiSkill evaluate() semantics
    # ==================================================================

    def check_success(self):
        pos = self.ball.get_pose().p
        dist = float(np.linalg.norm(pos[:2] - self._target_xy))
        return dist < GOAL_RADIUS

    def check_early_stop(self):
        if self.tactile_overpressed():
            self.metadata["early_stop"] = True
            self.metadata["min_depth"] = float(
                torch.min(self._tactile_manager.get_min_depth()).item(),
            )
            return True
        return False
