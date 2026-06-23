"""
LiftPegUpright task — adapted from ManiSkill ``LiftPegUpright-v1``.

Task: Move a peg from lying flat on the table (long axis along world X)
to any upright position (long axis ∥ world Z) and release on the table.

Reference: https://github.com/haosulab/ManiSkill (Apache 2.0)
"""

from ._base_task import *
from ._registry import register_task
import numpy as np

PEG_HALF_WIDTH = 0.025   # ManiSkill: peg_half_width
PEG_HALF_LENGTH = 0.12   # ManiSkill: peg_half_length


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 50
    adaptive_grasp_depth_threshold = 27.0
    manipulated_actor_name: str | None = "peg"


@register_task("lift_peg_upright")
class Task(BaseTask):
    cfg: TaskCfg

    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect',
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        stash = Pose([-1.0, 0.0, PEG_HALF_WIDTH + 0.001], [1, 0, 0, 0])
        self.peg = self._actor_manager.add_from_usd_file(
            name="peg", asset_path="Peg.usda", pose=stash.clone(),
        )

    # ==================================================================
    #  Episode — ManiSkill LiftPegUpright-v1 randomisation
    # ==================================================================

    def _reset_actors(self):
        rng = self.rng

        # ManiSkill: xy = rand(2)*0.2-0.1, body X along world X (lying flat)
        xy = rng.uniform(-0.1, 0.1, size=2)
        pose = Pose(
            [0.5 + xy[0], xy[1], PEG_HALF_WIDTH + 0.001],
            [1, 0, 0, 0],
        )
        # ManiSkill: euler2quat(pi/2, 0, 0) → long axis (body X) along world X
        pose.q = t3d.euler.euler2quat(np.pi / 2, 0, 0)
        self.peg.set_pose(pose)

        self.metadata["peg_half_length"] = PEG_HALF_LENGTH
        self.metadata["peg_half_width"] = PEG_HALF_WIDTH

    # ==================================================================
    #  Expert demo — grasp + rotate upright + release
    # ==================================================================

    def pre_move(self):
        self.delay(10)

        # --- 1. Open gripper ------------------------------------------------
        self.move(self.atom.open_gripper(1.0))

        # --- 2. Top-down grasp at peg centre --------------------------------
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
        self.metadata["grasp_noise"] = self.grasp_noise.tolist()
        cpose = cpose.add_offset(self.grasp_noise)

        cid = self.peg.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.peg, contact_point_id=cid,
            pre_dis=0.0, dis=0.0, is_close=True,
        ))
        self.origin_inhand_pose = self._robot_manager.get_inhand_pose(self.peg)

    def _play_once(self):
        # --- 3. Lift --------------------------------------------------------
        self.move(self.atom.move_by_displacement(z=0.15))

        # --- 4. Rotate peg upright ------------------------------------------
        # gravity_rotate uses tactile feedback to modulate grip force while
        # rotating the grasped object so its body-X axis aligns with [0,0,1].
        # This is the geometrically correct axis for a top-down → upright
        # reorientation (same pattern as lift_can).
        self.gravity_rotate(self.peg, [0, 0, 1], [1, 0, 0])

        # --- 5. Lower peg to table surface ----------------------------------
        # After rotation: peg long axis (0.12 half-length) is vertical.
        # Lower so the bottom end contacts the table.
        self.move(self.atom.move_by_displacement(z=-0.12))

        # --- 6. Release -----------------------------------------------------
        self.move(self.atom.open_gripper(1.0))
        self.move(self.atom.move_by_displacement(z=0.12))
        self.delay(30, is_save=False)

    # ==================================================================
    #  Success / early-stop
    # ==================================================================

    def check_success(self):
        """ManiSkill ``evaluate()`` semantics:

        The peg is upright if its body-X axis is nearly parallel to world Z,
        AND the peg centre is at half-length above the table (± 5 mm).
        """
        peg_mat = self.peg.get_pose().to_transformation_matrix()
        peg_body_x = peg_mat[:3, 0]          # local X axis in world frame
        world_z = np.array([0, 0, 1])

        # --- upright: body X dot world Z ≈ 1 (or -1 for inverted upright) ---
        upright = abs(np.dot(peg_body_x, world_z)) > 0.99

        # --- height: peg centre z ≈ half_length (± 5 mm) --------------------
        z_ok = abs(self.peg.get_pose().p[2] - PEG_HALF_LENGTH) < 0.005

        return bool(upright) and z_ok

    def check_early_stop(self):
        if self.tactile_overpressed():
            self.metadata["early_stop"] = True
            self.metadata["min_depth"] = float(
                torch.min(self._tactile_manager.get_min_depth()).item(),
            )
            return True
        return False
