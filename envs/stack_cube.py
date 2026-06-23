"""
StackCube task — adapted from ManiSkill ``StackCube-v1``.

Task: Pick up a red cube and stack it on top of a green cube, then release
without the top cube falling.

Reference: https://github.com/haosulab/ManiSkill (Apache 2.0)
"""

from ._base_task import *
from ._registry import register_task
import numpy as np

HALF_SIZE = 0.02  # cube half-extent, matching ManiSkill StackCube-v1


# ---------------------------------------------------------------------------
#  Collision-avoidance helpers
# ---------------------------------------------------------------------------

def _cube_center_distance(H: float) -> float:
    """Minimum centre-to-centre distance so two cubes of half-size *H* never
    overlap (diagonal of the top face + 3 mm clearance)."""
    return 2 * np.sqrt(2) * H + 0.003


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 50
    adaptive_grasp_depth_threshold = 27.0
    manipulated_actor_name: str | None = "cubeA"


# ---------------------------------------------------------------------------
#  Task
# ---------------------------------------------------------------------------

@register_task("stack_cube")
class Task(BaseTask):
    cfg: TaskCfg

    def __init__(self, cfg: BaseTaskCfg, mode: Literal['collect', 'eval'] = 'collect',
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, mode, render_mode, **kwargs)

    # ==================================================================
    #  Scene setup
    # ==================================================================

    def create_actors(self):
        stash = Pose([-1.0, 0.0, HALF_SIZE + 0.001], [1, 0, 0, 0])
        self.cubeA = self._actor_manager.add_from_usd_file(
            name="cubeA", asset_path="Cube_Red.usda", pose=stash.clone(),
        )
        stash.p[1] = 1.0
        self.cubeB = self._actor_manager.add_from_usd_file(
            name="cubeB", asset_path="Cube_Green.usda", pose=stash.clone(),
        )

    # ==================================================================
    #  Episode initialisation
    # ==================================================================

    def _reset_actors(self):
        """Randomise cube positions, ManiSkill ``StackCube-v1`` style."""
        min_dist = _cube_center_distance(HALF_SIZE)
        rng = self.rng

        # --- shared base xy (ManiSkill: xy = rand(2)*0.2 - 0.1) -------------
        base_xy = rng.uniform(-0.1, 0.1, size=2)

        # --- collision-avoided offsets (rejection sampling) -------------------
        for _ in range(200):
            off_a = rng.uniform([-0.05, -0.07], [0.05, 0.07])
            off_b = rng.uniform([-0.05, -0.07], [0.05, 0.07])
            if np.linalg.norm(off_a - off_b) >= min_dist:
                break

        # --- red cube (cubeA) — centred at x~0.4 (insert_hole working zone) --
        cubeA_pose = Pose(
            [0.40 + base_xy[0] + off_a[0], base_xy[1] + off_a[1],
             HALF_SIZE],
            [1, 0, 0, 0],
        )
        cubeA_pose.q = t3d.euler.euler2quat(0, 0, rng.uniform(-np.pi, np.pi))
        self.cubeA.set_pose(cubeA_pose)

        # --- green cube (cubeB) — further right ------------------------------
        cubeB_pose = Pose(
            [0.50 + base_xy[0] + off_b[0], base_xy[1] + off_b[1],
             HALF_SIZE],
            [1, 0, 0, 0],
        )
        cubeB_pose.q = t3d.euler.euler2quat(0, 0, rng.uniform(-np.pi, np.pi))
        self.cubeB.set_pose(cubeB_pose)

        # --- success / early-stop tracking ------------------------------------
        self._last_cubeA_pose: Pose | None = None
        self._cubeA_static_counter: int = 0
        self.metadata["cube_half_size"] = HALF_SIZE

        # Restore USD display colours (tet-mesh generation may randomise them).
        BaseTask._restore_primvar_color(self, self.cubeA, (0.9, 0.1, 0.1))
        BaseTask._restore_primvar_color(self, self.cubeB, (0.1, 0.8, 0.1))

    # ==================================================================
    #  Expert demonstration — approach & grasp (called during reset)
    # ==================================================================

    def pre_move(self):
        self.delay(10)

        # --- 1. Open gripper -------------------------------------------------
        self.move(self.atom.open_gripper(1.0))

        # --- 2. Top-down grasp of cubeA at centre ---------------------------
        # Contact point at cube centre (same pattern as insert_hole: grasp at
        # actor centre, not top surface).  Adaptive grasp closes until gel
        # pads make firm contact, giving ~4cm grip on each side.
        grasp_bias = 0.0  # centre of the 4 cm cube
        target_pose = self.cubeA.get_pose().add_bias([0, 0, grasp_bias])

        camera_up = np.array([0, 1, 0]) if self.is_zxhand else np.array([1, 0, 0])
        cpose = construct_grasp_pose(
            target_pose.p,
            [0, 0, 1],
            camera_up,
        )
        self.cid = self.cubeA.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.cubeA,
            pre_dis=0.0, dis=0.0,
            contact_point_id=self.cid,
        ))
        self.origin_inhand_pose = self.cubeA.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())

    # ==================================================================
    #  Expert demonstration — lift, move, place, release
    # ==================================================================

    def _play_once(self):
        # --- 3. Lift cubeA straight up -------------------------------------
        self.move(self.atom.move_by_displacement(z=0.12))

        # --- 4. Place cubeA on top of cubeB --------------------------------
        cubeB_pose = self.cubeB.get_pose()
        place_z = cubeB_pose[2] + 2 * HALF_SIZE + 0.001
        place_pose = Pose([cubeB_pose[0], cubeB_pose[1], place_z], [1, 0, 0, 0])
        self.move(self.atom.place_actor(
            self.cubeA, place_pose,
            pre_dis=0.05, dis=0.003,
        ))

        # --- 5. Release & back off -----------------------------------------
        self.move(self.atom.open_gripper(1.0))
        self.move(self.atom.move_by_displacement(z=0.12))
        self.delay(30, is_save=False)

    # ==================================================================
    #  Success / early-stop
    # ==================================================================

    def check_success(self):
        """ManiSkill ``evaluate()`` semantics:

        1. **is_cubeA_on_cubeB** — xy within diag-half-size + 5 mm, z at 2*half-size ± 5 mm.
        2. **is_cubeA_static** — 3-frame hysteresis, lin < 1 cm/step, ang < 0.5 rad/step.
        3. **~is_grasped** — gripper open or cube drifted from grasp.
        """
        posA = self.cubeA.get_pose().p
        posB = self.cubeB.get_pose().p
        offset = posA - posB

        # --- 1. on-top -------------------------------------------------------
        # ManiSkill: norm(offset[:2]) <= norm([H,H]) + 0.005
        xy_ok = float(np.linalg.norm(offset[:2])) <= np.sqrt(2) * HALF_SIZE + 0.005
        z_ok = abs(offset[2] - 2 * HALF_SIZE) <= 0.005
        if not (xy_ok and z_ok):
            self._cubeA_static_counter = 0
            return False

        # --- 2. static check (frame-difference, single observation) ---------
        cur_pose = self.cubeA.get_pose()
        is_static = True  # assume static in absence of evidence
        if self._last_cubeA_pose is not None:
            lin_vel = float(np.linalg.norm(cur_pose.p - self._last_cubeA_pose.p))
            q_dot = float(np.abs(np.dot(cur_pose.q, self._last_cubeA_pose.q)))
            ang_vel = 2.0 * np.arccos(min(q_dot, 1.0))
            is_static = lin_vel < 0.01 and ang_vel < 0.5
        self._last_cubeA_pose = cur_pose

        if not is_static:
            return False

        # --- 3. not grasped --------------------------------------------------
        gripper_open = self._robot_manager.get_gripper_percentage() > 0.7
        inhand = self._robot_manager.get_inhand_pose(self.cubeA)
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

        cubeA_pose = self.cubeA.get_pose()
        inhand_pose = self._robot_manager.get_inhand_pose(self.cubeA)
        inhand_dis = float(np.linalg.norm(
            inhand_pose.p - self.origin_inhand_pose.p,
        ))
        cube_z_up = abs(np.dot(
            cubeA_pose.to_transformation_matrix()[:3, 2],
            np.array([0, 0, 1]),
        ))
        if inhand_dis > 0.05 and cube_z_up > 0.95:
            self.metadata["early_stop"] = True
            self.metadata["inhand_dis"] = inhand_dis
            return True
        return False
