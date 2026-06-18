from ._base_task import *
import numpy as np

@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.74, 0.0, 0.066), rot=(0.512, 0.512, 0.487, 0.487), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None, # use existing camera
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    step_lim = 600
    planner_time_dilation_factor: float = 0.35

class Task(BaseTask):
    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        base_pose = Pose([0.55, 0.0, 0.002], [1, 0, 0, 0])
        prism_pose = Pose([0.4, 0.0, 0.002], [1, 0, 0, 0])

        self.slot = self._actor_manager.add_from_usd_file(
            name='slot',
            asset_path="HDMISlot.usd",
            pose=base_pose,
            density=1e5
        )

        self.prism = self._actor_manager.add_from_usd_file(
            name='prism',
            asset_path="HDMI.usd",
            pose=prism_pose
        )
    
    def _reset_actors(self):
        base_offset = self.create_noise([0.005, 0.005, 0.0])
        base_pose = Pose([0.55, 0.0, self.slot.get_pose()[2]], [1, 0, 0, 0]).add_offset(base_offset)
        self.slot.set_pose(base_pose)

    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(0.5))
        grasp_rotate = self.rng.uniform(-np.pi/18, np.pi/18)
        target_pose = self.prism.get_pose().add_bias([0, 0, 0.012]).add_rotation([0, grasp_rotate, 0])
        target_mat = target_pose.to_transformation_matrix()
        # The ZX hand's fingers open along a hand axis rotated 90 deg from the
        # gsmini panda fingers, so pick the orthogonal object axis as the gripper
        # open direction to keep the same horizontal contact plane on the HDMI.
        camera_up = target_mat[:3, 1] if self.cfg.tactile_sensor_type == 'zxhand' else target_mat[:3, 0]
        cpose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            camera_up
        )
        cid = self.prism.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.prism,
            contact_point_id=cid,
            is_close=False
        ))
        self.move(self.atom.close_gripper())
        # In-hand reference for check_success / check_early_stop (captured right
        # after the grasp closes, before the lift).
        self.origin_inhand_pose = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        self.move(self.atom.move_by_displacement(z=0.02))
        # A real grasp lifts the HDMI off the table; if it stayed down the grasp
        # missed, so abort the episode instead of recording a false success.
        if self.prism.get_pose().p[2] < 0.012:
            self.plan_success = False

        self.target_pose = self.slot.get_pose().add_bias([0.0, 0.0, 0.005])
        self.hole_pose = self.slot.get_pose().add_bias([0.0, 0.0, 0.0128])
        noise = self.create_noise([0.005, 0.005, 0.0])
        self.noise_pose = self.hole_pose.add_offset(noise)
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=self.noise_pose,
            pre_dis=0.02,
            dis=0.01,
            is_open=False
        ))

    def _play_once(self):
        self.move(self.atom.place_actor(
            self.prism,
            target_pose=self.hole_pose,
            pre_dis=0.01,
            dis=0.002,
            is_open=False
        ), time_dilation_factor=0.5)
        self.move(self.atom.move_by_displacement(
            z=0.005, xyz_coord='local'
        ), time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        self.move(self.atom.move_by_displacement(
            z=0.002, xyz_coord='local'
        ), time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        self.delay(20, is_save=True)

    def check_early_stop(self):
        if not hasattr(self, 'origin_inhand_pose'):
            return False
        prism_inhand = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        inhand_bias = np.abs(self.origin_inhand_pose.p[2] - prism_inhand.p[2])
        if inhand_bias > 0.04:
            self.metadata['early_stop'] = True
            self.metadata['inhand_bias'] = float(inhand_bias)
            return True
        return False

    def check_success(self, z_threshold=0.005):
        prism_pose = self.prism.get_pose().rebase(self.target_pose)
        prism_inhand = self.prism.get_pose().rebase(
            self._robot_manager.get_gripper_center_pose())
        ee_pose = self._robot_manager.get_ee_pose()
        self.metadata['rel_pose'] = prism_pose.tolist()
        inhand_ok = (
            hasattr(self, 'origin_inhand_pose')
            and np.abs(self.origin_inhand_pose.p[2] - prism_inhand.p[2]) < 0.04
        )
        return np.all(np.abs(prism_pose.p[:3]) < np.array([0.005, 0.005, z_threshold])) \
            and ee_pose[2] > 0.145 \
            and np.dot(prism_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965 \
            and inhand_ok