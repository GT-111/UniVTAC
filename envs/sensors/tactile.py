import torch
import numpy as np
from tacex import VBTSSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gf225.gf225_cfg import GF225Cfg
from tacex.simulation_approaches.fem_based import ManiSkillSimulatorCfg

from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject

from tacex_uipc import (
    UipcRLEnv,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcSimCfg
)

from ..utils.transforms import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask
    from tacex_uipc.sim import UipcIsaacAttachmentsCfg, UipcSim
    from tacex_uipc import UipcInteractiveScene

@configclass
class TactileCfg:
    name: str = 'tactile_sensor'
    sensor_type: str = 'unknown'  # 'gsmini', 'gf225', 'zxhand'
    sensor_cfg = None
    gelpad_cfg: UipcObjectCfg = None
    gelpad_attachment_cfg: UipcIsaacAttachmentsCfg = None

def create_gelsight_mini_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    name: str = "tactile_sensor",
    resolution = (320, 240),
    update_period = 1/120,
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
    sensor_type_label: str = "gsmini",
):
    from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
    sensor_cfg = GelSightMiniCfg(
        prim_path=prim_path,
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            resolution=resolution,
            update_period=update_period,
            data_types=["depth", "rgb"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,  # for rendering sensor output in the gui
        update_period=1/120,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            marker_shape=(9, 7),
            marker_interval=(2.40625, 2.45833),
            sub_marker_num=0,
            marker_radius=6,
            camera_to_surface=0.0283,
            real_size=(0.0266, 0.0209),
            sensor_type='gsmini',
        ),
        data_types=data_type
    )
    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 64
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        tactile_img_res=resolution,
        device="cuda",
    )

    cfg = TactileCfg(
        name=name,
        sensor_type=sensor_type_label,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            debug_vis=False,
        ),
    )
    return cfg

def create_gf225_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    data_type: list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    resolution = (480, 480)  # GF225 resolution
    update_period = 1/120
    
    sensor_cfg = GF225Cfg(
        prim_path=prim_path,
        sensor_camera_cfg=GF225Cfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            resolution=resolution,
            update_period=update_period,
            data_types=["depth"],
            clipping_range=(0.02, 0.0265),
        ),
        device="cuda",
        debug_vis=False,
        update_period=1/120,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            sub_marker_num=0,
            marker_radius=8,
            marker_shape=(9, 9),
            marker_interval=(2.0, 2.0),
            camera_to_surface=0.0265,
            real_size = (0.0235, 0.0250),
            sensor_type='gf225',
        ),
        data_types=data_type
    )
    
    from tacex.simulation_approaches.mlp_fots import MLPFOTSSimulatorCfg
    from tacex_assets import TACEX_ASSETS_DATA_DIR

    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 81
    sensor_cfg.optical_sim_cfg = MLPFOTSSimulatorCfg(
        calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GF225/calibs/480x480",
        tactile_img_res=resolution,
        device="cuda",
    )
    
    cfg = TactileCfg(
        name=name,
        sensor_type="gf225",
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            isaac_rigid_prim_path=gelpad_attachment_prim_path,
            debug_vis=False,
        ),
    )
    return cfg

def create_tactile_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    sensor_type:Literal['gsmini', 'gf225'] = "gsmini",
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    if sensor_type == "gsmini":
        return create_gelsight_mini_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            name=name,
            data_type=data_type,
        )
    elif sensor_type == "gf225":
        return create_gf225_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            gelpad_attachment_prim_path=gelpad_attachment_prim_path,
            name=name,
            data_type=data_type,
        )
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")


class VisualTactileSensor:
    """TacEx VBTS-based tactile sensor (GS Mini, GF225)."""

    def __init__(self, name:str, cfg:TactileCfg, robot, scene: 'UipcInteractiveScene', uipc_sim:'UipcSim'):
        self.cfg = cfg
        self.name = name
        self.scene = scene
        self.robot = robot
        self.uipc_sim = uipc_sim

        self.gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, self.gelpad, self.robot
        )
        self.sensor = VBTSSensor(self.cfg.sensor_cfg, self.gelpad)
    
    def setup(self):
        self.device = self.uipc_sim.cfg.device
        init_pts = self.gelpad._data.nodal_pos_w[self.attachment.attachment_points_idx].cpu().numpy()
        init_world_trans = self.gelpad.init_world_transform.cpu().numpy()
        self.origin_pts = (init_pts - init_world_trans[:3, 3]) @ (init_world_trans[:3, :3].T).T
        attach_pts = self.attachment.attachment_offsets
        init_trans = estimate_rigid_transform(self.origin_pts, attach_pts)
        self.attach_to_init = np.linalg.inv(init_trans)
        self.attach_to_init = torch.tensor(self.attach_to_init, dtype=torch.float64, device=self.device)

        self.sensor.marker_motion_simulator.marker_motion_sim.init_vertices()
    
    def get_attach_pose(self):
        if type(self.attachment.isaaclab_rigid_object) is Articulation:
            poses = self.attachment.isaaclab_rigid_object._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            pose = poses[:, self.attachment.rigid_body_id, 0:7].clone()
        elif type(self.attachment.isaaclab_rigid_object) is RigidObject:
            pose = self.attachment.isaaclab_rigid_object._root_physx_view.root_state_w.view(-1, 1, 13)
            pose = pose[:, self.attachment.rigid_body_id, 0:7].clone()
        else:
            raise RuntimeError("Need an Articulation or a RigidBody object for the Isaac X UIPC attachment.")
        return Pose.from_list(pose.flatten().tolist())

    def get_init_pts(self):
        curr_attach_pose = self.get_attach_pose()
        trans_to_attach = np.linalg.inv(curr_attach_pose.to_transformation_matrix())
        trans_to_attach = torch.tensor(trans_to_attach, dtype=torch.float64, device=self.device)
        trans_to_init = self.attach_to_init @ trans_to_attach
        return self.gelpad.data.nodal_pos_w @ trans_to_init[:3, :3].T + trans_to_init[:3, 3]
 
    def update(self, dt, force_recompute=False):
        self.gelpad.update(dt=dt)
        self.sensor.update(dt=dt, force_recompute=force_recompute)
    
    def set_debug_vis(self):
        if not self.sensor.cfg.debug_vis:
            return 
        for data_type in ['marker_motion']:
            self.sensor._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
    
    def get_observations(self, data_types: list[str] = None):
        obs = {}
        if data_types is None:
            data_types = ['rgb', 'rgb_marker', 'depth', 'points', 'pose', 'flow']
        for data_type in data_types:
            if data_type == 'rgb':
                obs['rgb'] = self.sensor.data.output['tactile_rgb'].squeeze(0)
            elif data_type == 'rgb_marker':
                obs['rgb_marker'] = self.sensor.data.output['marker_rgb'].squeeze(0)
            elif data_type == 'depth':
                obs['depth'] = self.sensor.data.output['height_map'].squeeze(0)
            elif data_type == 'marker':
                obs['marker'] = self.sensor.data.output['marker_motion'].squeeze(0)
            elif data_type == 'points':
                obs['points'] = self.get_init_pts()
            elif data_type == 'pose':
                obs['pose'] = self.get_attach_pose().totensor()
        return obs
    
    def _reset_idx(self):
        self.init_pose_mat = self.get_attach_pose().to_transformation_matrix()

    def get_min_depth(self):
        return torch.min(self.sensor.data.output['height_map']).item()


class TactileManager:
    def __init__(self, cfg_list: list[TactileCfg], task:'BaseTask'):
        self.task = task
        self.scene = task.scene
        self.uipc_sim = task.uipc_sim
        self.robot = task._robot_manager.robot
        
        self.tactiles = {}
        for cfg in cfg_list:
            if cfg.sensor_type in ('gsmini', 'gf225'):
                self.tactiles[cfg.name] = VisualTactileSensor(
                    cfg.name, cfg, self.robot, self.scene, self.uipc_sim
                )
            elif cfg.sensor_type == 'zxhand':
                from .zx_official import OfficialZXTactileSensor

                sensor_cfg = cfg.sensor_cfg
                finger_prim_path = sensor_cfg["finger_prim_path"].replace("env_.*", "env_0")
                self.tactiles[cfg.name] = OfficialZXTactileSensor(
                    name=cfg.name,
                    finger_prim_path=finger_prim_path,
                    side=sensor_cfg["side"],
                    device=str(task.device),
                )
            else:
                raise ValueError(
                    f"Unknown sensor_type '{cfg.sensor_type}' in TactileCfg '{cfg.name}'"
                )

    def update(self, dt, force_recompute=False):
        obj_pose_7d = None
        if hasattr(self.task, "_actor_manager") and self.task._actor_manager.actors:
            # Use the last actor as the manipulated object. For insertion tasks
            # this skips the fixed slot/base and selects the plug/prism.
            actor = next(reversed(self.task._actor_manager.actors.values()))
            obj_pose_7d = actor.get_pose().tolist()
        for tact in self.tactiles.values():
            if obj_pose_7d is not None and hasattr(tact, "obj_pose_7d"):
                tact.obj_pose_7d = np.asarray(obj_pose_7d)
            tact.update(dt=dt, force_recompute=force_recompute)
 
    def set_debug_vis(self, debug_vis):
        if not debug_vis: return
        for tact in self.tactiles.values():
            set_debug_vis = getattr(tact, "set_debug_vis", None)
            if callable(set_debug_vis):
                set_debug_vis()

    def get_observations(self, data_types: list[str] = None):
        obs = {}
        for name, tact in self.tactiles.items():
            obs[name] = tact.get_observations(data_types)
        return obs

    def get_min_depth(self):
        self.task._update_render()
        depth = []
        for tact in self.tactiles.values():
            depth.append(tact.get_min_depth())
        return torch.tensor(depth, dtype=torch.float32, device=self.task.device)

    def _reset_idx(self):
        for tact in self.tactiles.values():
            tact._reset_idx()

    def setup(self):
        for tact in self.tactiles.values():
            tact.setup()

    def close(self):
        for tact in self.tactiles.values():
            close_fn = getattr(tact, "close", None)
            if callable(close_fn):
                close_fn()