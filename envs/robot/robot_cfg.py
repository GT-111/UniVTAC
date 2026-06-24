from tacex_assets.robots.franka.franka_gsmini_gripper_uipc_high_res import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_gf225_gripper_uipc import (
    FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_zx_hand_gripper_uipc import (
    FRANKA_PANDA_ARM_ZX_HAND_HIGH_PD_CFG
)

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from tacex_assets import TACEX_ASSETS_DATA_DIR
from ..sensors.tactile import TactileCfg, create_tactile_cfg
from .._global import ASSETS_ROOT

@configclass
class RobotCfg:
    robot: ArticulationCfg = None
    tactiles: list[TactileCfg] = []

    gripper_offset: float = 0.131 # in m
    gripper_max_qpos: float = 0.039 # in m

    tactile_far_plane: float = 30.0 # in mm
    adaptive_grasp_depth_threshold: float = 27.5 # in mm, used for grasping
    contact_threshold: tuple[float, float] = (27.5, 28.0) # in mm, used in `gravity_rotate` api

    use_adaptive_grasp: bool = True
    wrist_camera_prim_path: str = "/World/envs/env_.*/Robot/WristCamera/Camera"

def create_franka_gsmini_gripper(data_type:list[str]):
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ),
    )
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_left",
            gelpad_attachment_body_name="gelsight_mini_case_left",
            name="left_tactile",
            sensor_type="gsmini",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_right",
            gelpad_attachment_body_name="gelsight_mini_case_right",
            name="right_tactile",
            sensor_type="gsmini",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=34.0,
        adaptive_grasp_depth_threshold=27.5,
        contact_threshold=(27.5, 28.0)
    )

def create_franka_gf225_gripper(data_type:list[str]):
    robot = FRANKA_PANDA_ARM_GF225_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ), 
    )
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_left",
            gelpad_attachment_body_name="GF225_left",
            name="left_tactile",
            sensor_type="gf225",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/GF225_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/GF225_gelpad_right",
            gelpad_attachment_body_name="GF225_right",
            name="right_tactile",
            sensor_type="gf225",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=29.0,
        adaptive_grasp_depth_threshold=26.8,
        contact_threshold=(26.5, 27.0)
    )

def create_franka_zx_hand_gripper(data_type: list[str]):
    """ZX hand with official xense USD + GelSight-aligned UIPC collision.

    Uses the official ``franka_zx_hand_real.usd`` from third_party/xense-sim4.5.
    UIPC collision bridging uses ``ZxGelpad`` / ``ZxRodProxy`` — the same
    ``UipcObject`` + ``UipcIsaacAttachments`` pattern as GelSight gel pads.
    Tactile sensing is handled by ``OfficialZXTactileSensor`` (xense plugin).
    """
    _XENSE_USD = (
        Path(__file__).resolve().parents[2]
        / "third_party/xense-sim4.5/xense/isaac/xense_assets/franka_zx_hand_real.usd"
    )
    robot = FRANKA_PANDA_ARM_ZX_HAND_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_XENSE_USD),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                # official ZX hand (cust_franka.py): open ~ pi/3, but USD caps the
                # joint at 1.0 rad (exclusive), so 0.99 = open, 0 = closed
                "right_Left_1_Joint": 0.99,
                "right_Right_1_Joint": 0.99,
            }
        ),
    )
    tactiles = [
        TactileCfg(
            name="left_tactile",
            sensor_type="zxhand",
            sensor_cfg={
                "finger_prim_path": "/World/envs/env_.*/Robot/xense_leftfinger",
                "side": "left",
            },
        ),
        TactileCfg(
            name="right_tactile",
            sensor_type="zxhand",
            sensor_cfg={
                "finger_prim_path": "/World/envs/env_.*/Robot/xense_rightfinger",
                "side": "right",
            },
        ),
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        # ZX: virtual panda_hand + runtime-calibrated offset (see robot.setup()).
        gripper_offset=0.0,
        gripper_max_qpos=0.99,
        # Official ZX tactile: get_min_depth returns processed cam depth (mm),
        # resting (no contact) = +1.0 mm (clip max), full indentation = -4.0 mm.
        # far_plane must equal the resting value so adaptive grasp uses coarse
        # steps until contact, then fine steps to reach the contact threshold.
        tactile_far_plane=1.0,
        adaptive_grasp_depth_threshold=-1.5,
        contact_threshold=(-2.0, -1.5),
        use_adaptive_grasp=False,
        wrist_camera_prim_path="/World/envs/env_.*/Robot/right_base_link/Hand_Camera",
    )


# ---------------------------------------------------------------------------
# Sensor-type → Robot-name mapping (single dispatch point)
# ---------------------------------------------------------------------------

_SENSOR_TO_ROBOT: dict[str, str] = {
    "gsmini": "franka_gsmini",
    "gf225":  "franka_gf225",
    "zxhand": "franka_zxhand",
}

_ROBOT_CFG_FACTORIES = {
    "gsmini": create_franka_gsmini_gripper,
    "gf225":  create_franka_gf225_gripper,
    "zxhand": create_franka_zx_hand_gripper,
}


def get_robot_name(sensor_type: str) -> str:
    """Map a tactile sensor type to a registered Robot subclass name."""
    name = _SENSOR_TO_ROBOT.get(sensor_type)
    if name is None:
        raise ValueError(
            f"Unknown sensor type '{sensor_type}'. "
            f"Known: {list(_SENSOR_TO_ROBOT.keys())}"
        )
    return name


def create_robot_cfg(sensor_type: str, data_type: list[str]) -> RobotCfg:
    """Create a RobotCfg for the given sensor type (single dispatch)."""
    factory = _ROBOT_CFG_FACTORIES.get(sensor_type)
    if factory is None:
        raise ValueError(
            f"Unknown sensor type '{sensor_type}'. "
            f"Known: {list(_ROBOT_CFG_FACTORIES.keys())}"
        )
    return factory(data_type)
