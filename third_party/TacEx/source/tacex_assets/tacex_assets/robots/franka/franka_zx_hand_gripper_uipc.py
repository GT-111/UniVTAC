# Copyright (c) 2022-2023, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Franka Emika Panda arm + Xense ZX hand.

The ZX hand (from xense_sim/xense_assets/franka_zx_hand_real.usd) has:
  - 10 finger joints (closed-chain 4-bar linkage, ~8 DOF actuated)
  - Built-in gel surfaces on xense_leftfinger / xense_rightfinger
  - Internal cameras: Right_Camera, Left_Camera (in-finger), Head_Camera (top)
  - No wrist camera — Head_Camera serves as the wrist-equivalent viewpoint

Reference: third_party/xense_sim/xense_assets/
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


FRANKA_PANDA_ARM_ZX_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/a25278/Workspaces/TactileWS/UniVTAC/third_party/xense-sim4.5/xense/isaac/xense_assets/franka_zx_hand_real.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Arm (7 DOF)
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            # ZX hand: closed-chain 4-bar linkage. Per the official example, ONLY
            # `right_Left_1_Joint` is driven; `right_Right_1_Joint` and the link
            # joints follow the loop. USD joint limit is [0, 1.0] rad. Start near
            # full open (the pose the grasp approaches in) so the runtime TCP
            # calibration matches the grasp-time finger geometry.
            "right_Left_1_Joint": 0.99,
            "right_Right_1_Joint": 0.99,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        # Gripper drive — EXACT match of the USD native drive on the single driven
        # joint `right_Left_1_Joint`: a stiff VELOCITY drive (stiffness 0, damping
        # 1e4, maxForce 10). vel_target>0 opens, <0 closes, =0 locks in place
        # (the high damping resists drift -> holds the grasp). The official
        # cust_gripper commands ONLY this joint; driving both over-constrains the
        # loop and collapses the gripper.
        "zx_drive": ImplicitActuatorCfg(
            joint_names_expr=["right_Left_1_Joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=130.0,
            stiffness=0.0,
            damping=10000.0,
        ),
        # Opposite finger: USD applies NO drive — it is free and follows the loop.
        "zx_free": ImplicitActuatorCfg(
            joint_names_expr=["right_Right_1_Joint", "left_RevoluteJoint", "right_RevoluteJoint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=130.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # Linkage "0" joints — USD native stiffness 0.05.
        "zx_link0": ImplicitActuatorCfg(
            joint_names_expr=["right_Left_0_Joint", "right_Right_0_Joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=130.0,
            stiffness=0.05,
            damping=0.0,
        ),
        # Support joints — USD native stiffness 1e-4.
        "zx_support": ImplicitActuatorCfg(
            joint_names_expr=["right_Left_Support_Joint", "right_Right_Support_Joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=130.0,
            stiffness=0.0001,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FRANKA_PANDA_ARM_ZX_HAND_HIGH_PD_CFG = FRANKA_PANDA_ARM_ZX_HAND_CFG.copy()
"""Stiffer PD control variant for task-space control via differential IK."""

FRANKA_PANDA_ARM_ZX_HAND_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# Stiff arm PD so the asymmetric gripper-close reaction can't drift the wrist off
# the object (teleport-holding the arm during close disturbs the closed chain).
FRANKA_PANDA_ARM_ZX_HAND_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 1500.0
FRANKA_PANDA_ARM_ZX_HAND_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 200.0
FRANKA_PANDA_ARM_ZX_HAND_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 1500.0
FRANKA_PANDA_ARM_ZX_HAND_HIGH_PD_CFG.actuators["panda_forearm"].damping = 200.0
# Gripper actuators keep the USD-native gains (see above) — do NOT override.
