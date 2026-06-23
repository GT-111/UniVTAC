"""Robot abstraction layer — embodiments, planners, and the robot registry.

Usage::

    from envs.robot import create_robot, Robot, RobotConfig
    from envs.robot.registry import register_robot, list_robots

    # Create a robot by name (preferred — single dispatch point)
    robot = create_robot("franka_gsmini", cfg, task)

    # Backward-compatible path (still supported)
    from envs.robot.robot import RobotManager
    robot = RobotManager(robot_cfg, task)
"""

from .base import Robot, RobotConfig
from .registry import create_robot, register_robot, list_robots
from .planner import Planner, PlannerResult, LinearPlanner

__all__ = [
    "Robot",
    "RobotConfig",
    "create_robot",
    "register_robot",
    "list_robots",
    "Planner",
    "PlannerResult",
    "LinearPlanner",
]
