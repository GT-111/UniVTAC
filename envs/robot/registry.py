"""Robot registry — the single dispatch point for embodiment selection.

Usage::

    from envs.robot.registry import register_robot, create_robot

    @register_robot("franka_gsmini")
    class FrankaGelSight(Robot): ...

    robot = create_robot("franka_gsmini", cfg, task)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Robot, RobotConfig
    from .._base_task import BaseTask

_ROBOT_REGISTRY: dict[str, type["Robot"]] = {}


def register_robot(name: str):
    """Decorator — each Robot subclass announces itself under one or more names.

    ``@register_robot('franka_gsmini')`` and ``@register_robot('franka_gf225')``
    can decorate the *same* class when two sensor variants share the same
    mechanical behaviour.
    """

    def decorator(cls: type["Robot"]):
        _ROBOT_REGISTRY[name] = cls
        return cls

    return decorator


def create_robot(name: str, cfg: "RobotConfig", task: "BaseTask") -> "Robot":
    """Create a Robot instance by registered name.

    This is the **only** place in the codebase that maps a string name to a
    concrete Robot class.  Adding a new embodiment means:
        1. Write a Robot subclass.
        2. Decorate it with ``@register_robot('new_name')``.
        3. Done.  Zero changes to tasks or BaseTask.
    """
    if name not in _ROBOT_REGISTRY:
        raise ValueError(
            f"Unknown robot '{name}'.  Registered: {list(_ROBOT_REGISTRY.keys())}"
        )
    return _ROBOT_REGISTRY[name](cfg, task)


def list_robots() -> list[str]:
    """Return all registered robot names."""
    return sorted(_ROBOT_REGISTRY.keys())
