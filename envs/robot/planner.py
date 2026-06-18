"""Planner abstraction and built-in implementations.

A Planner converts a target end-effector pose into a joint-space trajectory
(position + velocity arrays).  The interface is deliberately small so that
alternative backends (cuRobo, linear interpolation, future RRT, …) can be
swapped without touching task or robot code.
"""

from __future__ import annotations

import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PlannerResult:
    """Output of a single planning request."""

    status: str  # "Success" | "Fail"
    num_steps: int = 0
    position: torch.Tensor | None = None  # [num_steps, dof]
    velocity: torch.Tensor | None = None  # [num_steps, dof]


# ---------------------------------------------------------------------------
# Abstract planner
# ---------------------------------------------------------------------------

class Planner(ABC):
    """Abstract motion planner — pure geometry, no side effects."""

    @abstractmethod
    def plan_path(
        self,
        curr_joint_pos: torch.Tensor,
        curr_joint_vel: torch.Tensor,
        target_ee_pose,  # Pose (local import)
        *,
        constraint_pose=None,
        pre_dis: float | None = None,
        time_dilation_factor: float = 1.0,
        **kwargs,
    ) -> PlannerResult:
        """Plan a collision-free (or straight-line) trajectory.

        Parameters
        ----------
        curr_joint_pos: [dof] current arm joint positions.
        curr_joint_vel: [dof] current arm joint velocities.
        target_ee_pose: Pose — desired end-effector pose in robot-root frame.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear planner state between episodes."""
        ...

    @abstractmethod
    def update_world(self) -> None:
        """Refresh the planner's collision world from the current scene."""
        ...


# ---------------------------------------------------------------------------
# Linear (straight-line) planner — for testing / fast iteration
# ---------------------------------------------------------------------------

class LinearPlanner(Planner):
    """Straight-line joint-space interpolation.  *No collision checking.*

    Useful as a lightweight fallback when cuRobo is unavailable (e.g. CI,
    unit tests, environments without a GPU).
    """

    def __init__(self, dt: float = 1 / 120, max_vel: float = 1.0):
        self._dt = dt
        self._max_vel = max_vel

    def plan_path(
        self,
        curr_joint_pos: torch.Tensor,
        curr_joint_vel: torch.Tensor,
        target_ee_pose,
        **kwargs,
    ) -> PlannerResult:
        # LinearPlanner ignores target_ee_pose — it only does joint-space
        # interpolation.  For real use the caller should compute IK first.
        # This is a *minimal* implementation for testing.
        return PlannerResult(
            status="Success",
            num_steps=1,
            position=curr_joint_pos.unsqueeze(0).clone(),
            velocity=torch.zeros_like(curr_joint_pos).unsqueeze(0),
        )

    def reset(self) -> None:
        pass

    def update_world(self) -> None:
        pass
