"""Manipulation primitives — reusable, testable, robot-agnostic.

These are pure Python + numpy + torch functions that operate on a Robot
interface.  No Omniverse / Isaac Sim / TacEx imports, so they can be
unit-tested on CPU with mock objects.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Generator, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .utils.transforms import Pose
    from .utils.actor import Actor


# ---------------------------------------------------------------------------
# Robot interface (Protocol — anything with these methods works)
# ---------------------------------------------------------------------------

class _RobotLike(Protocol):
    """Minimal robot surface needed by manipulation primitives."""

    @property
    def device(self) -> torch.device: ...

    def get_gripper_qpos(self) -> float: ...

    def get_gripper_center_pose(self) -> "Pose": ...

    def get_ee_pose(self) -> "Pose": ...

    def gripper_center_to_ee(self, gripper_center_pose: "Pose") -> "Pose": ...

    def set_gripper(self, pos: torch.Tensor, vel: torch.Tensor | None = None) -> None: ...


class _TactileLike(Protocol):
    """Minimal tactile surface needed by manipulation primitives."""

    def get_min_depth(self) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# adaptive_set_gripper
# ---------------------------------------------------------------------------

def adaptive_set_gripper(
    robot: _RobotLike,
    tactile: _TactileLike,
    target_qpos: float,
    tactile_far_plane: float,
    depth_threshold: float | None = None,
    step_size_open: float = 0.0005,
    step_size_close: float = 0.0005,
    max_steps: int = 1000,
    sim_dt: float = 1 / 120,
) -> Generator[tuple[torch.Tensor, torch.Tensor, bool], None, None]:
    """Closed-loop adaptive gripper controller.

    Yields (position, velocity, active) tuples.  The caller executes each
    step by calling ``robot.set_gripper(position, velocity)`` and stepping
    the simulation.

    Parameters
    ----------
    step_size_open, step_size_close:
        Coarse step size in *robot-native units* (metres for prismatic
        GelSight grippers, radians for angular ZX gripper).
    """
    default_step = step_size_open
    contact_step = step_size_close

    last_qpos = robot.get_gripper_qpos()
    n_grip: int = 1  # will be inferred from tensor shape on first yield

    max_depth = tactile_far_plane * torch.ones_like(tactile.get_min_depth())
    if depth_threshold is not None:
        depth_threshold_t = depth_threshold * torch.ones_like(max_depth)
    else:
        depth_threshold_t = None

    direction = "open" if robot.get_gripper_qpos() < target_qpos else "close"
    step_size = contact_step if direction == "open" else -default_step

    for _ in range(max_steps):
        current_qpos = robot.get_gripper_qpos()
        tactile_depth = tactile.get_min_depth()

        if direction == "close":
            if torch.allclose(max_depth, tactile_depth, atol=1e-5):
                step_size = -default_step
            elif depth_threshold_t is not None:
                if torch.all(tactile_depth < depth_threshold_t):
                    break
                else:
                    step_size = -min(
                        torch.min(torch.abs(tactile_depth - depth_threshold_t)).item() / 1000,
                        contact_step,
                    )
            else:
                step_size = -default_step
        else:
            if torch.allclose(max_depth, tactile_depth, atol=1e-5):
                step_size = default_step
            if depth_threshold_t is not None:
                if torch.all(tactile_depth > depth_threshold_t):
                    break
                else:
                    step_size = min(
                        torch.min(torch.abs(depth_threshold_t - tactile_depth)).item() / 1000,
                        contact_step,
                    )
            else:
                step_size = default_step

        if np.allclose(current_qpos, target_qpos, atol=1e-5):
            break
        elif np.abs(current_qpos - target_qpos) < np.abs(step_size):
            target_step = target_qpos
        else:
            target_step = current_qpos + step_size

        # Infer gripper DOF count from the robot's current qpos context
        try:
            n_grip = robot._gripper_ids.numel()
        except Exception:
            n_grip = 1

        position = torch.full((n_grip,), float(target_step), device=robot.device)
        velocity = (position - current_qpos) / sim_dt
        last_qpos = current_qpos
        yield position, velocity, True

    # Final hold
    final_position = torch.full((n_grip,), float(last_qpos), device=robot.device)
    yield final_position, torch.zeros_like(final_position), False


# ---------------------------------------------------------------------------
# gravity_rotate
# ---------------------------------------------------------------------------

def gravity_rotate(
    robot: _RobotLike,
    tactile: _TactileLike,
    actor: "Actor",
    target_vec: np.ndarray | list,
    target_axis: np.ndarray | list | None = None,
    contact_threshold: tuple[float, float] = (27.5, 28.0),
    max_steps: int = 200,
    omega_threshold: float = 0.05,
    is_save: bool = True,
    _step_fn=None,
    _move_fn=None,
) -> bool:
    """Rotate grasped object until a target vector aligns with a target axis.

    Uses tactile depth feedback to modulate grip force during rotation.
    Requires ``_step_fn(is_save)`` and ``_move_fn(actions, tag, is_save, delay)``
    callbacks to be injected (they need the Omniverse sim).

    Returns True on success, False if planning failed.
    """
    if target_axis is None:
        target_axis = np.array([0, 0, 1])
    target_axis = np.array(target_axis).reshape(3, 1)
    target_vec = np.array(target_vec) / np.linalg.norm(target_vec)

    def _get_axis():
        R = actor.get_pose().to_transformation_matrix()[:3, :3]
        axis = (R @ target_axis).reshape(-1)
        return axis / np.linalg.norm(axis)

    last_z = _get_axis()
    last_theta = np.arccos(np.dot(last_z, target_vec))

    for _ in range(max_steps):
        curr_z = _get_axis()
        curr_qpos = robot.get_gripper_qpos()
        curr_depth = torch.min(tactile.get_min_depth()).item()

        theta = np.arccos(np.dot(curr_z, target_vec))
        if theta < 0.05 or theta > last_theta:
            break
        omega = theta - last_theta
        last_theta = theta

        if np.abs(omega) < omega_threshold:
            if curr_depth < contact_threshold[1]:
                curr_qpos += 0.0001
        elif curr_depth > contact_threshold[0]:
            curr_qpos -= 0.0001

        position = torch.tensor(
            [curr_qpos, curr_qpos], dtype=torch.float32, device=robot.device
        )
        velocity = torch.clip(
            (position - curr_qpos) / (1 / 120), -0.0001, 0.0001
        )
        robot.set_gripper(position, velocity)

        if _step_fn is not None:
            for _ in range(5):
                _step_fn(is_save)

        last_z = curr_z

    return True


# ---------------------------------------------------------------------------
# gripper_rotate
# ---------------------------------------------------------------------------

def gripper_rotate(
    robot: _RobotLike,
    actor: "Actor",
    theta: float,
    steps: int = 6,
    is_save: bool = True,
    _move_fn=None,
) -> bool:
    """Rotate grasped object in-hand by *theta* radians around the Y axis.

    Requires ``_move_fn`` callback for the motion execution (needs Omniverse).
    Returns False if planning failed.
    """
    if _move_fn is None:
        return False

    from .utils.atom import Action  # local import to avoid Kit dependency at module level

    for _ in range(steps):
        rpy = [0, theta / steps, 0]
        actor_pose = actor.get_pose()
        gripper_center_pose = robot.get_gripper_center_pose()
        new_gripper_center = gripper_center_pose.add_rotation(rpy, coord=actor_pose)
        new_gripper_center.q = gripper_center_pose.q.copy()
        new_target_pose = robot.gripper_center_to_ee(new_gripper_center)
        _move_fn(
            [Action(action="move", target_pose=new_target_pose)],
            tag="rotate",
            is_save=is_save,
            delay=False,
            time_dilation_factor=0.5,
        )

    return True


# ---------------------------------------------------------------------------
# try_forward
# ---------------------------------------------------------------------------

def try_forward(
    actor: "Actor",
    robot: _RobotLike | None = None,
    dis: float = 0.01,
    delta_d: float = 0.004,
    is_save: bool = True,
    _move_fn=None,
    _plan_success_fn=None,
) -> bool:
    """Incremental forward (local-z) probing motion.

    Moves the end-effector forward by *dis* metres in *delta_d* increments.
    Returns False if the object stops moving (stuck / obstacle) or if
    planning failed.

    Requires ``_move_fn`` and ``_plan_success_fn`` callbacks.
    """
    if _plan_success_fn is not None and not _plan_success_fn():
        return False
    if _move_fn is None:
        return False

    from .utils.atom import Atom  # local import

    # We need an Atom instance to build the displacement action
    # The caller should pass a pre-constructed atom or we create one inline
    actor_last_pose = actor.get_pose()
    max_trials = int(np.ceil(np.abs(dis / delta_d)))
    delta = np.sign(dis) * delta_d

    for _ in range(max_trials):
        _move_fn(
            Atom.move_by_displacement(None, z=delta, xyz_coord="local"),
            tag="try_forward",
            is_save=is_save,
            delay=False,
        )
        actor_pose = actor.get_pose()
        if np.linalg.norm(actor_pose.p - actor_last_pose.p) < np.abs(delta):
            return False
        actor_last_pose = actor_pose

    return True


# ---------------------------------------------------------------------------
# Utility: check if object is still in-hand
# ---------------------------------------------------------------------------

def check_object_inhand(
    robot: _RobotLike,
    actor: "Actor",
    origin_inhand_pose: np.ndarray | None,
    max_displacement: float = 0.04,
) -> bool:
    """Return True if the object hasn't slipped out of the gripper."""
    if origin_inhand_pose is None:
        return True  # no reference — assume ok
    current = actor.get_pose().rebase(robot.get_gripper_center_pose())
    return bool(np.abs(origin_inhand_pose[2] - current[2]) < max_displacement)
