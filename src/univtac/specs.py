"""Observation and action specification validation.

Modeled after the VLA Evaluation Harness ``DimSpec``.

Each policy declares its action/observation spec; each task declares what it
expects. The eval runner cross-checks them at startup — catching format
mismatches (wrong action dim, inverted gripper convention) before wasting
GPU hours on 100 episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DimSpec:
    """Describes one dimension group in an action or observation array.

    Example::

        EE_POSE_8D = DimSpec("ee_pose", 8, "eef_pos3_quat4_gripper1", range=(-1.0, 1.0))
    """

    name: str
    """Human-readable label (e.g. 'ee_pose', 'qpos')."""

    dims: int
    """Number of dimensions in this group."""

    format: str = ""
    """Description of the format (e.g. 'pos3_rot3_gripper1')."""

    range: tuple[float, float] | None = None
    """Expected value range, or None if unconstrained."""

    def validate(self, value: np.ndarray) -> list[str]:
        """Check a value against this spec. Returns list of error messages."""
        errors: list[str] = []
        if value.shape[-1] != self.dims:
            errors.append(
                f"{self.name}: expected {self.dims} dims, got {value.shape[-1]}"
            )
        if self.range is not None:
            lo, hi = self.range
            if (value < lo).any() or (value > hi).any():
                errors.append(
                    f"{self.name}: values outside [{lo}, {hi}]"
                )
        return errors

    def is_compatible(self, other: DimSpec) -> tuple[bool, str]:
        """Check if two specs are compatible (same dims, compatible format)."""
        if self.dims != other.dims:
            return False, f"Dim mismatch: {self.dims} vs {other.dims}"
        return True, ""


# ── Predefined common specs ─────────────────────────────────────────────

QPOS_7D = DimSpec("qpos", 7, "joint_positions", range=(-3.15, 3.15))
EE_POSE_8D = DimSpec("ee_pose", 8, "eef_pos3_quat4_gripper1")
DELTA_EE_7D = DimSpec("delta_ee", 7, "delta_pos3_rot3_gripper1")
DELTA_EE_8D = DimSpec("delta_ee", 8, "delta_pos3_quat4_gripper1")


@dataclass
class ActionSpec:
    """Complete action specification for validation."""

    groups: list[DimSpec] = field(default_factory=list)

    @property
    def total_dim(self) -> int:
        return sum(g.dims for g in self.groups)

    def validate(self, action: Any) -> list[str]:
        """Validate an action against this spec."""
        errors: list[str] = []
        arr = np.asarray(action)
        offset = 0
        for group in self.groups:
            slice_val = arr[..., offset:offset + group.dims]
            errors.extend(group.validate(slice_val))
            offset += group.dims
        if offset != arr.shape[-1]:
            errors.append(f"Total dims {offset} != actual {arr.shape[-1]}")
        return errors
