"""Common type aliases for UniVTAC."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

#: Task name string (e.g. "grasp_classify", "insert_HDMI")
TaskName = str

#: Policy name string (e.g. "ACT", "OpenPI", "ViTAL")
PolicyName = str

#: Sensor type string
SensorType = Literal["gsmini", "gf225", "zxhand"]

#: Instruction type
InstructionType = Literal["seen", "unseen"]

#: Observation dict — camera images, robot state, tactile data
Observation = dict[str, Any]

#: Action array
Action = Any  # np.ndarray, but avoid numpy import at module level

#: Config dict (raw YAML/JSON)
ConfigDict = dict[str, Any]

#: File system path
FSPath = str | Path
