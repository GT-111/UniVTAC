"""Official Xense ZX-hand tactile sensor — faithful port of the official plugin
(xense-sim4.5 `xense_peg_insert_tactile_office/office_usb_insert.py`).

The ZX hand (xense_leftfinger / xense_rightfinger) has NO in-gel camera and NO
UIPC gel. The official plugin generates tactile by:
  1. adding an ORTHOGRAPHIC depth camera as a child of each finger,
  2. feeding that depth (after a fixed offset/clip/scale) + the contacted
     object pose + the camera (sensor) pose into the official `xensim.FemSensor`
     (g1-ws gel model),
  3. reading get_image / get_marker / get_force from the FEM.

This module replicates that exactly inside the UniVTAC (Isaac Lab) runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
# official Xense Isaac plugin (complete): xensim/xensesdk live under pip_prebundle
XENSE_SIM_ROOT = _REPO_ROOT / "third_party" / "xense-sim4.5" / "pip_prebundle"
DEFAULT_CALIB_FILE = XENSE_SIM_ROOT / "xensim" / "assets" / "fem" / "g1-ws_table.npz"
DEFAULT_FEM_FILE = XENSE_SIM_ROOT / "xensim" / "assets" / "fem" / "g1-ws.npz"

# official finger-camera params (from office_usb_insert.py)
CAM_RES = (64, 100)            # (width, height) passed to omni Camera.resolution
FEM_DEPTH_SIZE = (64, 100)
CAM_CLIP = (1e-9, 0.005)
CAM_APERTURE = 0.0195
# left/right finger camera local translation + Z-euler (deg)
LEFT_CAM = dict(translation=(0.0, 0.003, 0.037), euler_deg=(0.0, 0.0, 270.0))
RIGHT_CAM = dict(translation=(0.0, -0.003, 0.037), euler_deg=(0.0, 0.0, 90.0))

# Finger-body → gel-surface offsets as np.ndarray (for UIPC bridge & FK).
# In the finger-body local frame, the camera (= gel surface) is mounted at:
LEFT_CAM_LOCAL  = np.array(LEFT_CAM["translation"])   # (0.0,  0.003, 0.037)
RIGHT_CAM_LOCAL = np.array(RIGHT_CAM["translation"])  # (0.0, -0.003, 0.037)


def _ensure_display():
    disp = os.environ.get("XENSE_DISPLAY") or os.environ.get("DISPLAY") or ":1"
    os.environ["DISPLAY"] = disp


class OfficialZXTactileSensor:
    """One ZX-hand finger tactile sensor (camera + official FemSensor)."""

    _sys_path_added = False

    def __init__(self, name: str, finger_prim_path: str, side: str, device: str = "cuda"):
        self.name = name
        self.finger_prim_path = finger_prim_path  # e.g. /World/envs/env_0/Robot/xense_leftfinger
        self.side = side
        self.device = device
        self._cam = None
        self._fem = None
        self._Matrix4x4 = None
        self._official: dict | None = None
        self._last_depth = None
        # obj pose (contacted object) as 7d [x,y,z,qw,qx,qy,qz]; set by the task.
        self.obj_pose_7d: np.ndarray | None = None

    # ------------------------------------------------------------------
    def setup(self):
        _ensure_display()
        if not OfficialZXTactileSensor._sys_path_added:
            import sys
            sys.path.insert(0, str(XENSE_SIM_ROOT))
            OfficialZXTactileSensor._sys_path_added = True

        # camera (official uses omni.isaac.sensor.Camera). The sensor-camera
        # extension may not be auto-loaded in a headless launch -> enable it.
        try:
            from isaacsim.core.utils.extensions import enable_extension
        except Exception:
            from omni.isaac.core.utils.extensions import enable_extension
        for ext in ("isaacsim.sensors.camera", "omni.isaac.sensor"):
            try:
                enable_extension(ext)
            except Exception:
                pass
        try:
            from isaacsim.sensors.camera import Camera
        except Exception:
            from omni.isaac.sensor import Camera
        try:
            import isaacsim.core.utils.numpy.rotations as rot_utils
        except Exception:
            import omni.isaac.core.utils.numpy.rotations as rot_utils

        params = LEFT_CAM if self.side == "left" else RIGHT_CAM
        self._cam = Camera(
            prim_path=f"{self.finger_prim_path}/{self.side}finger_camera",
            frequency=30,
            translation=np.array(params["translation"]),
            resolution=CAM_RES,
            orientation=rot_utils.euler_angles_to_quats(np.array(params["euler_deg"]), degrees=True),
        )
        self._cam.initialize()
        self._cam.set_projection_mode("orthographic")
        self._cam.add_motion_vectors_to_frame()
        self._cam.set_clipping_range(*CAM_CLIP)
        self._cam.add_distance_to_image_plane_to_frame()
        self._cam.set_horizontal_aperture(CAM_APERTURE)

        # official FEM (g1-ws gel)
        from xensim.core import FemSensor, Matrix4x4
        self._Matrix4x4 = Matrix4x4
        self._fem = FemSensor(
            calibrate_file=str(DEFAULT_CALIB_FILE),
            fem_file=str(DEFAULT_FEM_FILE),
            depth_size=FEM_DEPTH_SIZE,
            visible=False,
            title=f"ZX-{self.side}",
        )
        print(f"[OfficialZX] '{self.name}' ({self.side}) camera+FemSensor ready @ {self.finger_prim_path}")

    # ------------------------------------------------------------------
    def update(self, dt=None, force_recompute=False):
        if self._cam is None:
            self.setup()
        depth = self._cam.get_depth()
        if depth is None:
            return
        # official depth conditioning
        depth = np.nan_to_num(depth, nan=2.0)
        depth = depth - 0.005
        depth = depth.clip(-0.004, 0.001)
        self._last_depth = depth

        pos, quat = self._cam.get_world_pose(camera_axes="usd")
        sensor_mat = self._Matrix4x4.fromVector7d(*pos, *quat)
        if self.obj_pose_7d is not None:
            o = self.obj_pose_7d
            obj_mat = self._Matrix4x4.fromVector7d(*o)
        else:
            obj_mat = sensor_mat  # no object pose -> relative identity (normal only)

        self._fem.step(obj_mat, sensor_mat, depth * 500.0, nstep=3)
        self._fem.update()
        self._official = {
            "image": np.asarray(self._fem.get_image()),
            "diff_image": np.asarray(self._fem.get_diff_image()),
            "marker": np.asarray(self._fem.get_marker()),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
        """Coerce a FemSensor image to a contiguous HWC uint8 RGB array."""
        img = np.asarray(img)
        if img.dtype != np.uint8:
            # FemSensor image is already display-ready; clip floats to 0-255.
            if img.dtype.kind == "f" and img.max() <= 1.0 + 1e-6:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        return np.ascontiguousarray(img)

    def get_observations(self, data_types: list[str] = None):
        if data_types is None:
            data_types = ["rgb", "rgb_marker", "marker", "depth", "pose"]
        obs = {}
        official = self._official or {}
        # H, W match the FEM depth_size (W, H) -> tactile image is (H, W).
        h, w = FEM_DEPTH_SIZE[1], FEM_DEPTH_SIZE[0]
        for dt in data_types:
            if dt in ("rgb", "rgb_marker"):
                img = official.get("image")
                if img is not None:
                    arr = self._to_hwc_uint8(img)
                else:
                    arr = np.zeros((h, w, 3), dtype=np.uint8)
                obs[dt] = torch.from_numpy(arr).to(self.device)
            elif dt == "marker":
                mk = official.get("marker")
                if mk is not None:
                    obs[dt] = torch.from_numpy(np.ascontiguousarray(mk)).float().to(self.device)
                else:
                    obs[dt] = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            elif dt == "depth":
                if self._last_depth is not None:
                    arr = np.ascontiguousarray(self._last_depth.astype(np.float32))
                else:
                    arr = np.zeros((h, w), dtype=np.float32)
                obs[dt] = torch.from_numpy(arr).to(self.device)
            elif dt == "pose":
                if self._cam is not None:
                    pos, quat = self._cam.get_world_pose(camera_axes="usd")
                    obs[dt] = torch.tensor([*pos, *quat], dtype=torch.float32, device=self.device)
                else:
                    obs[dt] = torch.zeros(7, dtype=torch.float32, device=self.device)
        return obs

    def get_min_depth(self):
        # camera depth is distance-to-image-plane (m); after the official offset
        # it is ~[-0.004, 0.001]; deeper contact = more negative. Return mm-ish.
        if self._last_depth is None:
            return 0.0
        return float(self._last_depth.min()) * 1000.0

    def _reset_idx(self):
        pass

    def close(self):
        if self._fem is not None:
            try:
                self._fem.terminate()
            except Exception:
                pass
            self._fem = None
