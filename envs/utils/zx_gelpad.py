"""ZX hand UIPC collision proxies — GelSight-aligned SoftPositionConstraint pattern.

Identical constraint mechanism as GelSight gel pads (UipcIsaacAttachments
applies SoftPositionConstraint + animator-driven aim_position).  The only
difference is attachment-vertex selection: we pick the back-face vertices
geometrically instead of via PhysX sweep.
"""

from __future__ import annotations

import weakref
from pathlib import Path

import numpy as np
import omni.timeline
import omni.usd
import torch
from pxr import Gf, UsdGeom
from uipc import builtin, view
from uipc.constitution import SoftPositionConstraint

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from tacex_uipc.objects import UipcObject, UipcObjectCfg

from ..sensors.zx_official import LEFT_CAM_LOCAL, RIGHT_CAM_LOCAL

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._base_task import BaseTask

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"

_YOUNGS = 0.1   # StableNeoHookean, same as gsmini/gf225
_DENSITY = 1e3   # mass_density, matches old ZxFingerCollisionProxy
_STRENGTH = 1e4  # constraint strength, same as old code
_HALF = 0.003    # gel block half-thickness in Y (6mm total)
_RELIEVE = 3e-4  # 0.3mm relief from finger body to avoid Fabric overlap


class ZxGelpad:
    """FEM gel pad tracking a ZX hand finger body — same SoftPositionConstraint
    + animation aim_position pattern as GelSight UipcIsaacAttachments."""

    def __init__(self, task: BaseTask, side: str):
        self.task = task
        self.side = side
        self._body = f"xense_{side}finger"
        self._prim = f"/World/envs/env_.*/zx_{side}_proxy"
        self._prim0 = f"/World/envs/env_0/zx_{side}_proxy"
        self._cam_local = LEFT_CAM_LOCAL if side == "left" else RIGHT_CAM_LOCAL

        self._obj: UipcObject | None = None
        self._bid: int | None = None
        self._att: list[int] = []
        self._off: np.ndarray = np.zeros(0)  # [N,3] body-local offsets
        self._aim: np.ndarray = np.zeros(0)   # [N,3] current aim world positions
        self._ok = False
        self._hp = self._hs = None

    # ── USD prim (before env clone) ────────────────────────────────

    def create_usd_prim(self, stage):
        from isaacsim.core.utils.stage import add_reference_to_stage

        usd_path = str(_ASSETS_DIR / "zx_gel_block.usd")
        add_reference_to_stage(usd_path=usd_path, prim_path=self._prim0)
        # Near gel-surface [~0.375, ±0.003, 0.195] — only ~5mm teleport.
        # Dummy [0,±0.05,0.05] → gel is 38cm jump → huge FEM pre-strain.
        y = 0.015 if self.side == "left" else -0.015
        UsdGeom.Xformable(stage.GetPrimAtPath(self._prim0)).AddTranslateOp().Set(
            Gf.Vec3d(0.37, y, 0.19)
        )

    # ── UIPC objects (after env clone, before setup_sim) ───────────

    def build(self, uipc_sim):
        cfg = UipcObjectCfg(
            prim_path=self._prim,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=_YOUNGS),
            mass_density=_DENSITY,
        )
        self._obj = UipcObject(cfg, uipc_sim)
        SoftPositionConstraint().apply_to(
            self._obj.uipc_meshes[0], _STRENGTH)
        self.task.scene.uipc_objects[f"zx_{self.side}_proxy"] = self._obj

        # Hide the USD prim from cameras: the ZX hand uses an external
        # ortho depth camera that would see the FEM block and corrupt
        # tactile readings (GelSight uses an in-gel camera, so it doesn't
        # have this problem).
        try:
            stage = omni.usd.get_context().get_stage()
            prim0 = stage.GetPrimAtPath(self._prim0)
            if prim0.IsValid():
                UsdGeom.Imageable(prim0).MakeInvisible()
        except Exception:
            pass

        sim_utils.SimulationContext.instance().add_physics_callback(
            f"zx_{self.side}_sync", self._sync)

        s = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._hp = s.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda e, o=weakref.proxy(self): o._on_play(e), order=10)
        self._hs = s.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda e, o=weakref.proxy(self): o._on_stop(e), order=10)

    # ── PLAY / STOP ───────────────────────────────────────────────

    def _on_play(self, _):
        if not self._ok:
            self._init()
            self._ok = True

    def _on_stop(self, _):
        self._ok = False
        self._bid = None
        self._att = []
        self._off = np.zeros(0)
        self._aim = np.zeros(0)

    def _init(self):
        """Body-local offsets targeting gel-surface cam_local position.
        Block near gel surface → ~1cm teleport → minimal pre-strain."""
        r = self.task._robot_manager.robot
        ids, _ = r.find_bodies(self._body)
        self._bid = int(ids[0])

        m = self._obj.uipc_meshes[0]
        v = m.positions().view()[:, :, 0].copy()
        T_obj = self._obj.init_world_transform.cpu().numpy()
        vl = (v - T_obj[:3, 3]) @ T_obj[:3, :3]
        yb = vl[:, 1]
        mk = (yb < (-_HALF + 1e-4)) if self.side == "left" \
             else (yb > (_HALF - 1e-4))
        self._att = [int(i) for i in np.where(mk)[0]]

        # cam_local + vl_rel → blocks placed at gel surface, follow finger.
        fp, fq = self._get()
        Rf = math_utils.matrix_from_quat(
            torch.tensor(fq, dtype=torch.float64)).cpu().numpy()
        vl_body = (Rf.T @ vl[self._att].T).T
        vl_rel = vl_body - vl_body.mean(axis=0)
        self._off = np.zeros((len(self._att), 3), dtype=np.float64)
        self._off[:, 0] = self._cam_local[0] + vl_rel[:, 0]
        self._off[:, 1] = self._cam_local[1] + vl_rel[:, 1]
        self._off[:, 2] = self._cam_local[2] + vl_rel[:, 2]

        a = self._obj._uipc_sim.scene.animator()
        a.insert(self._obj.uipc_scene_objects[0], self._anim)

    # ── body pose helper ──────────────────────────────────────────

    def _get(self):
        r = self.task._robot_manager.robot
        p = r._root_physx_view.get_link_transforms().clone()
        p[..., 3:7] = math_utils.convert_quat(p[..., 3:7], to="wxyz")
        return (p[0, self._bid, :3].cpu().numpy().astype(np.float64),
                p[0, self._bid, 3:7].cpu().numpy().astype(np.float64))

    # ── physics callback + animator (same pattern as UipcIsaacAttachments) ──

    def _sync(self, dt=0):
        """Physics callback: compute aim_positions from body pose."""
        if not self._ok or self._bid is None or len(self._off) == 0:
            return
        fp, fq = self._get()
        ft = torch.tensor(fp).unsqueeze(0)
        qt = torch.tensor(fq).unsqueeze(0)
        ot = torch.tensor(self._off, dtype=torch.float64).unsqueeze(0)
        self._aim = (math_utils.transform_points(ot, pos=ft, quat=qt)
                     .squeeze(0).cpu().numpy().astype(np.float64))

    def _anim(self, info):
        """Animation: write is_constrained=1 + aim_position (same as GelSight)."""
        if len(self._aim) == 0:
            return
        gs = info.geo_slots()
        if not gs:
            return
        g = gs[0].geometry()
        ic = g.vertices().find(builtin.is_constrained)
        view(ic)[:] = 0
        view(ic)[self._att] = 1
        view(g.vertices().find(builtin.aim_position))[self._att] = \
            self._aim.reshape(-1, 3, 1)

    def __del__(self):
        if self._hp:
            self._hp.unsubscribe()
        if self._hs:
            self._hs.unsubscribe()


# ═══════════════════════════════════════════════════════════════════════
# ZxRodProxy — disabled (no mesh yet)
# ═══════════════════════════════════════════════════════════════════════

class ZxRodProxy:
    def __init__(self, task, body_name):
        pass
    def create_usd_prim(self, stage):
        pass
    def build(self, uipc_sim):
        pass


# ═══════════════════════════════════════════════════════════════════════
# ZxGelpadManager
# ═══════════════════════════════════════════════════════════════════════

_ROD_BODIES: list[str] = [
    "right_Left_0_Joint",
    "right_Right_0_Joint",
    "right_Left_Support_Joint",
    "right_Right_Support_Joint",
]


class ZxGelpadManager:
    def __init__(self, t):
        self.task = t
        self.left = ZxGelpad(t, "left")
        self.right = ZxGelpad(t, "right")
        self.rods = [ZxRodProxy(t, body) for body in _ROD_BODIES]
        self._rdy = False

    def create_usd_prims(self):
        s = omni.usd.get_context().get_stage()
        self.left.create_usd_prim(s)
        self.right.create_usd_prim(s)
        for rod in self.rods:
            rod.create_usd_prim(s)

    def create_uipc_objects(self):
        u = self.task.uipc_sim
        self.left.build(u)
        self.right.build(u)
        for rod in self.rods:
            rod.build(u)
        self._rdy = True

    def setup(self):
        pass

    def reset(self):
        if self._rdy:
            self.left._aim = np.zeros(0)
            self.right._aim = np.zeros(0)
