"""UIPC collision proxy for ZX hand fingers — thin contact panel.

The ZX hand uses camera-based tactile sensing (FemSensor), not UIPC
deformable gel pads.  Objects live in UIPC (for Actor compatibility)
while fingers are PhysX articulation links.  Without a UIPC body on
the finger side the IPC solver cannot detect contact → fingers pass
through objects.

This module creates a THIN (2 mm) invisible affine-body panel for
each finger, positioned precisely at the gel contact surface, driven
by ``SoftTransformConstraint`` to track the PhysX finger body.

The panel is deliberately thin so it does **not** block the FemSensor
camera view and is hidden from rendering (the visual finger geometry
already renders the correct shape).

Pattern: follows ``UipcIsaacAttachments`` (GelSight) callback design
but with an affine body driven by ``SoftTransformConstraint`` instead
of per-vertex ``SoftPositionConstraint``.
"""

from __future__ import annotations

import numpy as np
import omni
import omni.usd
import torch
import weakref

from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from tacex_uipc.objects import UipcObject, UipcObjectCfg
from tacex_uipc.sim import UipcSim

from uipc import Animation, builtin, view
from uipc.constitution import SoftTransformConstraint

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._base_task import BaseTask

# ── Contact-panel geometry ────────────────────────────────────────────
# Thin panel approximating the ZX finger gel contact area.
# Dimensions ~ 20 mm (X) × 2 mm (Y) × 54 mm (Z)
_PANEL_HALF_W = 0.010   # x
_PANEL_HALF_H = 0.001   # y — 2 mm thick (thin panel, not blocky)
_PANEL_HALF_D = 0.027   # z


def _create_panel_mesh_prim(stage, prim_path: str):
    """Create a USD Xform+Mesh prim for the thin collision panel.

    No pre-computed tet data — ``UipcObject`` will call ``MeshGenerator``.
    Returns the Xform prim.
    """
    # 8 vertices, centered at origin
    hx, hy, hz = _PANEL_HALF_W, _PANEL_HALF_H, _PANEL_HALF_D
    verts = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ])
    tris = np.array([
        [0,1,3],[1,2,3], [4,6,5],[4,7,6],   # ±z
        [0,4,5],[0,5,1], [1,5,6],[1,6,2],   # ±x
        [2,6,7],[2,7,3], [3,7,4],[3,4,0],   # ±y
    ], dtype=np.uint32)

    xform = stage.DefinePrim(str(prim_path), "Xform")
    mesh_path = str(prim_path) + "/mesh"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.GetPointsAttr().Set(verts)
    mesh.GetFaceVertexCountsAttr().Set([3] * len(tris))
    mesh.GetFaceVertexIndicesAttr().Set(tris.flatten())
    # Hide from rendering — the real finger geometry is already visible
    mesh.MakeInvisible()
    return xform


# ── Proxy ─────────────────────────────────────────────────────────────

class ZxFingerCollisionProxy:
    """UIPC collision proxy for one ZX finger.

    Lifecycle (GelSight ``UipcIsaacAttachments`` pattern):
    1. ``create_usd_prim()``  — USD panel prim in env_0 (before clone)
    2. ``build()``            — UipcObject + constraint + PLAY subscription
    3. ``[sim.reset()]``      — PLAY → animator.insert() BEFORE world.init()
    4. ``[sim.step()]``       — physics callback → _animate → IPC solver
    """

    def __init__(self, task: BaseTask, side: str):
        self.task = task
        self.side = side
        self.body_name = f"xense_{side}finger"
        self._uipc_obj: UipcObject | None = None
        self._body_id: int | None = None
        self._next_mat: np.ndarray | None = None
        self._box_init_world: torch.Tensor | None = None
        self._is_initialized = False
        self._initialize_handle = None
        self._invalidate_handle = None

    @property
    def prim_path_glob(self) -> str:
        return f"/World/envs/env_.*/zx_{self.side}finger_proxy"

    @property
    def prim_path_concrete(self) -> str:
        return f"/World/envs/env_0/zx_{self.side}finger_proxy"

    # -- Phase 1: USD prim (before clone) --

    def create_usd_prim(self, stage):
        prim = _create_panel_mesh_prim(stage, self.prim_path_concrete)
        # Start above ground to pass UIPC sanity checks
        xform = UsdGeom.Xformable(prim)
        y_offs = 0.05 if self.side == "left" else -0.05
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, y_offs, 0.05))

    # -- Phase 2: UIPC object + callbacks (after clone, before setup_sim) --

    def build(self, uipc_sim: UipcSim):
        from isaaclab.assets import AssetBaseCfg

        cfg = UipcObjectCfg(
            prim_path=self.prim_path_glob,
            spawn=None,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
            ),
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(
                m_kappa=100.0,
                kinematic=False,  # driven via SoftTransformConstraint
            ),
            mass_density=1e3,
        )
        self._uipc_obj = UipcObject(cfg, uipc_sim)

        SoftTransformConstraint().apply_to(
            self._uipc_obj.uipc_meshes[0], np.array([100.0, 100.0]))

        self._box_init_world = self._uipc_obj.init_world_transform.clone()

        self.task.scene.uipc_objects[
            f"zx_{self.side}finger_proxy"] = self._uipc_obj

        # Physics callback — registered BEFORE "uicp_step"
        sim_utils.SimulationContext.instance().add_physics_callback(
            f"zx_finger_sync_{self.side}", self._sync_callback)

        # PLAY timeline subscription (GelSight pattern)
        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._initialize_handle = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda ev, obj=weakref.proxy(self): obj._on_play(ev), order=10)
        self._invalidate_handle = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda ev, obj=weakref.proxy(self): obj._on_stop(ev), order=10)

    # -- PLAY / STOP callbacks --

    def _on_play(self, _event):
        if not self._is_initialized:
            self._init_impl()
            self._is_initialized = True

    def _on_stop(self, _event):
        self._is_initialized = False
        self._body_id = None

    def _init_impl(self):
        robot = self.task._robot_manager.robot
        body_ids, _ = robot.find_bodies(self.body_name)
        self._body_id = body_ids[0]
        animator = self._uipc_obj._uipc_sim.scene.animator()
        animator.insert(self._uipc_obj.uipc_scene_objects[0], self._animate)

    # -- Animation callback (UIPC side) --

    def _animate(self, info: Animation.UpdateInfo):
        if self._next_mat is None:
            return
        slots = info.geo_slots()
        if not slots:
            return
        geo = slots[0].geometry()
        view(geo.instances().find(builtin.is_constrained))[:] = 1
        view(geo.instances().find(builtin.aim_transform))[:] = self._next_mat
        self._next_mat = None

    # -- Physics callback (PhysX → UIPC sync) --

    def _sync_callback(self, dt=0):
        if not self._is_initialized or self._body_id is None:
            return
        robot = self.task._robot_manager.robot
        poses = robot._root_physx_view.get_link_transforms().clone()
        poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
        curr = poses[:, self._body_id, 0:7]  # [N, 7]

        N = curr.shape[0]
        device = curr.device
        R_cur = math_utils.matrix_from_quat(curr[:, 3:7])
        T_cur = torch.zeros(N, 4, 4, device=device, dtype=torch.float64)
        T_cur[:, 0:3, 0:3] = R_cur
        T_cur[:, 0:3, 3] = curr[:, 0:3]
        T_cur[:, 3, 3] = 1.0

        box_init = self._box_init_world.to(device=device, dtype=torch.float64)
        self._next_mat = (T_cur @ torch.inverse(box_init).unsqueeze(0)).cpu().numpy()

    def __del__(self):
        if self._initialize_handle:
            self._initialize_handle.unsubscribe()
        if self._invalidate_handle:
            self._invalidate_handle.unsubscribe()


# ── Manager ────────────────────────────────────────────────────────────

class ZxFingerCollisionManager:
    """Orchestrates UIPC collision proxies for both ZX fingers."""

    def __init__(self, task: BaseTask):
        self.task = task
        self.left = ZxFingerCollisionProxy(task, "left")
        self.right = ZxFingerCollisionProxy(task, "right")
        self._ready = False

    def create_usd_prims(self):
        stage = omni.usd.get_context().get_stage()
        self.left.create_usd_prim(stage)
        self.right.create_usd_prim(stage)

    def create_uipc_objects(self):
        self.left.build(self.task.uipc_sim)
        self.right.build(self.task.uipc_sim)
        self._ready = True

    def setup(self):
        pass  # init is automatic via PLAY subscription

    def reset(self):
        if self._ready:
            self.left._next_mat = None
            self.right._next_mat = None
