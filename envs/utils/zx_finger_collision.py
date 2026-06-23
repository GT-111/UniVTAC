"""UIPC FEM gel-block proxy for ZX hand.

add_reference_to_stage loads pre-baked USD → Fabric → FEM init works.
Block at dummy start, per-vertex SoftPositionConstraint teleports to
gel surface (FWD_Y=6mm ensures cover of gel contact at Z≈43mm).

This is the only path that passes global_vertex_offset (programmatic
DefinePrim or robot-USD embedding both fail).
"""

from __future__ import annotations
import numpy as np, omni, omni.usd, torch, weakref
from pxr import Gf, UsdGeom
import isaaclab.sim as sim_utils, isaaclab.utils.math as math_utils
from tacex_uipc.objects import UipcObject, UipcObjectCfg
from tacex_uipc.sim import UipcSim
from uipc import Animation, builtin, view
from uipc.constitution import SoftPositionConstraint
from ..sensors.zx_official import LEFT_CAM_LOCAL, RIGHT_CAM_LOCAL
from typing import TYPE_CHECKING
if TYPE_CHECKING: from .._base_task import BaseTask

_HALF = 0.003; _STR = 1e4; _YOUNGS = 0.1


class ZxFingerCollisionProxy:
    def __init__(self, task, side):
        self.task = task; self.side = side
        self.body = f"xense_{side}finger"
        self._cam = LEFT_CAM_LOCAL if side == "left" else RIGHT_CAM_LOCAL
        self._obj = None; self._bid = None; self._att = []; self._off = None
        self._aim = np.zeros(0); self._ok = False; self._hp = self._hs = None

    @property
    def g(self): return f"/World/envs/env_.*/zx_{self.side}_proxy"
    @property
    def c(self): return f"/World/envs/env_0/zx_{self.side}_proxy"

    def create_usd_prim(self, stage):
        from pathlib import Path
        from isaacsim.core.utils.stage import add_reference_to_stage
        gel = str(Path(__file__).resolve().parents[2] / "assets" / "zx_gel_block.usd")
        add_reference_to_stage(usd_path=gel, prim_path=self.c)
        # Place at gel-surface WORLD pose (not dummy).
        # Must be via add_reference_to_stage for FEM init to work.
        fp = f"/World/envs/env_0/Robot/{self.body}"
        fprim = stage.GetPrimAtPath(fp)
        if fprim.IsValid():
            T = np.array(UsdGeom.Xformable(fprim)
                         .ComputeLocalToWorldTransform(0)).reshape(4,4).T
            gel_pos = T[:3,3] + T[:3,:3] @ self._cam
        else:
            gel_pos = (0, 0.05, 0.05)
        UsdGeom.Xformable(stage.GetPrimAtPath(self.c)).AddTranslateOp().Set(
            Gf.Vec3d(float(gel_pos[0]), float(gel_pos[1]), float(gel_pos[2])))

    def build(self, uipc_sim):
        cfg = UipcObjectCfg(prim_path=self.g,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=_YOUNGS), mass_density=1e3)
        self._obj = UipcObject(cfg, uipc_sim)
        SoftPositionConstraint().apply_to(self._obj.uipc_meshes[0], _STR)
        self.task.scene.uipc_objects[f"zx_{self.side}_proxy"] = self._obj
        sim_utils.SimulationContext.instance().add_physics_callback(
            f"zx_{self.side}_sync", self._sync)
        s = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._hp = s.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda e,o=weakref.proxy(self): o._on_play(e), order=10)
        self._hs = s.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda e,o=weakref.proxy(self): o._on_stop(e), order=10)

    def _on_play(self, _):
        if not self._ok: self._init(); self._ok = True
    def _on_stop(self, _):
        self._ok = False; self._bid = None; self._att = []
        self._off = None; self._aim = np.zeros(0)

    def _init(self):
        r = self.task._robot_manager.robot
        ids,_ = r.find_bodies(self.body); self._bid = int(ids[0])
        m = self._obj.uipc_meshes[0]; v = m.positions().view()[:,:,0].copy()
        T = self._obj.init_world_transform.cpu().numpy()
        vl = (v - T[:3,3]) @ T[:3,:3]; yb = vl[:,1]
        mk = yb < (-_HALF + 1e-4) if self.side == "left" \
             else yb > (_HALF - 1e-4)
        self._att = [int(i) for i in np.where(mk)[0]]
        # Block already at gel surface world pose — offsets from actual positions
        fp,fq = self._get()
        Rf = math_utils.matrix_from_quat(
            torch.tensor(fq, dtype=torch.float64)).cpu().numpy()
        self._off = ((v[self._att] - fp) @ Rf).astype(np.float64)
        a = self._obj._uipc_sim.scene.animator()
        a.insert(self._obj.uipc_scene_objects[0], self._anim)

    def _get(self):
        r = self.task._robot_manager.robot
        p = r._root_physx_view.get_link_transforms().clone()
        p[...,3:7] = math_utils.convert_quat(p[...,3:7], to="wxyz")
        return (p[0,self._bid,:3].cpu().numpy().astype(np.float64),
                p[0,self._bid,3:7].cpu().numpy().astype(np.float64))

    def _sync(self, dt=0):
        if not self._ok or self._bid is None: return
        if self._off is None or len(self._off)==0: return
        fp,fq = self._get()
        ot = torch.tensor(self._off, dtype=torch.float64)
        ft = torch.tensor(fp).unsqueeze(0); qt = torch.tensor(fq).unsqueeze(0)
        self._aim = math_utils.transform_points(
            ot.unsqueeze(0), pos=ft, quat=qt).squeeze(0).cpu().numpy().astype(np.float64)

    def _anim(self, info):
        if len(self._aim)==0: return
        gs = info.geo_slots()
        if not gs: return
        g = gs[0].geometry()
        ic = g.vertices().find(builtin.is_constrained); view(ic)[:] = 0
        view(ic)[self._att] = 1
        view(g.vertices().find(builtin.aim_position))[self._att] = \
            self._aim.reshape(-1,3,1)

    def __del__(self):
        if self._hp: self._hp.unsubscribe()
        if self._hs: self._hs.unsubscribe()


class ZxFingerCollisionManager:
    def __init__(self, t):
        self.task = t
        self.left = ZxFingerCollisionProxy(t,"left")
        self.right = ZxFingerCollisionProxy(t,"right")
        self._rdy = False
    def create_usd_prims(self):
        s = omni.usd.get_context().get_stage()
        self.left.create_usd_prim(s); self.right.create_usd_prim(s)
    def create_uipc_objects(self):
        self.left.build(self.task.uipc_sim)
        self.right.build(self.task.uipc_sim)
        self._rdy = True
    def setup(self): pass
    def reset(self):
        if self._rdy: self.left._aim = np.zeros(0); self.right._aim = np.zeros(0)
