# ZX Hand UIPC Bridge — Issues Log

## Context

ZX hand fingers are PhysX articulation links; grasped objects (Actors) are
UIPC FEM deformable bodies. Without a common physics representation, the
IPC solver cannot detect contact → fingers pass through objects.

We need a UIPC body per finger, driven to track the PhysX finger pose, to
bridge the two physics engines.  GelSight Mini does this via pre-baked
FEM gel pads in the robot USD + per-vertex SoftPositionConstraint +
UipcIsaacAttachments.  We tried to replicate this pattern for ZX hand.

---

## Issue 1: FEM block init fails with `global_vertex_offset`

**Symptom**: `RuntimeError: Failed to initialize global_vertex_offset` in
`UipcSim.setup_sim()`.  The `FiniteElementMethod` UIPC backend does not
assign a global vertex ID to our gel block.

**Root cause**: Procedurally-created USD prims (via `stage.DefinePrim`)
are not registered with Fabric.  The UIPC backend reads prim data through
Fabric; without Fabric registration, `world.init()` silently skips the
object.  `AssetBase._initialize_callback` swallows the exception and sets
`_is_initialized = True` unconditionally, permanently bricking the object.

**Fix attempted / outcome**:
- `stage.DefinePrim` → ❌ always fails
- `add_reference_to_stage` (Fabric path) → ✅ works (FEM initializes)
- Robot USD embedding (gel block prims inside `franka_zx_hand_real.usd`) → ❌ fails
- Placing block at gel-surface world pose → ❌ fails
- Placing block at dummy position `(0, ±0.05, 0.05)` → ✅ works (FEM initializes)

**Only working init path**: `add_reference_to_stage` + dummy start position.

---

## Issue 2: FEM block cannot teleport 700mm fast enough

**Symptom**: When the FEM block starts at dummy position and per-vertex
SoftPositionConstraint targets gel surface (37mm from finger body, ~700mm
dummy-to-target gap), the block takes dozens of physics steps to converge.
The finger reaches the object long before the FEM block arrives → apparent
penetration.

**Root cause**: StableNeoHookean (0.1 MPa stiffness, 217 vertices / 660
tets) resists instantaneous deformation.  Even with `1e4` constraint
strength, converging a 700 mm displacement through FEM deformation takes
many IPC solve iterations.

**Contrast with gsmini**: GelSight gel pad starts AT the correct world
position (it's a child of the robot articulation in the USD).  The
SoftPositionConstraint does maintenance only (sub-millimeter deltas), not
teleport.

**Attempted workarounds**:
- `1e6` constraint strength → FEM solver instability
- Forward Y bias (`FWD_Y=6mm`) → helped reach gel surface but block
  geometry alters grasp semantics
- Starting block at gel-surface world pose → FEM init fails (Issue 1)

---

## Issue 3: Thin gel block does not cover entire finger

**Symptom**: Even with FEM block working, finger surfaces outside the gel
contact patch (finger body sides, tip, back) are NOT represented in UIPC
→ PhysX finger passes through UIPC objects in those regions.

**Root cause**: The gel block (20×6×54mm, positioned at gel surface) only
covers a small fraction of the finger volume.  The ZX finger collision
mesh is 20×25.5×53.8mm — much larger than the gel patch.  Our proxy
misses ~80% of the finger surface.

**Contrast with gsmini**: GelSight gel pad covers the ENTIRE gripping
surface.  The gel case behind the pad never contacts objects.  UIPC gel
pad → full coverage.

**Current state**: Using full-finger-size affine body (20×25.5×54mm) with
SoftTransformConstraint.  This gives full finger coverage but loses FEM
deformability.  Seed 8 success: 60s, 861 steps.  However penetration may
still occur because of Issue 2 (constraint lag).

---

## Issue 4: `AssetBase._initialize_callback` silently swallows exceptions

**Symptom**: `_initialize_impl` failures produce no visible error until
`setup_sim()` hits `global_vertex_offset` — by which point the root cause
is lost.

**Root cause**: `/home/a25278/Workspaces/IsaacLab/source/isaaclab/isaaclab/assets/asset_base.py:296`:
`self._is_initialized = True` is **unconditional** — executes after the
try/except regardless of whether `_initialize_impl` succeeded.  The object
is permanently bricked; `setup_sim()` sees it missing from `uipc_objects`.

**Fix**: Move `self._is_initialized = True` inside the try block, after
`_initialize_impl()`, so failure allows retry on next PLAY event.

**Status**: Not fixed (upstream Isaac Lab issue).  Workaround: ensure
`_initialize_impl` never fails (Issue 1 fix).

---

## Issue 5: Pre-baked USD file fragility

**Symptom**: `assets/zx_gel_block.usd` randomly corrupted during
development (tet data lost, wrong dimensions, missing `defaultPrim`).

**Root cause**: Generation script evolved through many iterations;
intermediate failures left incomplete files.  The file must have
`defaultPrim` set for `add_reference_to_stage` to work.  Tet attribute
types must match gsmini's `Gelpad_high_res.usd` exactly (`uint[]` for
indices, `float3[]` for points).

**Current state**: File regenerated successfully:
- 20×2mm variant: 159v/369t (stopped work due to thinness)
- 20×6mm variant: 217v/660t (current working copy)
- Generation requires Kit runtime (`isaaclab.app.AppLauncher`)

---

## Issue 6: Robot USD path override causes Fabric inconsistency

**Symptom**: When `create_franka_zx_hand_gripper` overrides the robot
USD to a custom path (`franka_zx_hand_with_gel.usd`), the ZX articulation
still loads but gel blocks inside the robot hierarchy fail FEM init.

**Root cause**: Robot USD is loaded via `UsdFileCfg.spawn_from_usd()` →
`add_reference_to_stage` which creates a USD composition arc.  Gel blocks
created as children of `xense_leftfinger`/`xense_rightfinger` inherit
RigidBodyAPI from the articulation, which UIPC rejects.

**Current state**: Custom USD exists but is NOT used.  Bridge creates
prims via `add_reference_to_stage` at `/World/envs/env_0/zx_*_proxy`
(outside Robot hierarchy).

---

## Working Approaches

### A. FEM + per-vertex (gsmini-aligned) — `add_reference_to_stage` + dummy pos
- Init: ✅ (dummy position only)
- Full finger coverage: ❌ (6mm thin block)
- Seed 8: ~106s, Plan True, Check False (FWD_Y=6mm needed)

### B. AffineBody + SoftTransformConstraint + gel offset 37mm
- Init: ✅ (always works)
- Full finger coverage: only at gel surface
- Seed 8: 65s success (thin panel 2mm at gel surface)

### C. AffineBody + full finger mesh + SoftTransformConstraint
- Init: ✅
- Full finger coverage: ✅ (20×25.5×54mm)
- Seed 8: 60s success (but penetration still observed due to constraint lag)

---

## Key Files

| File | Purpose |
|------|---------|
| `envs/utils/zx_finger_collision.py` | Bridge: creates UIPC proxy, hooks physics callback + animator |
| `assets/zx_gel_block.usd` | Pre-baked FEM tet mesh (20×6mm, 217v/660t) |
| `assets/franka_zx_hand_with_gel.usd` | Custom robot USD with gel blocks (not currently used) |
| `envs/robot/robot_cfg.py` | Robot config; uses original `franka_zx_hand_real.usd` |
| `envs/sensors/zx_official.py` | Official ZX tactile sensor; exports `LEFT_CAM_LOCAL`/`RIGHT_CAM_LOCAL` |
| `envs/sensors/tactile.py` | TactileManager; sets `obj_pose_7d` for FemSensor |
| `envs/_base_task.py` | BaseTask; bridge lifecycle integration + `manipulated_actor_name` |

## Next Steps

1. [ ] Fix `AssetBase._initialize_callback` in Isaac Lab (upstream) —
   move `_is_initialized = True` inside try block
2. [ ] Investigate why gel-surface world pose breaks FEM init — may be a
   UIPC sanity check rejecting geometry too close to ground plane
3. [ ] Consider pre-generating a "full finger" FEM block USD that spans
   the actual finger + gel area combined (20×25.5×60mm) for complete coverage
4. [ ] Test with higher constraint strength on AffineBody to eliminate
   penetration (currently 100.0 → try 1e4 or 1e6)
5. [ ] Run multi-seed evaluation (seeds 0-15) on affine body approach
   to establish baseline success rate
