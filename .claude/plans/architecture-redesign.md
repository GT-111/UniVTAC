# UniVTAC Architecture Redesign

## Benchmark Comparison

| Capability | UniVTAC (now) | ManiSkill | robosuite | RLBench |
|---|---|---|---|---|
| **Physics** | UIPC (soft-body) + PhysX | PhysX GPU | MuJoCo | CoppeliaSim |
| **Tactile** | ✅ GelSight/GF225/ZX | ❌ | ❌ | ❌ |
| **Motion planning** | ✅ cuRobo (collision-free) | ❌ PD controllers | ❌ OSC/impedance | ✅ RRT-Connect |
| **Expert demos** | Scripted via Atom + planner | Teleop | Teleop/keyframes | Waypoints + RRT |
| **Task API** | 🟡 1070-line god class | ✅ 6 methods | ✅ 5 methods | ✅ 4 methods |
| **Robot abstraction** | 🟡 if/elif everywhere | ✅ BaseAgent | ✅ RobotModel | ✅ Robot + planner |
| **Config** | 🟡 4 formats, no validation | ✅ Dataclass | ✅ Dict | ✅ Task-specific |

**Key finding:** UniVTAC's tactile + cuRobo + UIPC stack is unique. The problem is the software architecture, not the simulation backend.

---

## The Core Problem: Robot Coupling

Currently the codebase has **~30 if/elif branches** checking `robot_type` or `tactile_sensor_type`. Every file knows about every embodiment:

```
BaseTask.load_robot_and_sensors():  if gsmini elif gf225 elif zxhand
BaseTask.tactile_overpressed():     if is_zxhand else gsmini_threshold
BaseTask.take_dense_action():       if robot_type == franka_zx_hand ...
BaseTask.adaptive_set_gripper():    if robot_type == franka_zx_hand ...
RobotManager.__init__():            if gsmini/gf225 elif zxhand
RobotManager.setup():               if robot_type == franka_zx_hand ...
RobotManager.get_ee_pose():         if robot_type == franka_zx_hand ...
RobotManager.set_arm():             if robot_type == franka_zx_hand ...
RobotManager.set_gripper():         if robot_type == franka_zx_hand ...
RobotManager.plan_gripper():        if robot_type == franka_zx_hand ...
robot_cfg.py:                       3 separate factory functions
sensors/tactile.py:                 if gsmini elif gf225 elif zxhand
collect.py:                         if gsmini elif gf225 elif zxhand
grasp_classify.py:                  if self.is_zxhand ...
insert_HDMI.py:                     if zxhand else gsmini direction
```

This is the "type-code" anti-pattern. Adding a third embodiment would mean touching 8+ files.

---

## Proposed Architecture — 5 Layers

```
┌──────────────────────────────────────────────────────────┐
│ Layer 5: Scripts & Policies                              │
│   eval_policy.py, collect_data.py                        │
│   BasePolicy (ABC) → ACT, ViTAL, OpenPI, NullPolicy      │
├──────────────────────────────────────────────────────────┤
│ Layer 4: Tasks  ← clean, robot-agnostic                  │
│   @register_task("insert_hole")                          │
│   BaseTask (~350 lines, simulation lifecycle only)       │
│     _load_scene / _load_actors / _initialize_episode     │
│     _pre_move / _play_once / evaluate / _get_obs_extra   │
│   ManipulationPrimitives (extracted, testable)            │
├──────────────────────────────────────────────────────────┤
│ Layer 3: Action System                                   │
│   Atom → builds Action lists (canonical task interface)  │
│   Action → single command (move/gripper/all)             │
│   Planner(ABC) → CuRoboPlanner | LinearPlanner           │
├──────────────────────────────────────────────────────────┤
│ Layer 2: Robot Hierarchy  ← THE KEY REFACTOR             │
│   Robot(ABC)                                             │
│     ├── FrankaGelSight    (gsmini + gf225)              │
│     └── FrankaZXHand      (xense zx hand)               │
│   TactileManager (already abstracted, keep)              │
│   CameraManager (already abstracted, keep)               │
├──────────────────────────────────────────────────────────┤
│ Layer 1: Simulation Foundation  ← KEPT AS-IS             │
│   Isaac Sim 4.5 + UIPC + TacEx + cuRobo                 │
└──────────────────────────────────────────────────────────┘
```

---

## Layer 2: Robot Hierarchy — Design

### The Abstract Robot

```python
# envs/robot/robot.py
class Robot(ABC):
    """Abstract robot: arm + gripper + planner + sensors.
    
    Tasks call these methods. They NEVER branch on robot_type.
    """
    
    config: RobotConfig
    articulation: Articulation
    planner: Planner
    
    # ── State queries (same interface for all embodiments) ──
    
    @abstractmethod
    def get_ee_pose(self) -> Pose: ...
    
    @abstractmethod
    def get_gripper_center_pose(self) -> Pose: ...
    
    @abstractmethod
    def get_qpos(self) -> torch.Tensor: ...
    
    @abstractmethod
    def get_gripper_qpos(self) -> float: ...
    
    # ── Low-level control (different per embodiment) ──
    
    @abstractmethod
    def set_arm(self, qpos, vel=None, force=True): ...
    
    @abstractmethod
    def set_gripper(self, pos, vel=None, force=True): ...
    
    # ── Motion planning ──
    
    def plan_arm(self, target_pose, **kwargs) -> Trajectory:
        """Delegates to self.planner.plan_path(). Same for all embodiments."""
        return self.planner.plan_path(...)
    
    @abstractmethod
    def plan_gripper(self, pos, type) -> GripperPlan:
        """Different per embodiment: prismatic steps vs angular bang-bang."""
        ...
    
    # ── Tactile-aware ──
    
    @abstractmethod
    def is_overpressed(self, tactile_manager) -> bool:
        """Each embodiment knows its own over-pressure condition."""
        ...
    
    @abstractmethod
    def gripper_percent_to_qpos(self, pct: float) -> float:
        """Different ranges: 0.039m prismatic vs 0.99rad angular."""
        ...
    
    # ── Sensor setup (shared pattern, different configs) ──
    
    @abstractmethod
    def create_tactile_configs(self) -> list[TactileCfg]: ...
    
    # ── Adaptive grasp ──
    
    @property
    @abstractmethod
    def uses_adaptive_grasp(self) -> bool: ...
    
    @property
    @abstractmethod
    def adaptive_grasp_step(self) -> float:
        """Step size for adaptive grasp: 0.0005m prismatic vs 0.012rad angular."""
        ...
```

### The Two Implementations

```python
class FrankaGelSight(Robot):
    """Franka + GelSight Mini or GF225 — UIPC soft-body gel simulation.
    
    Gripper is prismatic (0 to 0.039m). PD position control.
    EE pose from Isaac Sim body state. Tactile from UIPC gel deformation.
    """
    
    hand_name = "panda_hand"
    arm_joints = ["panda_joint1", ..., "panda_joint7"]
    gripper_joints = ["panda_finger_joint1", "panda_finger_joint2"]
    gripper_max_qpos = 0.039  # meters
    uses_adaptive_grasp = True
    adaptive_grasp_step = 0.0005  # meters
    
    def get_ee_pose(self) -> Pose:
        return self._body_pose_in_root(self._body_idx)
    
    def set_arm(self, qpos, vel=None, force=True):
        self.articulation.set_joint_position_target(qpos, joint_ids=self._arm_ids)
        if force:
            self.articulation.root_physx_view.set_dof_positions(...)
    
    def set_gripper(self, pos, vel=None, force=True):
        self.articulation.set_joint_position_target(pos, joint_ids=self._gripper_ids)
        ...
    
    def plan_gripper(self, pos, type) -> GripperPlan:
        """Linspace prismatic steps of 0.0005m."""
        ...
    
    def is_overpressed(self, tactile) -> bool:
        return tactile.get_min_depth().min() < self.config.contact_threshold[0]
    
    def gripper_percent_to_qpos(self, pct: float) -> float:
        return self.gripper_max_qpos * pct


class FrankaZXHand(Robot):
    """Franka + Xense ZX Hand — rigid finger with xensim FemSensor tactile.
    
    Gripper is angular (0 to 0.99 rad). Velocity bang-bang control.
    EE pose from cuRobo FK. Tactile from official xensim camera depth.
    """
    
    hand_name = "panda_link8"
    arm_joints = ["panda_joint1", ..., "panda_joint7"]
    gripper_joints = ["right_Left_1_Joint"]  # single driven joint
    gripper_max_qpos = 0.99  # radians
    uses_adaptive_grasp = False  # use plain plan_gripper ramp
    adaptive_grasp_step = 0.012  # radians
    
    def get_ee_pose(self) -> Pose:
        return self._curobo_ee_in_root()  # ZX has no panda_hand body
    
    def set_arm(self, qpos, vel=None, force=True):
        if force:
            # Arm-only teleport — must not touch closed-chain gripper DOFs
            self.articulation.write_joint_state_to_sim(...)
        else:
            self.articulation.set_joint_position_target(...)
    
    def set_gripper(self, pos, vel=None, force=True):
        # Velocity bang-bang: constant speed, never zeroed
        target = float(torch.as_tensor(pos).reshape(-1)[0])
        speed = self._ZX_GRIPPER_SPEED
        if target <= 0.4 * self.gripper_max_qpos:
            speed = -speed
        self.articulation.set_joint_velocity_target(speed, ...)
    
    def plan_gripper(self, pos, type) -> GripperPlan:
        """Constant velocity target for 80 steps — bang-bang control."""
        ...
    
    def is_overpressed(self, tactile) -> bool:
        return tactile.get_min_depth().min() < -3.8  # ZX depth is neg on contact
    
    def gripper_percent_to_qpos(self, pct: float) -> float:
        return self.gripper_max_qpos * pct
    
    def _calibrate_offset(self):
        """ZX-specific: calibrate ee→gripper offset via gel midpoint."""
        ...
```

### Robot Factory

```python
# envs/robot/__init__.py
def create_robot(config: RobotConfig, task) -> Robot:
    """Single dispatch point — the ONLY place that branches on embodiment type."""
    if config.embodiment == "franka_gsmini":
        return FrankaGelSight(config, task, sensor_type="gsmini")
    elif config.embodiment == "franka_gf225":
        return FrankaGelSight(config, task, sensor_type="gf225")
    elif config.embodiment == "franka_zxhand":
        return FrankaZXHand(config, task)
    raise ValueError(f"Unknown embodiment: {config.embodiment}")
```

The `if/elif` chain still exists — but **once**, in a 10-line factory function, not spread across 8 files.

### RobotConfig

```python
@dataclass
class RobotConfig:
    embodiment: Literal["franka_gsmini", "franka_gf225", "franka_zxhand"]
    prim_path: str = "/World/envs/env_.*/Robot"
    planner: Literal["curobo", "linear"] = "curobo"
    planner_time_dilation: float = 1.0
    
    # Embodiment-specific thresholds (set by factory, not by user)
    tactile_far_plane: float = 30.0
    adaptive_grasp_threshold: float | None = None
    contact_threshold: tuple[float, float] = (27.5, 28.0)
```

---

## Layer 4: Task Framework — Adapted from ManiSkill

```python
# envs/_base_task.py
class BaseTask(UipcRLEnv):
    """Simulation lifecycle only. ~350 lines after refactor."""
    
    # ── Class attrs (set by task author or default) ──
    has_pre_move: bool = True  # set False for tasks without grasp phase
    
    # ── Methods task authors OVERRIDE ──
    
    def _load_scene(self, options):        """Static geometry. Called once."""; pass
    def _load_actors(self, options):       """Dynamic objects. Called once."""; pass
    def _initialize_episode(self, env_idx, options): """Pose randomization."""; pass
    def _pre_move(self):                   """Grasp/approach. Skipped if not has_pre_move."""; pass
    def _play_once(self):                  """Task execution."""; pass
    def evaluate(self) -> dict:            """Return {"success": bool}."""; return {"success": False}
    def _get_obs_extra(self, info) -> dict:"""Task-specific obs."""; return {}
    
    # ── Methods task authors CALL ──
    
    def spawn_actor(self, usd_path, name, pose, **kw) -> Actor: ...
    def move(self, actions: list[Action], **kw) -> bool: ...
    def delay(self, steps=20): ...
    def random_pose(self, x_range, y_range) -> Pose: ...
    
    # ── Properties available to task authors ──
    
    @property
    def robot(self) -> Robot: ...
    @property  
    def atom(self) -> Atom: ...
    @property
    def primitives(self) -> ManipulationPrimitives: ...
    @property
    def rng(self) -> np.random.Generator: ...
```

### Task Registry

```python
# envs/_registry.py
_TASK_REGISTRY: dict[str, type[BaseTask]] = {}

def register_task(name: str):
    def decorator(cls):
        _TASK_REGISTRY[name] = cls
        return cls
    return decorator

def make_task(name: str, **kwargs) -> BaseTask:
    return _TASK_REGISTRY[name](**kwargs)
```

### What a new task looks like

```python
@register_task("peg_insertion")
class PegInsertionTask(BaseTask):
    """Insert a peg into a hole."""
    
    SUPPORTED_ROBOTS = [FrankaGelSight, FrankaZXHand]  # both work!
    
    def _load_actors(self):
        self.peg = self.spawn_actor("Peg.usd", "peg",
            pose=Pose([0.35, 0, 0.01]))
        self.hole = self.spawn_actor("Hole.usd", "hole",
            pose=Pose([0.6, 0, 0.002]), static=True)
    
    def _initialize_episode(self, env_idx, options):
        self.peg.set_pose(self.random_pose([0.3, 0.4], [-0.05, 0.05]))
    
    def _pre_move(self):
        self.move(self.atom.grasp_actor(self.peg, pre_dis=0.05))
    
    def _play_once(self):
        self.move(self.atom.place_actor(self.peg, self.hole.get_pose()))
        self.primitives.try_forward(self.peg, dis=0.015)
        self.move(self.atom.open_gripper())
    
    def evaluate(self):
        rel = self.peg.get_pose().rebase(self.hole.get_pose())
        return {"success": bool(rel.p[2] < -0.03 and np.linalg.norm(rel.p[:2]) < 0.005)}
```

**~25 lines. Robot-agnostic. No if/elif. No `tactile_sensor_type` checks.**

---

## Where the if/elif Branches Go

| Current location | After refactor |
|---|---|
| `RobotManager.__init__` (6 branches) | `create_robot()` factory (1 place) |
| `RobotManager.setup()` (2 branches) | `FrankaGelSight.setup()` + `FrankaZXHand.setup()` |
| `RobotManager.get_ee_pose()` (2 branches) | Polymorphic: each subclass implements |
| `RobotManager.set_arm()` (2 branches) | Polymorphic |
| `RobotManager.set_gripper()` (2 branches) | Polymorphic |
| `RobotManager.plan_gripper()` (2 branches) | Polymorphic |
| `BaseTask.load_robot_and_sensors()` (3 branches) | `create_robot()` factory |
| `BaseTask.tactile_overpressed()` (2 branches) | `Robot.is_overpressed()` polymorphic |
| `BaseTask.adaptive_set_gripper()` (2 branches) | Each Robot provides `adaptive_grasp_step` |
| `BaseTask.take_dense_action()` (1 branch) | Fixed on Robot, or moved to Robot method |
| `collect.py` gripper qpos (3 branches) | `robot.gripper_percent_to_qpos()` |
| `grasp_classify.py` adaptive grasp (1 branch) | `robot.uses_adaptive_grasp` property |
| `insert_HDMI.py` grasp direction (1 branch) | Robot provides grasp_direction preference |
| `robot_cfg.py` (3 factory functions) | `create_robot()` + Robot subclasses |

**30+ branches → ~15 polymorphic methods + 1 factory function.**

---

## Other Cross-Cutting Concerns (details from previous version)

### Planner Abstraction

Planner does pure math. `BaseTask.move()` does execution.

```python
class Planner(ABC):
    def plan_arm(self, target_pose, **kw) -> Trajectory: ...
    def update_world(self): ...

class CuRoboPlanner(Planner): ...    # collision-free, production
class LinearPlanner(Planner): ...    # straight-line, testing
```

### Manipulation Primitives

```python
class ManipulationPrimitives:
    """Pure logic, injectable Robot + step_fn. Testable without Omniverse."""
    def __init__(self, robot: Robot, step_fn): ...
    def adaptive_grasp(self, target, threshold) -> Generator: ...
    def gravity_align(self, actor, target_vec) -> bool: ...
    def rotate_in_hand(self, actor, theta, steps=6) -> bool: ...
    def try_forward(self, actor, dis, delta) -> bool: ...
```

### Atom — Canonical Interface

ALL 9 task modules use `self.atom.xxx()` exclusively. Raw `Action(...)` only in `_base_task.py`. The design formalizes this: tasks use Atom, Atom returns `list[Action]`, `move()` accepts `list[Action]`.

### BasePolicy — Deepen

`BasePolicy` becomes an ABC with `@abstractmethod encode_obs()` and `@abstractmethod eval()`. Shared `task_settings` loading moves to `BasePolicy.__init__()`. ACT/Ablation merged.

### Scripts — Shared Core

`get_config()` extracted to `scripts/_eval_core.py`. Both `eval_policy.py` and `parallel_eval_policy.py` import from it.

---

## Migration Plan

```
Phase 1: Robot Hierarchy (foundation for everything)
├── 1a. Create Robot ABC + RobotConfig          [1 new file]
├── 1b. Extract FrankaGelSight from RobotManager [1 new file, ~200 lines]
├── 1c. Extract FrankaZXHand from RobotManager  [1 new file, ~250 lines]
├── 1d. Create create_robot() factory           [modify robot/__init__.py]
├── 1e. Create Planner ABC (extract from cuRobo) [1 new file]
└── 1f. Fix envs/utils/__init__.py bare-except   [1 line fix]

Phase 2: Task Framework
├── 2a. Create @register_task + make_task()     [1 new file]
├── 2b. Create ManipulationPrimitives           [1 new file]
├── 2c. Refactor BaseTask (~1070→~350 lines)    [modify _base_task.py]
└── 2d. Migrate 8 tasks to new API              [8 files, 1 per task]

Phase 3: Policies & Scripts
├── 3a. Deepen BasePolicy (ABC + shared init)   [modify _base_policy.py + 5 policies]
├── 3b. Collapse ACT/Ablation                   [delete policy/Ablation/]
├── 3c. Extract TemporalAggregationEngine       [1 new file]
├── 3d. Extract scripts/_eval_core.py           [1 new file]
└── 3e. Move SigLIP patch to runtime             [modify OpenPI/deploy_policy.py]

Phase 4: Tests
└── 4a. tests/ with pytest                      [new dir, ongoing]
       - test_robot_config.py
       - test_primitives.py
       - test_temporal_agg.py
       - test_observation_encoding.py
```

---

## Resolved Design Decisions

From the user's feedback on the previous draft:

1. **Planner scope:** Planner = pure math (`plan_arm`, `plan_gripper`). BaseTask.move() = execution loop. `adaptive_set_gripper` = closed-loop controller → lives in ManipulationPrimitives.

2. **pre_move:** Configurable via `has_pre_move: bool = True` class attribute. Default runs both `_pre_move()` then `_play_once()`. Tasks that don't need a grasp phase set `has_pre_move = False`.

3. **Atom canonical:** Confirmed. All 9 task modules use Atom exclusively. Raw `Action(...)` only in `_base_task.py`. Atom IS the interface for tasks; direct Action construction allowed only inside Atom and primitives.

4. **Migration:** All 8 tasks migrated. Order: `_base_task` → `_primitives` → `_registry` → then tasks one by one.
