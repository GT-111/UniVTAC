# UniVTAC Architecture

## Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     univtac CLI                          │
│  src/univtac/cli/main.py                                │
│  eval | collect | list | validate                       │
└──────────┬──────────────────────────────────┬───────────┘
           │                                  │
     ┌─────▼──────┐                   ┌──────▼────────┐
     │ eval_policy │                   │ parallel_eval  │
     │ (single)    │                   │ (multi-GPU)    │
     └─────┬──────┘                   └──────┬────────┘
           │                                  │
     ┌─────▼──────────────────────────────────▼─────────┐
     │                  BaseTask                         │
     │  envs/_base_task.py  (task lifecycle)             │
     │  - scene setup, actor management, step loop       │
     │  - tactile observation pipeline                   │
     │  - action execution (arm + gripper + settle)      │
     └──────┬──────────────────────────────┬────────────┘
            │                              │
    ┌───────▼────────┐           ┌────────▼──────────┐
    │   Robot (ABC)   │           │  TactileManager    │
    │  envs/robot/    │           │  sensors/tactile/  │
    │  - FrankaGelSight│          │  - GelSight Mini   │
    │  - FrankaZXHand  │          │  - GF225           │
    │  - Planner (ABC) │          │  - ZX Hand FemSensor│
    └───────┬─────────┘           └────────┬──────────┘
            │                              │
    ┌───────▼─────────┐           ┌───────▼──────────┐
    │  cuRobo Planner  │           │  tacex_uipc       │
    │  (CUDA motion)   │           │  (C++ FEM via     │
    │  curobo/         │           │   libuipc)        │
    └─────────────────┘           └──────────────────┘
```

## Episode Flow

```
1. CLI/script parses args, loads configs
2. AppLauncher starts Omniverse Kit App
3. BaseTask.__init__ → scene setup, robot creation, actor spawning
4. Policy.__init__ → model loading, weight warmup
5. For each seed:
   a. task.reset(seed, instruction)   → reset sim, randomize textures, set start pose
   b. policy.reset()                  → clear action chunk buffer
   c. Loop step_lim times:
      - task._get_observations()      → cameras + tactile + robot state
      - policy.eval(task, obs)        → policy predicts action (may use action chunking)
      - task.take_action(action)      → robot executes action via plan_arm + set_arm
      - if success or early_stop → break
   d. task.clean_cache()              → save video, depth images, results
6. Results saved to eval_result/<policy>/<task>/<config>/<timestamp>/
```

## Policy Interface

All policies implement `BasePolicy` (`policy/_base_policy.py`):

```python
class BasePolicy:
    def __init__(self, config: dict): ...
    def reset(self): ...                    # per-episode state reset
    def eval(self, task, observation): ...  # predict + execute actions
    def close(self): ...                    # cleanup
```

Policies are loaded dynamically:
```python
policy_module = importlib.import_module(f"policy.{policy_name}")
policy = policy_module.Policy(deploy_config)
```

Key distinction:
- **ACT / ViTAL / Ablation**: Run inside the Isaac Sim process (use sim state access)
- **OpenPI**: Pure PyTorch inference, uses Isaac Sim only for environment stepping

## Robot Hierarchy

```
Robot (ABC)                          # envs/robot/base.py
├── FrankaGelSight(Robot)            # envs/robot/franka_gelsight.py
│   Registered as: "franka_gsmini", "franka_gf225"
│   - Prismatic gripper (0–0.039 m), PD position control
│   - EE pose from Isaac Sim body state (fast, no FK)
│   - arm teleport via set_dof_positions
│
└── FrankaZXHand(Robot)              # envs/robot/franka_zxhand.py
    Registered as: "franka_zxhand"
    - Angular gripper (0–0.99 rad), velocity bang-bang
    - EE pose from cuRobo FK (no panda_hand body in ZX USD)
    - arm teleport via write_joint_state_to_sim (arm-only)
    - Finger-gel calibration, post-settle steps
```

Single dispatch point:
```python
from envs.robot import create_robot
robot = create_robot("franka_zxhand", cfg, task)
```

Legacy path (`RobotManager` in `envs/robot/robot.py`) is still functional as backward-compat shim.

## Config System

```
task_config/default.yml          → sensor type, observation keys, decimation, seeds
policy/<Name>/deploy.yml         → policy_name, checkpoint paths, model params
instructions/<task>.json         → seen/unseen instruction strings
policy/task_settings.json        → per-task camera type (head/all) + downsample
```

Config loader supports:
- `extends: _base.yml` — deep-merge inheritance
- `${oc.env:UNIVTAC_CKPTS_DIR,/default/path}` — env var interpolation
- Auto `.yml` suffix if omitted

## Docker Image Layers

```
nvidia/cuda:12.8.0-runtime-ubuntu22.04  (~2.5 GB)
  └── base        — EGL, Vulkan, uv, system deps     (~3 GB)
       └── isaac-sim  — Isaac Sim 4.5 + PyTorch cu128  (~20 GB)
            └── isaac-lab  — Isaac Lab 2.1.1            (~1 GB)
                 └── tacex  — vcpkg + C++/CUDA build   (~2 GB)
                      └── univtac  — project + policies  (~2 GB)
```

- **Dev mode** (`docker-compose.yml`): Source volume-mounted → edit code without rebuild
- **CI mode** (`docker-compose.ci.yml`): Pure image → reproducible results
- Host uv cache (~120 GB) is shared via `--build-context` to avoid re-downloading Isaac Sim

## Error Handling

- **Episode-level isolation**: One failing seed never aborts the entire eval
- **Error recording**: Each failure captures `failure_reason` (exception type + truncated message)
- **Statistics**: Success rate + error count + timeout count + early stop count
- **Parallel eval**: Worker failures are isolated; result queue collects all outcomes

## Key Design Decisions

1. **No conda** — `uv` for everything (venv, pip, dependency resolution)
2. **Dynamic policy loading** via `importlib` — no registration step needed, drop a folder in `policy/` and it's discoverable
3. **Robot ABC owns all embodiment differences** — tasks never branch on `robot_type`
4. **Config extends pattern** — base configs shared, child configs override specific fields
5. **Docker as first-class path** — same CLI command works natively and in containers
