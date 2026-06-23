# UniVTAC — Development Guide

## Quick Start

```bash
# Docker (recommended for eval/reproduction)
export OMNI_KIT_ACCEPT_EULA=YES
docker compose run --rm univtac univtac eval grasp_classify default NullPolicy/deploy --total-num 1

# Native (for active development)
source .venv/bin/activate
univtac eval grasp_classify default NullPolicy/deploy --total-num 1
```

Docker dev mode volume-mounts source — edit Python code without rebuild.

## Environment Setup (Native)

```bash
# 1. System deps
sudo apt install -y cmake build-essential gcc-11 g++-11 pkg-config git-lfs

# 2. Python environment
uv venv --python 3.10 --seed
source .venv/bin/activate
uv sync

# 3. Isaac Sim 4.5
export OMNI_KIT_ACCEPT_EULA=YES
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# 4. Isaac Lab 2.1.1
git clone https://github.com/isaac-sim/IsaacLab ../IsaacLab
cd ../IsaacLab && git checkout v2.1.1
./isaaclab.sh -i
cd -

# 5. TacEx (modified source in third_party)
cd third_party/TacEx && bash tacex.sh -i && cd -
bash scripts/build_tacex_uipc.sh

# 6. UniVTAC CLI
uv pip install -e .

# Verify
export OMNI_KIT_ACCEPT_EULA=YES
python -c "import isaacsim; import isaaclab; print('OK')"
univtac list tasks
```

## Docker Images

```bash
bash docker/build.sh              # build all 5 layers
bash docker/build.sh isaac-sim    # build single layer
bash docker/push.sh --tag 0.2.0   # push to GHCR
```

Layers: `base` → `isaac-sim` → `isaac-lab` → `tacex` → `univtac`

Build script shares host uv cache (~120 GB) via `--build-context` — avoids re-downloading Isaac Sim.

## Project Structure

```
policy/               — Policy plugins (dynamic import via importlib)
  _base_policy.py       — BasePolicy abstract class
  ACT/                  — Action Chunking Transformer
  Ablation/             — Ablation study variants
  ViTAL/                — Visuo-Tactile ACT (CLIP-pretrained backbones)
  OpenPI/               — pi0.5 adapter (self-contained, PyTorch-only)

envs/                 — Isaac Sim task environments
  _base_task.py         — BaseTask (god class, being refactored)
  robot/                — Robot abstraction (ABC hierarchy)
    base.py               — Robot ABC + RobotConfig
    franka_gelsight.py    — FrankaGelSight (gsmini + gf225)
    franka_zxhand.py      — FrankaZXHand
    registry.py           — @register_robot + create_robot()
    planner.py            — Planner ABC + LinearPlanner
    robot.py              — RobotManager (legacy, backward-compat)

src/univtac/          — CLI + config + types (NEW package)
  cli/main.py            — univtac eval|collect|list|validate
  cli/config_loader.py   — YAML extends + env var interpolation
  config.py              — EvalConfig, DeployConfig, TaskConfig dataclasses
  registry.py            — task/policy auto-discovery
  specs.py               — DimSpec action/obs validation

scripts/              — Eval/training entry points
  eval_policy.py         — Single-process evaluation
  parallel_eval_policy.py — Multi-GPU evaluation
  _eval_core.py          — Shared config loading

docker/               — Dockerfiles + build/push scripts
task_config/          — Per-sensor YAML configs (default, gf225, zxhand)
instructions/         — JSON task instruction files
assets/               — Simulation assets (URDF, USD, textures)
third_party/          — Vendored: cuRobo (submodule), TacEx (bundled), xense-sim4.5
```

## CLI Reference

```bash
univtac eval <task> <task_config> <deploy_config> [--gpu 0] [--total-num 100] [--workers 1]
univtac collect <task> <task_config> <deploy_config>
univtac list tasks          # 9 tasks
univtac list policies       # 5 policies
univtac validate config <path>
```

Legacy shell scripts (`eval_policy.sh`, `parallel_eval.sh`) still work.

## Policy Architecture

Each policy is self-contained under `policy/<Name>/`:

```
policy/<Name>/
  __init__.py         # from .deploy_policy import *
  deploy_policy.py    # class Policy(BasePolicy): reset(), eval(), close()
  deploy.yml          # policy_name + model-specific config
  *.py                # model implementation
```

Policies are loaded dynamically:
```python
policy_module = importlib.import_module(f"policy.{policy_name}")
policy = policy_module.Policy(deploy_config)
```

## Robot Architecture

Robot hierarchy replaces ~30 `if robot_type == ...` branches with polymorphic dispatch:

```python
from envs.robot import create_robot
robot = create_robot("franka_zxhand", cfg, task)
robot.setup()
# All methods are uniform — no robot_type branches in task code
robot.set_arm(pos, vel)
robot.set_gripper(pos, vel)
robot.get_ee_pose()
robot.plan_gripper(0.5)
```

Adding a new robot: subclass `Robot(ABC)`, decorate with `@register_robot("name")`, add sensor→robot mapping in `robot_cfg.py`. Zero changes to tasks.

## Robot Config (Single Dispatch)

```python
from envs.robot.robot_cfg import create_robot_cfg, get_robot_name

cfg   = create_robot_cfg("gsmini", data_type=["rgb", "depth"])
name  = get_robot_name("zxhand")   # → "franka_zxhand"
```

## Config System

- `task_config/*.yml` — sensor type, observation keys, decimation, seeds
- `policy/<Name>/deploy.yml` — policy_name, checkpoint paths, model params
- `instructions/<task>.json` — seen/unseen instructions
- `policy/task_settings.json` — per-task camera type + downsample

Config loader supports:
- `extends: _base.yml` — deep-merge inheritance
- `${oc.env:UNIVTAC_CKPTS_DIR,/default}` — env var interpolation

## OpenPI (pi05) Deployment

Self-contained, pure PyTorch, zero JAX.

```
policy/OpenPI/
  deploy_policy.py     — Policy(BasePolicy), model loading + inference
  deploy.yml           — checkpoint_dir + tactile_mode
  pi0_pytorch.py       — PI0Pytorch (3.6B params)
  gemma_pytorch.py     — PaliGemmaWithExpertModel
  transformers_replace/ — SigLIP patch (auto-applied)
```

Config auto-detection: reads `config.json` + `metadata.pt` from checkpoint.

```bash
univtac eval grasp_classify default OpenPI/deploy
```

## OpenPI Training

Uses openpi repo (Python 3.11, JAX). See `../openpi` for training scripts.

## Key Config Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | All Python deps + CLI entry point + editable sources |
| `task_config/*.yml` | Per-sensor task configuration |
| `policy/<Name>/deploy.yml` | Policy deployment (checkpoints, model params) |
| `policy/task_settings.json` | Per-task camera type and downsample |
| `docker-compose.yml` | Dev mode (source mount) |
| `docker-compose.ci.yml` | CI / reproduction mode |

## Notes

- No conda — `uv` for everything
- `omni.kit` modules only exist when Kit App is running
- `pip install` → `uv pip install` everywhere
- `OMNI_KIT_ACCEPT_EULA=YES` required for headless Isaac Sim
- OpenPI SigLIP patch auto-applied at import time
- Use `${oc.env:UNIVTAC_CKPTS_DIR}` instead of hardcoded absolute paths
- Docker dev mode: edit code → `docker compose run` directly, no rebuild
