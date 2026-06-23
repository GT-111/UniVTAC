# Contributing to UniVTAC

## Development Setup

```bash
# Native install (see docs/Installation.md for full guide)
uv venv --python 3.10 --seed && source .venv/bin/activate
uv sync
uv pip install -e .          # CLI + src/univtac package

# Verify
univtac list tasks
univtac list policies
```

## Code Style

- **Formatting**: `ruff format` (line width 100)
- **Linting**: `ruff check`
- **Type hints**: Required for public APIs in `src/univtac/`, optional in `envs/` and `policy/`

```bash
make lint          # Auto-fix
make check         # CI mode (no auto-fix, fails on issues)
```

## Adding a Policy

1. Create `policy/YourPolicy/` with:
   - `__init__.py` — `from .deploy_policy import *`
   - `deploy_policy.py` — subclass `BasePolicy`, implement `encode_obs()`, `eval()`, `reset()`, `close()`
   - `deploy.yml` — `policy_name: YourPolicy` plus model-specific config

2. See `policy/ACT/` and `policy/OpenPI/` for examples.

3. The CLI discovers it automatically: `univtac list policies`

## Adding a Task

1. Create `envs/your_task.py` with:
   - `TaskCfg` class (extends `BaseTaskCfg`)
   - `Task` class (extends `BaseTask`)

2. Add instruction file: `instructions/your_task.json`

3. Add task config: `task_config/default.yml` (or `default_gf225.yml`, `default_zxhand.yml`)

4. The CLI discovers it automatically: `univtac list tasks`

## Config Conventions

- All configs are YAML. Use `.yml` extension.
- Task configs in `task_config/`, deploy configs in `policy/<Name>/`.
- Supports `extends` inheritance:
  ```yaml
  extends: _base.yml
  checkpoint_dir: ${oc.env:UNIVTAC_CKPTS_DIR}/my_checkpoint
  ```
- Use `${oc.env:VAR,default}` for paths that vary across machines — never hardcode absolute paths.

## Robot Abstraction

Adding a new robot embodiment:

1. Subclass `Robot` (in `envs/robot/base.py`)
2. Implement: `setup()`, `get_ee_pose()`, `set_arm()`, `set_gripper()`, `plan_arm()`, `plan_gripper()`, `is_overpressed()`
3. Decorate with `@register_robot("your_robot_name")`
4. Add sensor→robot mapping in `envs/robot/robot_cfg.py`
5. Zero changes to tasks or `BaseTask` needed

## Testing

```bash
# Quick smoke test (no GPU needed for CLI)
univtac list tasks
univtac list policies
univtac validate config policy/NullPolicy/deploy.yml

# Full pipeline test (needs Isaac Sim + GPU)
univtac eval grasp_classify default NullPolicy/deploy --total-num 1 --print-only
```

## Docker Images

```bash
# Build all layers (needs EULA + GPU for verification steps)
bash docker/build.sh

# Build a single layer
bash docker/build.sh isaac-sim

# Push to GHCR
bash docker/push.sh --tag 0.2.0
```

Image layers: `base` → `isaac-sim` → `isaac-lab` → `tacex` → `univtac`

## PR Workflow

1. Branch from `main`
2. Make changes, keep PRs focused
3. Run `make check` before pushing
4. If changing `envs/robot/` or `envs/_base_task.py`, smoke-test with `univtac eval ... NullPolicy --total-num 1`
5. PR description should include what was tested
