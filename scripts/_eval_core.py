"""Shared config loading for evaluation scripts.

Extracted from ``eval_policy.py`` and ``parallel_eval_policy.py`` to
eliminate the duplicated ``get_config()`` function and config setup.
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Literal


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TASK_CONFIG_ROOT = _PROJECT_ROOT / "task_config"
_INSTRUCTIONS_ROOT = _PROJECT_ROOT / "instructions"
_POLICY_ROOT = _PROJECT_ROOT / "policy"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def get_config(
    file: str,
    default_root: Path,
    type: Literal["yaml", "json"],
) -> tuple[dict | list, Path]:
    """Load a YAML or JSON config file, auto-appending the extension if needed.

    Returns ``(parsed_config, resolved_path)``.
    """
    if type == "yaml":
        if file.endswith((".yml", ".yaml")):
            resolved = Path(file)
        else:
            resolved = default_root / f"{file}.yml"
        with open(resolved, "r") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader), resolved
    else:
        if file.endswith(".json"):
            resolved = Path(file)
        else:
            resolved = default_root / f"{file}.json"
        with open(resolved, "r") as f:
            return json.load(f), resolved


# ---------------------------------------------------------------------------
# Task config setup (shared between single and parallel eval)
# ---------------------------------------------------------------------------

def load_eval_configs(
    task_name: str,
    task_config_arg: str,
    deploy_config_arg: str,
) -> dict:
    """Load and cross-reference task config, deploy config, and instructions.

    Returns a dict with all parsed configs ready for environment construction.
    """
    task_config, task_config_file = get_config(
        task_config_arg, default_root=_TASK_CONFIG_ROOT, type="yaml"
    )
    deploy_config, deploy_config_file = get_config(
        deploy_config_arg, default_root=_POLICY_ROOT, type="yaml"
    )

    policy_name = deploy_config["policy_name"]
    deploy_config["task_name"] = task_name
    deploy_config["task_config"] = task_config_file.stem

    # Instructions
    instruction_file = deploy_config.get("instruction_file", task_name)
    try:
        instructions, _ = get_config(
            instruction_file, default_root=_INSTRUCTIONS_ROOT, type="json"
        )
        if not isinstance(instructions, dict) or "seen" not in instructions:
            instructions = {"seen": ["Empty"], "unseen": ["Empty"]}
    except Exception:
        instructions = {"seen": ["Empty"], "unseen": ["Empty"]}

    return {
        "task_name": task_name,
        "policy_name": policy_name,
        "task_config": task_config,
        "task_config_file": task_config_file,
        "deploy_config": deploy_config,
        "deploy_config_file": deploy_config_file,
        "instructions": instructions,
    }


def build_env_config(task_module, task_config: dict, deploy_config: dict) -> tuple:
    """Build the environment config from a loaded task module."""
    env_cfg = task_module.TaskCfg()
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.tactile_sensor_type = task_config.get("sensor_type", "gsmini")
    env_cfg.scene.num_envs = 1
    seed = deploy_config.get("seed", 0)
    return env_cfg, seed
