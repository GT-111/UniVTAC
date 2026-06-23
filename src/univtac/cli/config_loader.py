"""YAML config loader with ``extends`` inheritance and env-var interpolation.

Modeled after the VLA Evaluation Harness config system.

Usage::

    from univtac.cli.config_loader import load_config

    cfg = load_config("policy/OpenPI/deploy.yml")
    # If deploy.yml has ``extends: _base.yml``, the base config is deep-merged first.
    # ``${oc.env:VAR,default}`` placeholders are resolved after merging.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_RE = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ``${oc.env:VAR,default}`` placeholders in strings."""
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            var = m.group(1)
            default = m.group(2)
            return os.environ.get(var, default if default is not None else m.group(0))
        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts. ``override`` values take precedence."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_yaml(path: str | Path) -> dict:
    """Load a single YAML file. Returns ``{}`` if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(data).__name__}")
    return data


def load_config(config_path: str | Path, resolve_env: bool = True) -> dict:
    """Load a YAML config with optional ``extends`` inheritance.

    Resolution rules (applied in order):

    1. If the top-level config has an ``extends`` key, load the referenced file
       first, then deep-merge the current config on top.
    2. ``extends`` can be:
       - A string (single base file, resolved relative to the current file)
       - A list of strings (multiple bases, merged left-to-right)
    3. After merging, ``${oc.env:VAR,default}`` placeholders are resolved
       against environment variables.

    Args:
        config_path: Path to the YAML config file.
        resolve_env: If True (default), resolve ``${oc.env:...}`` placeholders.

    Returns:
        Parsed and merged config dict.
    """
    config_path = Path(config_path)
    base_dir = config_path.parent

    # Resolve file path (auto-append .yml if no extension)
    if not config_path.suffix:
        for ext in (".yml", ".yaml"):
            candidate = config_path.with_suffix(ext)
            if candidate.exists():
                config_path = candidate
                break

    data = load_yaml(config_path)

    # --- extends inheritance ---
    extends = data.pop("extends", None)
    if extends is not None:
        if isinstance(extends, str):
            extends = [extends]
        merged: dict = {}
        for base_ref in extends:
            base_path = (base_dir / base_ref).resolve()
            if not base_path.suffix:
                for ext in (".yml", ".yaml"):
                    candidate = base_path.with_suffix(ext)
                    if candidate.exists():
                        base_path = candidate
                        break
            base_data = load_config(base_path, resolve_env=False)
            merged = _deep_merge(merged, base_data)
        merged = _deep_merge(merged, data)
        data = merged

    # --- env var interpolation ---
    if resolve_env:
        data = _resolve_env_vars(data)

    return data


def load_task_config(name: str) -> dict:
    """Load a task config from ``task_config/``."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    return load_config(root / "task_config" / name)


def load_deploy_config(name: str) -> dict:
    """Load a deploy config from ``policy/``."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    return load_config(root / "policy" / name)
