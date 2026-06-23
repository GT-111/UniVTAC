"""Import-string resolution for policies and tasks.

Modeled after the VLA Evaluation Harness ``registry.py`` using the
``lazyregistry`` library for lazy import resolution.

Usage::

    from univtac.registry import resolve_import_string

    PolicyCls = resolve_import_string("policy.ACT.deploy_policy:Policy")
    TaskCls  = resolve_import_string("envs.grasp_classify:Task")
"""

from __future__ import annotations

import importlib
from typing import Any


def resolve_import_string(import_path: str) -> Any:
    """Resolve a ``"module.path:ClassName"`` string to a Python object.

    Args:
        import_path: Import path string, e.g. ``"policy.ACT.deploy_policy:Policy"``.

    Returns:
        The resolved Python class or object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the name is not found in the module.
        ValueError: If the import_path format is invalid.
    """
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import string '{import_path}'. "
            f"Expected format: 'module.path:ClassName'"
        )

    module_path, attr_name = import_path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}") from e

    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{attr_name}'"
        )

    return getattr(module, attr_name)


def list_available_tasks() -> list[str]:
    """Discover available task environments from ``envs/``."""
    import os
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    envs_dir = project_root / "envs"
    tasks = []
    for f in sorted(envs_dir.glob("*.py")):
        name = f.stem
        if name.startswith("_"):
            continue
        # Try inspecting the file for a Task class without importing
        # (importing needs Isaac Sim runtime)
        try:
            source = f.read_text()
            if "class Task" in source or "class " in source:
                tasks.append(name)
        except Exception:
            continue
    return tasks


def list_available_policies() -> list[str]:
    """Discover available policy plugins from ``policy/``."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    policy_dir = project_root / "policy"
    policies = []
    for d in sorted(policy_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name.startswith("."):
            continue
        deploy_yml = d / "deploy.yml"
        if deploy_yml.exists():
            policies.append(d.name)
    return policies
