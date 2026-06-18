"""Task registry — discoverable task lookup.

Usage::

    from envs._registry import register_task, list_tasks

    @register_task("insert_hole")
    class Task(BaseTask): ...

    print(list_tasks())  # ['collect', 'grasp_classify', 'insert_HDMI', ...]
"""

from __future__ import annotations

_TASK_REGISTRY: dict[str, type] = {}


def register_task(name: str):
    """Decorator — each Task class announces itself under a string name.

    The decorated class still uses the name ``Task`` so that
    ``importlib.import_module(f"envs.{task_name}")`` continues to work.
    """

    def decorator(cls):
        _TASK_REGISTRY[name] = cls
        return cls

    return decorator


def list_tasks() -> list[str]:
    """Return all registered task names in sorted order."""
    return sorted(_TASK_REGISTRY.keys())


def get_task_class(name: str) -> type | None:
    """Look up a task class by name.  Returns None if not found."""
    return _TASK_REGISTRY.get(name)
