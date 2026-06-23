"""Typed config dataclasses for evaluation and deployment.

Modeled after the VLA Evaluation Harness ``config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from univtac.types import SensorType


@dataclass
class DockerConfig:
    """Docker execution settings (mirrors VLA harness pattern)."""

    image: str | None = None
    """Docker image to run evaluation inside. If None, run natively."""

    volumes: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    gpus: int | None = None
    cpus: int | None = None


@dataclass
class TaskConfig:
    """Parsed task configuration."""

    sensor_type: SensorType = "gsmini"
    observations: dict[str, Any] = field(default_factory=dict)
    decimation: int = 1
    save_frequency: int = 0
    video_frequency: int = 0
    render_frequency: int = 0
    random_texture: bool = False
    episode_num: int = 100
    save_dir: str = "eval_result"
    use_seed: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> TaskConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


@dataclass
class DeployConfig:
    """Parsed policy deployment configuration."""

    policy_name: str = ""
    seed: int = 0
    instruction_type: Literal["seen", "unseen"] = "seen"
    instruction_file: str | None = None
    tactile_mode: str = "left_only"
    exec_horizon: int = 4
    checkpoint_dir: str | None = None
    tokenizer_path: str | None = None

    # Extra kwargs that vary per policy
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> DeployConfig:
        known = {f.name for f in cls.__dataclass_fields__.values__()}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        cfg = cls(**kwargs)
        cfg.extra = extra
        return cfg


@dataclass
class EvalConfig:
    """Complete evaluation configuration (benchmark + server)."""

    task_name: str = ""
    task_config: TaskConfig = field(default_factory=TaskConfig)
    deploy_config: DeployConfig = field(default_factory=DeployConfig)
    instructions: dict[str, list[str]] = field(default_factory=dict)
    docker: DockerConfig = field(default_factory=DockerConfig)
    output_dir: str = "eval_result"
    total_num: int = 100
    start_seed: int = 0
    max_seed: int = -1
    gpu: int = 0
    workers: int = 1

    @classmethod
    def from_configs(
        cls,
        task_name: str,
        task_cfg_dict: dict,
        deploy_cfg_dict: dict,
        instructions: dict | None = None,
    ) -> EvalConfig:
        return cls(
            task_name=task_name,
            task_config=TaskConfig.from_dict(task_cfg_dict),
            deploy_config=DeployConfig.from_dict(deploy_cfg_dict),
            instructions=instructions or {},
        )
