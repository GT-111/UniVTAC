"""UniVTAC CLI — unified entry point for evaluation, collection, and config management.

Usage::

    univtac eval grasp_classify default ACT/deploy --gpu 0
    univtac collect grasp_classify default ACT/deploy
    univtac list tasks
    univtac list policies
    univtac validate config task_config/default.yml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Project root detection ────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _find_project_root() -> Path:
    """Find the UniVTAC project root (where pyproject.toml lives)."""
    # Walk up from this file
    for parent in [Path(__file__).resolve().parent.parent.parent.parent]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: current directory
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd
    return _PROJECT_ROOT


def _make_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="univtac",
        description="UniVTAC — Unified Simulation Platform for Visuo-Tactile Manipulation",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── eval ──────────────────────────────────────────────────────────
    eval_p = sub.add_parser("eval", help="Run policy evaluation")
    eval_p.add_argument("task", help="Task name (e.g. grasp_classify, insert_HDMI)")
    eval_p.add_argument("task_config", help="Task config name (e.g. default, demo)")
    eval_p.add_argument("deploy_config", help="Deploy config path (e.g. ACT/deploy, OpenPI/deploy)")
    eval_p.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    eval_p.add_argument("--total-num", type=int, default=100, help="Number of episodes")
    eval_p.add_argument("--start-seed", type=int, default=-1, help="Start seed (auto if -1)")
    eval_p.add_argument("--max-seed", type=int, default=-1, help="Max seed")
    eval_p.add_argument("--workers", type=int, default=1, help="Parallel workers (1 = single process)")
    eval_p.add_argument("--expert-check", action="store_true", help="Run expert check before eval")
    eval_p.add_argument("--print-only", action="store_true", help="Print only, no file output")

    # ── collect ───────────────────────────────────────────────────────
    col_p = sub.add_parser("collect", help="Run data collection")
    col_p.add_argument("task", help="Task name")
    col_p.add_argument("task_config", help="Task config name")
    col_p.add_argument("deploy_config", help="Deploy config path")
    col_p.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    col_p.add_argument("--total-num", type=int, default=100, help="Number of episodes")

    # ── list ──────────────────────────────────────────────────────────
    list_p = sub.add_parser("list", help="List available tasks or policies")
    list_p.add_argument("what", choices=["tasks", "policies"], help="What to list")

    # ── validate ──────────────────────────────────────────────────────
    val_p = sub.add_parser("validate", help="Validate configuration")
    val_p.add_argument("what", choices=["config"], help="What to validate")
    val_p.add_argument("path", help="Path to config file")

    return parser


# ── Command implementations ───────────────────────────────────────────

def cmd_eval(args: argparse.Namespace) -> int:
    """Run policy evaluation (shells out to existing scripts)."""
    project_root = _find_project_root()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    if args.workers > 1:
        # Parallel eval
        script = project_root / "scripts" / "parallel_eval_policy.py"
        cmd = [
            sys.executable, str(script),
            args.task, args.task_config, args.deploy_config,
            "--total_num", str(args.total_num),
            "--workers", str(args.workers),
        ]
    else:
        # Single-process eval
        script = project_root / "scripts" / "eval_policy.py"
        cmd = [
            sys.executable, str(script),
            args.task, args.task_config, args.deploy_config,
            "--total_num", str(args.total_num),
        ]

    if args.start_seed != -1:
        cmd += ["--start_seed", str(args.start_seed)]
    if args.max_seed != -1:
        cmd += ["--max_seed", str(args.max_seed)]
    if getattr(args, "expert_check", False):
        cmd.append("--expert_check")
    if getattr(args, "print_only", False):
        cmd.append("--print_only")

    print(f"[univtac] Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(project_root))


def cmd_collect(args: argparse.Namespace) -> int:
    """Run data collection (shells out to existing scripts)."""
    project_root = _find_project_root()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    script = project_root / "scripts" / "collect_data.py"
    cmd = [
        sys.executable, str(script),
        args.task, args.task_config, args.deploy_config,
        "--total_num", str(args.total_num),
    ]

    print(f"[univtac] Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(project_root))


def cmd_list(args: argparse.Namespace) -> int:
    """List available tasks or policies."""
    from univtac.registry import list_available_policies, list_available_tasks

    if args.what == "tasks":
        print("Available tasks:")
        for t in list_available_tasks():
            print(f"  - {t}")
    else:
        print("Available policies:")
        for p in list_available_policies():
            print(f"  - {p}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a configuration file."""
    from univtac.cli.config_loader import load_config

    try:
        cfg = load_config(args.path)
        print(f"✓ Config valid: {args.path}")
        print(f"  Keys: {list(cfg.keys())}")
        return 0
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Validation failed: {e}", file=sys.stderr)
        return 1


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "eval":
        return cmd_eval(args)
    elif args.command == "collect":
        return cmd_collect(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "validate":
        return cmd_validate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
