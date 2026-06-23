<h1 align="center">UniVTAC</h1>

> **UniVTAC**: A Unified Simulation Platform for Visuo-Tactile Manipulation — Data Generation, Learning, and Benchmarking<br>
> [arXiv](https://arxiv.org/abs/2602.10093) | [PDF](https://arxiv.org/pdf/2602.10093) | [Website](https://univtac.github.io/) | [HuggingFace](https://huggingface.co/datasets/byml/UniVTAC) | [Modelscope](https://modelscope.cn/datasets/byml2024/UniVTAC)

UniVTAC is a **tactile-aware simulation benchmark** built on NVIDIA Isaac Lab and TacEx (UIPC-based FEM tactile simulation). It provides a unified framework for collecting expert demonstrations, training visuotactile policies, and evaluating them across contact-rich manipulation tasks — with high-fidelity tactile feedback from GelSight Mini, GF225, or Xense ZX Hand sensors.

---

## Quick Start (Docker)

```bash
# 1. Clone
git clone --recurse-submodules https://github.com/byml-c/UniVTAC.git && cd UniVTAC

# 2. Pull pre-built image (or build locally: bash docker/build.sh)
docker pull ghcr.io/univtac/univtac:latest

# 3. Accept Isaac Sim EULA & run an eval
export OMNI_KIT_ACCEPT_EULA=YES
docker compose run --rm univtac \
    univtac eval grasp_classify default NullPolicy/deploy --total-num 1
```

> **Dev mode:** `docker-compose.yml` volume-mounts your source tree. Edit Python code and re-run — no rebuild needed.

<details>
<summary>Native install (for active development)</summary>

```bash
# 1. System deps
sudo apt install -y cmake build-essential gcc-11 g++-11 pkg-config git-lfs

# 2. Python environment
uv venv --python 3.10 --seed && source .venv/bin/activate
uv sync

# 3. Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab ../IsaacLab
cd ../IsaacLab && git checkout v2.1.1 && ./isaaclab.sh -i && cd -

# 4. TacEx + C++/CUDA build
cd third_party/TacEx && bash tacex.sh -i && cd -
bash scripts/build_tacex_uipc.sh

# 5. Install CLI
uv pip install -e .

# 6. Verify
export OMNI_KIT_ACCEPT_EULA=YES
univtac eval grasp_classify default NullPolicy/deploy --total-num 1
```

See [Installation Guide](./docs/Installation.md) for full details.
</details>

---

## CLI Reference

```bash
# Run evaluation (single process)
univtac eval <task> <task_config> <deploy_config> [--gpu 0] [--total-num 100]

# Run evaluation (multi-GPU parallel)
univtac eval <task> <task_config> <deploy_config> --workers 8

# Collect expert demonstrations
univtac collect <task> <task_config> <deploy_config>

# Discover what's available
univtac list tasks        # 9 tasks
univtac list policies     # 5 policies

# Validate config files
univtac validate config policy/OpenPI/deploy.yml
```

Legacy shell scripts (`bash eval_policy.sh ...`) continue to work alongside the CLI.

---

## Task Gallery

| Task | Module | Description |
|---|---|---|
| **Grasp & Classify** | `grasp_classify` | Grasp an object and classify it by tactile feedback |
| **Lift Can** | `lift_can` | Grasp and lift a cylindrical can |
| **Lift Bottle** | `lift_bottle` | Grasp and lift a bottle near a wall |
| **Insert HDMI** | `insert_HDMI` | Insert an HDMI connector into a port |
| **Insert Hole** | `insert_hole` | Precision peg-in-hole insertion |
| **Insert Tube** | `insert_tube` | Insert a tube into a fixture |
| **Pull Out Key** | `pull_out_key` | Extract a key from a lock |
| **Put Bottle in Shelf** | `put_bottle_in_shelf` | Place a bottle onto a shelf |
| **Collect** | `collect` | Record contact-rich tactile data for pretraining |

All tasks support three sensor configurations: `default` (GelSight Mini), `default_gf225`, `default_zxhand`.

---

## Policies

| Policy | Architecture | Tactile Input |
|--------|-------------|---------------|
| **ACT** | Action Chunking Transformer (DETR backbone) | Optional (vision-only / vision+tactile) |
| **Ablation** | ACT variants — modality ablations & data-scale sweeps | Configurable |
| **ViTAL** | ACT + CLIP-pretrained visuotactile encoders | Fused |
| **OpenPI** | pi0.5 (3.6B params, PaliGemma + action expert) | Optional (head / head+wrist / head+wrist+tactile) |
| **NullPolicy** | Zero-action baseline (pipeline smoke test) | None |

### Example eval commands

```bash
# ACT vision-only on GelSight Mini
univtac eval grasp_classify default ACT/deploy --total-num 100

# OpenPI head-only (no wrist, no tactile) on ZX Hand
univtac eval insert_HDMI demo_zxhand OpenPI/deploy_headonly --total-num 50

# Parallel eval across 4 GPUs
univtac eval lift_can default OpenPI/deploy --workers 4 --total-num 200
```

---

## Project Structure

```
UniVTAC/
├── policy/               # Policy plugins (dynamic import)
│   ├── _base_policy.py     # BasePolicy ABC
│   ├── ACT/                # Action Chunking Transformer
│   ├── Ablation/           # Ablation study variants
│   ├── ViTAL/              # Visuo-Tactile ACT (CLIP backbone)
│   ├── OpenPI/             # pi0.5 adapter (pure PyTorch, zero JAX)
│   └── NullPolicy/         # Zero-action baseline
├── envs/                  # Isaac Sim task environments
│   ├── _base_task.py       # BaseTask (1070 lines — being refactored)
│   └── robot/              # Robot abstraction layer (ABC hierarchy)
├── scripts/               # Entry points (eval, collect, replay, convert)
├── src/univtac/           # CLI + config + types (new package)
├── task_config/           # Per-sensor YAML configs
├── instructions/          # JSON task instructions
├── assets/                # Simulation assets (URDF, USD, textures)
├── third_party/           # Vendored deps (cuRobo, TacEx, xense-sim4.5)
├── docker/                # Dockerfiles + build/push scripts
├── docs/                  # Installation, Collection, Deploy, Architecture
└── pyproject.toml         # All Python deps (uv)
```

---

## Data

Pre-collected datasets (100 episodes/task) available on [HuggingFace](https://huggingface.co/datasets/byml/UniVTAC) and [Modelscope](https://modelscope.cn/datasets/byml2024/UniVTAC).

To collect your own data:
```bash
univtac collect grasp_classify default ACT/deploy
```

See [Data Collection Guide](./docs/Collection.md) for details.

---

## Development

```bash
make lint          # Auto-fix lint issues (ruff)
make format        # Auto-format code
make check         # CI check (lint + format)
make smoke         # Quick import check + task/policy listing
make docker-build  # Build all Docker images
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) and [Architecture](./docs/architecture.md) for details.

---

## Citation

If you find our work useful, please consider citing:

```
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile
         Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda
          and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie
          and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```

## License

MIT — see [LICENSE](./LICENSE).

## Contact

<div style="text-align: center;">
  <img src="https://box.nju.edu.cn/seafhttp/f/fc1021a908ff49309f22/?op=view" alt="Wechat Group" width="300"/>
</div>
