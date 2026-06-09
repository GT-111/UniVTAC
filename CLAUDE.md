# UniVTAC — Development Guide

## Environment Setup (uv, no conda)

```bash
# 1. Create virtual environment
uv venv --python 3.10 --seed
source .venv/bin/activate

# 2. Install Isaac Sim 4.5
export OMNI_KIT_ACCEPT_EULA=YES
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# 3. Install Isaac Lab 2.1.1
git clone https://github.com/isaac-sim/IsaacLab ../IsaacLab
cd ../IsaacLab && git checkout v2.1.1
./isaaclab.sh -i
cd -

# 4. Install TacEx (modified source in third_party)
cd third_party/TacEx
bash tacex.sh -e          # create uv venv
source .venv/bin/activate
bash tacex.sh -i          # install tacex + tacex_assets + tacex_tasks
cd -

# 5. OpenPI deps (transformers SigLIP patch is auto-applied at import time)
uv pip install transformers==4.53.2 safetensors sentencepiece
```

### Verify

```bash
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

# Core imports
python -c "import isaacsim; import isaaclab; print('OK')"

# Headless RL test
python ../IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-v0 --headless --num_envs 4 --max_iterations 2
```

## Project Structure

```
policy/           — Policy plugins (dynamic import via importlib)
  _base_policy.py   — BasePolicy abstract class
  ACT/              — Action Chunking Transformer
  ViTAL/            — Visuo-Tactile ACT (CLIP-pretrained backbones)
  Ablation/         — Ablation study variants (same arch as ACT)
  OpenPI/           — openpi pi05 adapter (self-contained, PyTorch-only)

envs/             — Isaac Sim task environments
scripts/           — Eval/training entry points
  eval_policy.py     — Single-process policy evaluation
  parallel_eval_policy.py — Multi-GPU evaluation

third_party/TacEx/ — Modified TacEx (tactile sensor pipeline)
```

## Policy Architecture

Each policy is a self-contained Python package under `policy/<name>/`:

```
policy/<Name>/
  __init__.py       # from .deploy_policy import *
  deploy_policy.py   # class Policy(BasePolicy): encode_obs(), eval(), reset(), close()
  deploy.yml         # policy_name: <Name>, plus model-specific config
  *.py               # model implementation (e.g., act_policy.py)
```

Eval scripts load policies dynamically:
```python
policy_module = importlib.import_module(f"policy.{policy_name}")
policy = policy_module.Policy(deploy_config)
```

## OpenPI (pi05) Deployment

Self-contained in `policy/OpenPI/` — pure PyTorch, zero JAX, no external runtime dependency.

```
policy/OpenPI/
  deploy_policy.py    # Policy(BasePolicy) — model loading + inference
  deploy.yml          # checkpoint_dir + tactile_mode (fallback)
  pi0_pytorch.py      # PI0Pytorch (3.6B params, adapted from openpi)
  gemma_pytorch.py    # PaliGemmaWithExpertModel
  preprocessing.py    # Image resize/pad
  config.py           # Pi0Config + GemmaConfig
  tokenizer.py        # PaligemmaTokenizer (sentencepiece)
  transformers_replace/  # SigLIP patch (auto-applied at import)
```

**Config auto-detection:** reads `config.json` (action_dim/action_horizon) and `metadata.pt` (tactile_mode) from the checkpoint. `deploy.yml` only needs `checkpoint_dir` for fine-tuned models.

**Eval (single GPU):**
```bash
python scripts/eval_policy.py grasp_classify default OpenPI/deploy
```

**Eval (multi-GPU):**
```bash
python scripts/parallel_eval_policy.py grasp_classify default OpenPI/deploy --workers 8
```

## OpenPI (pi05) Training

Training uses the openpi repo (Python 3.11, JAX-based). See `../openpi` for the training scripts.

```bash
# 1. Convert UniVTAC HDF5 → LeRobot format
# 2. Compute norm stats
# 3. Train (PyTorch DDP)
# 4. Deploy: set checkpoint_dir in policy/OpenPI/deploy.yml, then eval
```

Full workflow documented in the openpi repo at `examples/univtac/`.

## Key Config Files

| File | Purpose |
|------|---------|
| `policy/task_settings.json` | Per-task camera type (head/all) and downsample |
| `policy/<Name>/deploy.yml` | Policy name, model paths, seed |
| `policy/<Name>/train_config*.yml` | Training hyperparameters |

## Notes

- No conda anywhere — all `conda_env.yaml` files deleted, `tacex.sh` uses `--venv` flag
- `omni.kit` modules only exist when Kit App is running via `AppLauncher` (headless or GUI)
- `pip install` → use `uv pip install` everywhere for speed
- Isaac Sim requires `OMNI_KIT_ACCEPT_EULA=YES` in headless environments
- OpenPI SigLIP patch is shipped with the policy and auto-applied — no manual steps needed
- OpenPI checkpoint path, tokenizer path, and tactile mode are configurable via `deploy.yml` and env vars — see `policy/OpenPI/` for defaults
