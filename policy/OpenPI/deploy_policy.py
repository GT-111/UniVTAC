"""OpenPI (pi05) policy adapter — self-contained, PyTorch-only.

Compatible with both single-process (eval_policy.py) and multi-GPU
(parallel_eval_policy.py) evaluation. Each worker loads its own model copy.

Tactile mode is auto-detected from checkpoint metadata when available
(fine-tuned models save it in metadata.pt). Falls back to deploy.yml.
"""

import sys, json, dataclasses, shutil
from pathlib import Path

import numpy as np
import torch
import safetensors.torch

# ---- apply transformers SigLIP patch (one-time, idempotent) ----
_PKG = Path(__file__).parent
_PATCH_DIR = _PKG / "transformers_replace" / "models"
if _PATCH_DIR.exists():
    _SITE = Path(torch.__file__).parent.parent if hasattr(torch, "__file__") else None
    if _SITE is None:
        import site; _SITE = Path(site.getsitepackages()[0])
    _DST = _SITE / "transformers" / "models"
    if not (_DST / "siglip" / "check.py").exists():
        shutil.copytree(str(_PATCH_DIR), str(_DST), dirs_exist_ok=True)

sys.path.append(str(_PKG.parent))

from _base_policy import BasePolicy
from .config import Pi0Config
from .pi0_pytorch import PI0Pytorch
from .tokenizer import PaligemmaTokenizer
from . import transforms as _t


@dataclasses.dataclass
class _Obs:
    images: dict
    image_masks: dict
    state: torch.Tensor
    tokenized_prompt: torch.Tensor | None = None
    tokenized_prompt_mask: torch.Tensor | None = None
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None


class Policy(BasePolicy):
    def __init__(self, args):
        super().__init__(args)

        # ---- task camera settings ----
        self.task_name = args["task_name"]
        with open(Path(__file__).parent.parent / "task_settings.json") as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get("camera_type", "head")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- checkpoint (local path or HuggingFace repo) ----
        ckpt_src = args.get("checkpoint_dir",
            "/data/temp_storage/exps/openpi/openpi-assets/checkpoints/pi05_base_pytorch")
        if "/" in ckpt_src and not ckpt_src.startswith(("/", ".", "~")):
            # HuggingFace repo ID — download to cache
            from huggingface_hub import snapshot_download
            ckpt_dir = Path(snapshot_download(ckpt_src, repo_type="model"))
        else:
            ckpt_dir = Path(ckpt_src)

        # ---- auto-detect model config ----
        meta_path = ckpt_dir / "metadata.pt"
        config_json = ckpt_dir / "config.json"

        action_dim, action_horizon, self.tactile_mode = 32, 10, args.get("tactile_mode", "left_only")
        meta_loaded = False
        if meta_path.exists():
            try:
                meta = torch.load(meta_path, map_location="cpu", weights_only=False)
                cfg = meta.get("config", {})
                action_dim = cfg.get("model", {}).get("action_dim", action_dim)
                action_horizon = cfg.get("model", {}).get("action_horizon", action_horizon)
                self.tactile_mode = cfg.get("data", {}).get("tactile_mode") or self.tactile_mode
                meta_loaded = True
                print(f"[OpenPI] tactile_mode={self.tactile_mode} (from metadata.pt)")
            except Exception:
                print(f"[OpenPI] metadata.pt needs flax — using config.json fallback")
        if not meta_loaded:
            if config_json.exists():
                ckpt_cfg = json.load(open(config_json))
                action_dim = ckpt_cfg.get("action_dim", action_dim)
                action_horizon = ckpt_cfg.get("action_horizon", action_horizon)
            print(f"[OpenPI] tactile_mode={self.tactile_mode} (from deploy.yml)")
        if not meta_loaded and not config_json.exists():
            raise FileNotFoundError(f"No config found in {ckpt_dir} (need metadata.pt or config.json)")

        # ---- execution horizon (how many actions to execute per inference) ----
        exec_horizon = args.get("exec_horizon", 1)
        if exec_horizon <= 0:
            exec_horizon = action_horizon  # auto: use full chunk
        self.exec_horizon = min(exec_horizon, action_horizon)

        print(f"[OpenPI] task={self.task_name}, camera={self.camera_type}, "
              f"action_dim={action_dim}, action_horizon={action_horizon}, "
              f"exec_horizon={self.exec_horizon}")

        # ---- load norm stats (pi05 uses quantile normalization: q01/q99) ----
        assets = ckpt_dir / "assets"
        norm_dir = assets / "franka" if (assets / "franka").exists() else next(assets.iterdir())
        with open(norm_dir / "norm_stats.json") as f:
            ns = json.load(f)["norm_stats"]
        self.s_q01 = np.array(ns["state"]["q01"], np.float32)
        self.s_q99 = np.array(ns["state"]["q99"], np.float32)
        self.a_q01 = np.array(ns["actions"]["q01"], np.float32)
        self.a_q99 = np.array(ns["actions"]["q99"], np.float32)

        # ---- action-space transforms (matching UniVTAC training pipeline) ----
        # mask = make_bool_mask(7, -2) → joints 0-6: delta, gripper 7-8: absolute
        # (norm stats may be computed on raw absolute values, but DeltaActions
        #  converts joints to delta before normalization during training.)
        self._action_mask = _t.make_bool_mask(7, -2)
        self._to_absolute = _t.AbsoluteActions(self._action_mask)

        # ---- build model ----
        cfg = Pi0Config(pi05=True, action_dim=action_dim, action_horizon=action_horizon)
        self.model = PI0Pytorch(cfg)

        safetensors.torch.load_model(self.model, str(ckpt_dir / "model.safetensors"))
        self.model = self.model.to(self.device).eval()
        self.action_dim = action_dim
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[OpenPI] Model loaded: {n_params:,} params | device={self.device}")

        # ---- tokenizer ----
        tokenizer_path = args.get("tokenizer_path", None)
        self.tokenizer = PaligemmaTokenizer(max_len=200, model_path=tokenizer_path)

    # ------------------------------------------------------------------

    def encode_obs(self, observation):
        """UniVTAC runtime obs → intermediate numpy dict."""
        obs = observation

        def to_np(t):
            return t.cpu().numpy().astype(np.uint8)

        head_rgb  = to_np(obs["observation"]["head"]["rgb"])
        wrist_rgb = to_np(obs["observation"]["wrist"]["rgb"]) if self.camera_type == "all" \
                    else np.zeros_like(head_rgb)
        tac_left  = to_np(obs["tactile"]["left_tactile"]["rgb_marker"])
        tac_right = to_np(obs["tactile"]["right_tactile"]["rgb_marker"])
        joint = obs["embodiment"]["joint"].cpu().numpy().astype(np.float32).flatten()

        # Map 4 sources → 3 model slots, same as openpi's UniVTACInputs
        if   self.tactile_mode == "none":         img2, img3 = wrist_rgb, np.zeros_like(head_rgb)
        elif self.tactile_mode == "left_only":     img2, img3 = wrist_rgb, tac_left
        elif self.tactile_mode == "right_only":    img2, img3 = wrist_rgb, tac_right
        elif self.tactile_mode == "side_by_side":  img2, img3 = wrist_rgb, np.concatenate([tac_left, tac_right], axis=1)
        elif self.tactile_mode == "drop_wrist":    img2, img3 = tac_left, tac_right
        else:                                      img2, img3 = wrist_rgb, tac_left  # default

        return {"head": head_rgb, "img2": img2, "img3": img3,
                "joint": joint[:7], "gripper": joint[7:9]}

    def eval(self, task, observation):
        enc = self.encode_obs(observation)

        # --- input pipeline (matches training: UniVTACInputs → Normalize → Tokenize → Pad) ---
        state = np.concatenate([enc["joint"], enc["gripper"]]).astype(np.float32)
        state_n = _t.normalize_quantile(state, self.s_q01, self.s_q99)
        state_pad = _t.pad_to_dim(state_n, self.action_dim)

        prompt = getattr(task, "instruction", "perform the manipulation task")
        tokens, tmask = self.tokenizer.tokenize(prompt, state_n) if self.action_dim > 8 \
                        else self.tokenizer.tokenize(prompt)

        def prep(arr):
            t = torch.from_numpy(arr).float().to(self.device) / 127.5 - 1.0
            return t.permute(2, 0, 1).unsqueeze(0) if t.dim() == 3 else t.unsqueeze(0)

        obs = _Obs(
            images={"base_0_rgb": prep(enc["head"]), "left_wrist_0_rgb": prep(enc["img2"]),
                    "right_wrist_0_rgb": prep(enc["img3"])},
            image_masks={k: torch.tensor([True], device=self.device) for k in
                         ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]},
            state=torch.from_numpy(state_pad).unsqueeze(0).to(self.device),
            tokenized_prompt=torch.from_numpy(tokens).unsqueeze(0).to(self.device),
            tokenized_prompt_mask=torch.from_numpy(tmask).unsqueeze(0).to(self.device),
        )

        with torch.no_grad():
            actions = self.model.sample_actions(str(self.device), obs, num_steps=10)

        # --- output pipeline (matches training: Unnormalize → AbsoluteActions → execute) ---
        # All deltas in the chunk are relative to the SAME first state (see DeltaActions).
        state0 = observation["embodiment"]["joint"][:9].cpu().numpy()

        for i in range(self.exec_horizon):
            if task.eval_success:
                break

            act = actions[0, i].cpu().float().numpy()
            act_denorm = _t.unnormalize_quantile(act, self.a_q01, self.a_q99)
            # HACK: training had double-delta (data already delta + DeltaActions).
            # model ≈ state[t+1] - 2*state[t], so recover: act + 2*state0
            act_abs = self._to_absolute(2.0 * state0, act_denorm)

            target_qpos = np.concatenate([act_abs[:7], act_abs[7:8]]).astype(np.float32)

            exec_succ, eval_succ = task.take_action(
                torch.from_numpy(target_qpos).to(task.device).float(), action_type="qpos")

            if not exec_succ or eval_succ:
                break

    def reset(self):
        pass

    def close(self):
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()
