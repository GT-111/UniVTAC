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
    _SITE = Path(torch.__file__).parent.parent / "site-packages" if hasattr(torch, "__file__") else None
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

        # ---- checkpoint ----
        ckpt_dir = Path(args.get("checkpoint_dir",
            "/data/temp_storage/exps/openpi/openpi-assets/checkpoints/pi05_base_pytorch"))

        # ---- auto-detect config from checkpoint ----
        with open(ckpt_dir / "config.json") as f:
            ckpt_cfg = json.load(f)
        action_dim = ckpt_cfg.get("action_dim", 32)
        action_horizon = ckpt_cfg.get("action_horizon", 10)

        # Auto-detect tactile_mode from metadata.pt (fine-tuned checkpoints only)
        meta_path = ckpt_dir / "metadata.pt"
        if meta_path.exists():
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            train_cfg = meta.get("config", {})
            data_cfg = train_cfg.get("data", {})
            self.tactile_mode = data_cfg.get("tactile_mode") or args.get("tactile_mode", "left_only")
            print(f"[OpenPI] tactile_mode={self.tactile_mode} (from checkpoint)")
        else:
            self.tactile_mode = args.get("tactile_mode", "left_only")
            print(f"[OpenPI] tactile_mode={self.tactile_mode} (from deploy.yml)")

        print(f"[OpenPI] task={self.task_name}, camera={self.camera_type}, "
              f"action_dim={action_dim}, action_horizon={action_horizon}")

        # ---- load norm stats ----
        # Try franka first (UniVTAC robot), fall back to first available
        assets = ckpt_dir / "assets"
        norm_dir = assets / "franka" if (assets / "franka").exists() else next(assets.iterdir())
        with open(norm_dir / "norm_stats.json") as f:
            ns = json.load(f)["norm_stats"]
        self.s_mean = np.array(ns["state"]["mean"], np.float32)
        self.s_std  = np.array(ns["state"]["std"],  np.float32)
        self.a_mean = np.array(ns["actions"]["mean"], np.float32)
        self.a_std  = np.array(ns["actions"]["std"],  np.float32)

        # ---- build model ----
        cfg = Pi0Config(pi05=True, action_dim=action_dim, action_horizon=action_horizon)
        self.model = PI0Pytorch(cfg)
        safetensors.torch.load_model(self.model, str(ckpt_dir / "model.safetensors"))
        self.model = self.model.to(self.device).eval()
        self.action_dim = action_dim
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[OpenPI] Model loaded: {n_params:,} params | device={self.device}")

        # ---- tokenizer ----
        self.tokenizer = PaligemmaTokenizer(max_len=200)

    # ------------------------------------------------------------------

    def encode_obs(self, observation):
        """UniVTAC runtime obs → intermediate numpy dict."""
        obs = observation

        def to_np(t):
            return t.cpu().numpy().astype(np.uint8)

        head_rgb  = to_np(obs["observation"]["head"]["rgb"])
        wrist_rgb = to_np(obs["observation"]["wrist"]["rgb"]) if self.camera_type == "all" \
                    else np.zeros_like(head_rgb)
        tac_left  = to_np(obs["tactile"]["left_gsmini"]["rgb_marker"])
        tac_right = to_np(obs["tactile"]["right_gsmini"]["rgb_marker"])
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

        # Normalize + pad state
        state = np.concatenate([enc["joint"], enc["gripper"]]).astype(np.float32)
        state_n = (state - self.s_mean) / (self.s_std + 1e-8)
        state_pad = np.pad(state_n, (0, max(0, self.action_dim - len(state_n))))

        # Tokenize
        prompt = getattr(task, "instruction", "perform the manipulation task")
        tokens, tmask = self.tokenizer.tokenize(state_n, prompt) if self.action_dim > 8 \
                        else self.tokenizer.tokenize(prompt)

        # Image: uint8 [0,255] → float32 [-1,1], HWC → 1CHW
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

        # Unnormalize first action in chunk → 8D qpos (7 arm + 1 gripper)
        act = actions[0, 0].cpu().float().numpy()
        act_denorm = act[:9] * (self.a_std + 1e-8) + self.a_mean
        qpos = np.concatenate([act_denorm[:7], act_denorm[7:8]])

        task.take_action(torch.from_numpy(qpos).to(task.device).float(), action_type="qpos")

    def reset(self):
        pass

    def close(self):
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()
