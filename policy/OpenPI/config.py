"""Minimal pi05 config — extracted from openpi, no JAX dependency."""

import dataclasses
import enum
from typing import Any


# ---------------------------------------------------------------------------
# Gemma config (from openpi/models/gemma.py)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 8.0


@dataclasses.dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, LoRAConfig] | None = None


def get_gemma_config(variant: str) -> GemmaConfig:
    if variant == "dummy":
        return GemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    if variant == "gemma_300m":
        return GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return GemmaConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b_lora":
        return GemmaConfig(
            width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256,
            lora_configs={"attn": LoRAConfig(rank=16, alpha=16.0), "ffn": LoRAConfig(rank=16, alpha=16.0)})
    if variant == "gemma_300m_lora":
        return GemmaConfig(
            width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256,
            lora_configs={"attn": LoRAConfig(rank=32, alpha=32.0), "ffn": LoRAConfig(rank=32, alpha=32.0)})
    raise ValueError(f"Unknown gemma variant: {variant}")


# ---------------------------------------------------------------------------
# Model type (from openpi/models/model.py)
# ---------------------------------------------------------------------------

class ModelType(enum.Enum):
    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION = (224, 224)


# ---------------------------------------------------------------------------
# Pi0Config (from openpi/models/pi0_config.py — inference only)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Pi0Config:
    pi05: bool = True
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"
    pytorch_compile_mode: str | None = None

    @property
    def model_type(self) -> ModelType:
        return ModelType.PI05 if self.pi05 else ModelType.PI0

    @property
    def discrete_state_input(self) -> bool:
        return False  # pi05 default for LIBERO/UniVTAC
