"""Action-space transforms ported from openpi — numpy-only, no JAX.

Matches the UniVTAC training pipeline:
  DataConfig.push(
    inputs=[DeltaActions(mask)],
    outputs=[AbsoluteActions(mask)],
  )
  where mask = make_bool_mask(7, -2) = [T×7, F×2]

Training input:  absolute_joints → DeltaActions → delta_joints + absolute_gripper
Training output: delta_joints + absolute_gripper → AbsoluteActions → absolute_joints + absolute_gripper
"""

import numpy as np


def make_bool_mask(*dims: int) -> np.ndarray:
    """Create a boolean mask. Positive dims = True, negative dims = False.

    Example:
        make_bool_mask(7, -2) → [T,T,T,T,T,T,T, F,F]
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * dim)
        else:
            result.extend([False] * (-dim))
    return np.array(result, dtype=bool)


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Zero-pad an array to the target dimension along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


class NormStats:
    """Quantile normalization stats."""
    def __init__(self, q01: np.ndarray, q99: np.ndarray):
        self.q01 = q01
        self.q99 = q99


def normalize_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Quantile normalize: (x - q01) / (q99 - q01 + eps) * 2 - 1 → [-1, 1]."""
    n = x.shape[-1]
    return (x - q01[..., :n]) / (q99[..., :n] - q01[..., :n] + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Quantile unnormalize: inverse of normalize_quantile."""
    dim = q01.shape[-1]
    if dim < x.shape[-1]:
        return np.concatenate([
            (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01,
            x[..., dim:],
        ], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


class DeltaActions:
    """Convert absolute actions to deltas for masked dimensions.

    masked[i]=True  → action[i] = action[i] - state[i]  (absolute → delta)
    masked[i]=False → action[i] unchanged               (keeps absolute)
    """

    def __init__(self, mask: np.ndarray | None = None):
        self.mask = mask

    def __call__(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return actions
        dims = self.mask.shape[-1]
        actions = actions.copy()
        delta = np.where(self.mask, state[..., :dims], 0.0)
        # Expand for batched actions (horizon, dims) but not for single actions (dims,)
        if actions.ndim > 1:
            delta = np.expand_dims(delta, axis=-2)
        actions[..., :dims] -= delta
        return actions


class AbsoluteActions:
    """Convert delta actions to absolute for masked dimensions.

    masked[i]=True  → action[i] = action[i] + state[i]  (delta → absolute)
    masked[i]=False → action[i] unchanged               (keeps absolute)
    """

    def __init__(self, mask: np.ndarray | None = None):
        self.mask = mask

    def __call__(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return actions
        dims = self.mask.shape[-1]
        actions = actions.copy()
        delta = np.where(self.mask, state[..., :dims], 0.0)
        if actions.ndim > 1:
            delta = np.expand_dims(delta, axis=-2)
        actions[..., :dims] += delta
        return actions
