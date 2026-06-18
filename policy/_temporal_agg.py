"""Temporal aggregation engine — shared between ACT, ViTAL, and Ablation.

Extracted from the near-identical ``class ACT`` wrappers in:
    - policy/ACT/act_policy.py  (lines 98-236)
    - policy/ViTAL/policy.py    (lines 186-330)

The engine handles action-chunk querying, exponential-weight temporal
aggregation, and timestep management.  It is pure numpy + torch — no
Omniverse / Isaac Sim dependency.
"""

from __future__ import annotations

import numpy as np
import torch


class TemporalAggregationEngine:
    """Exponential-weight temporal aggregation for action-chunk policies.

    Usage::

        engine = TemporalAggregationEngine(chunk_size=100, state_dim=8)
        for t in range(horizon):
            if t % engine.query_frequency == 0:
                chunk = policy(...)          # model forward pass
            action = engine.push(chunk)       # returns single aggregated action
            env.step(action)
    """

    def __init__(
        self,
        chunk_size: int,
        state_dim: int,
        temporal_agg: bool = True,
        max_timesteps: int = 3000,
        k: float = 0.01,
        device: str | torch.device = "cuda",
    ):
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.temporal_agg = temporal_agg
        self.max_timesteps = max_timesteps
        self.k = k
        self.device = torch.device(device) if isinstance(device, str) else device

        # Query every ``chunk_size`` steps unless temporal aggregation is on
        self.query_frequency = 1 if temporal_agg else chunk_size

        # Pre-allocated buffer for temporal aggregation
        self.all_time_actions: torch.Tensor | None = None
        if temporal_agg:
            self.all_time_actions = torch.zeros(
                (max_timesteps, max_timesteps + chunk_size, state_dim),
                device=self.device,
            )

        self.t: int = 0
        self._current_chunk: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, action_chunk: np.ndarray | torch.Tensor) -> np.ndarray:
        """Store a new action chunk and return the aggregated action for this step.

        Parameters
        ----------
        action_chunk: shape ``(1, chunk_size, state_dim)`` or ``(chunk_size, state_dim)``.

        Returns
        -------
        action: shape ``(state_dim,)`` — the single action to execute this step.
        """
        if isinstance(action_chunk, np.ndarray):
            action_chunk = torch.from_numpy(action_chunk).float().to(self.device)

        # Ensure [chunk_size, state_dim]
        while action_chunk.dim() > 2:
            action_chunk = action_chunk.squeeze(0)
        self._current_chunk = action_chunk

        if self.temporal_agg:
            assert self.all_time_actions is not None
            self.all_time_actions[
                [self.t], self.t : self.t + self.chunk_size
            ] = action_chunk

            actions_for_curr_step = self.all_time_actions[:, self.t]
            populated = torch.all(actions_for_curr_step != 0, dim=1)
            actions_for_curr_step = actions_for_curr_step[populated]

            exp_weights = np.exp(-self.k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = (
                torch.from_numpy(exp_weights).float().to(self.device).unsqueeze(1)
            )
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        else:
            raw_action = action_chunk[self.t % self.query_frequency].unsqueeze(0)

        self.t += 1
        return raw_action.cpu().numpy().flatten()

    def get_action(self, obs_encoded: dict, policy_callable) -> np.ndarray:
        """Convenience: call the policy and push the result.

        ``policy_callable`` receives the encoded observation and returns a
        numpy array of shape ``(1, chunk_size, state_dim)``.
        """
        if self.t % self.query_frequency == 0 or self._current_chunk is None:
            chunk = policy_callable(obs_encoded)
            return self.push(chunk)
        else:
            return self.push(self._current_chunk)

    def reset(self) -> None:
        """Clear temporal aggregation state for a new episode."""
        self.t = 0
        self._current_chunk = None
        if self.temporal_agg:
            self.all_time_actions = torch.zeros(
                (self.max_timesteps, self.max_timesteps + self.chunk_size, self.state_dim),
                device=self.device,
            )

    @property
    def current_timestep(self) -> int:
        return self.t
