"""Null policy — always outputs zero actions. Used for environment verification."""

import torch
import numpy as np

from _base_policy import BasePolicy


class Policy(BasePolicy):
    def __init__(self, args):
        super().__init__(args)
        self.task_name = args["task_name"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NullPolicy] task={self.task_name}, device={self.device}")
        print(f"[NullPolicy] This policy always outputs zero actions (0% success rate expected)")

    def encode_obs(self, observation):
        return observation

    def eval(self, task, observation):
        # Output zero action: 7 arm joints + 1 gripper = 8
        qpos = torch.zeros(8, device=task.device, dtype=torch.float32)
        # Slightly open gripper to avoid immediate collision
        qpos[-1] = 0.04  # small positive gripper opening
        task.take_action(qpos, action_type="qpos")

    def reset(self):
        pass

    def close(self):
        pass
