import torch
from dataclasses import dataclass


@dataclass
class PPOStats:
    loss: float = 0
    action_loss: float = 0
    value_loss: float = 0
    entropy_loss: float = 0
    returns_mean: float = 0
    returns_stddev: float = 0
    rewards_mean: float = 0
    rewards_min: float = float('inf')
    rewards_max: float = float('-inf')
    num_stats: int = 0

    def update(self, loss: float, action_loss: float, value_loss: float, entropy_loss: float,
               returns_mean: float, returns_stddev: float, rewards_mean: float,
                rewards_min: float, rewards_max: float):
        self.num_stats += 1
        self.loss += (loss - self.loss) / self.num_stats
        self.action_loss += (action_loss - self.action_loss) / self.num_stats
        self.value_loss += (value_loss - self.value_loss) / self.num_stats
        self.entropy_loss += (entropy_loss - self.entropy_loss) / self.num_stats
        self.returns_mean += (returns_mean - self.returns_mean) / self.num_stats
        self.returns_stddev += (returns_stddev - self.returns_stddev) / self.num_stats
        self.rewards_mean += (rewards_mean - self.rewards_mean) / self.num_stats
        self.rewards_min = min(self.rewards_min, rewards_min) if self.num_stats > 1 else rewards_min
        self.rewards_max = max(self.rewards_max, rewards_max) if self.num_stats > 1 else rewards_max

    def reset(self):
        self.num_stats = 0

        self.loss = 0
        self.action_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        self.returns_mean = 0
        self.returns_stddev = 0
        self.rewards_mean = 0
        self.rewards_min = float('inf')
        self.rewards_max = float('-inf')