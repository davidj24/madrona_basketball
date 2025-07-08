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
    num_stats = 0

    def update(self, loss: float, action_loss: float, value_loss: float, entropy_loss: float,
               returns_mean: float, returns_stddev: float):
        self.num_stats += 1
        self.loss += (loss - self.loss) / self.num_stats
        self.action_loss += (action_loss - self.action_loss) / self.num_stats
        self.value_loss += (value_loss - self.value_loss) / self.num_stats
        self.entropy_loss += (entropy_loss - self.entropy_loss) / self.num_stats
        self.returns_mean += (returns_mean - self.returns_mean) / self.num_stats
        self.returns_stddev += (returns_stddev - self.returns_stddev) / self.num_stats
