from time import perf_counter

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

    def update(self,
               loss: float,
               action_loss: float,
               value_loss: float,
               entropy_loss: float,
               returns_mean: float,
               returns_stddev: float,
               rewards_mean: float,
               rewards_min: float,
               rewards_max: float):
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


class PPOTimer:
    def __init__(self):
        self.t_iter = 0.0
        self.t_sim = 0.0
        self.t_inference = 0.0
        self.t_rollout = 0.0
        self.t_update = 0.0
        self.iter_step = 0

        self.global_step = 0

        # Temporary
        self.iter_start = None
        self.rollout_start = None
        self.sim_start = None
        self.inference_start = None
        self.update_start = None

    def start_iter(self):
        self.iter_start = perf_counter()

    def end_iter(self):
        assert (self.iter_start is not None), "Iteration start not set"
        elapsed = perf_counter() - self.iter_start
        self.t_iter += elapsed
        self.iter_start = None

    def reset(self):
        self.t_iter = 0.0
        self.t_sim = 0.0
        self.t_inference = 0.0
        self.t_rollout = 0.0
        self.t_update = 0.0
        self.iter_step = 0

    def start_rollout(self):
        self.rollout_start = perf_counter()

    def end_rollout(self):
        assert (self.rollout_start is not None), "Rollout start not set"
        elapsed = perf_counter() - self.rollout_start
        self.t_rollout += elapsed
        self.rollout_start = None

    def start_sim(self):
        self.sim_start = perf_counter()

    def end_sim(self):
        assert (self.sim_start is not None), "Simulation start not set"
        elapsed = perf_counter() - self.sim_start
        self.t_sim += elapsed
        self.sim_start = None

    def start_inference(self):
        self.inference_start = perf_counter()

    def end_inference(self):
        assert (self.inference_start is not None), "Inference start not set"
        elapsed = perf_counter() - self.inference_start
        self.t_inference += elapsed
        self.inference_start = None

    def start_update(self):
        self.update_start = perf_counter()

    def end_update(self):
        assert (self.update_start is not None), "Update start not set"
        elapsed = perf_counter() - self.update_start
        self.t_update += elapsed
        self.update_start = None

    def get_iter_fps(self):
        return int(self.iter_step / self.t_iter) if self.t_iter > 0 else 0

    def get_rollout_fps(self):
        return int(self.iter_step / self.t_rollout) if self.t_rollout > 0 else 0

    def get_sim_fps(self):
        return int(self.iter_step / self.t_sim) if self.t_sim > 0 else 0

    def get_inference_fps(self):
        return int(self.iter_step / self.t_inference) if self.t_inference > 0 else 0

    def get_update_fps(self):
        return int(self.iter_step / self.t_update) if self.t_update > 0 else 0

    def update_step(self, steps: int):
        self.iter_step += steps
        self.global_step += steps
