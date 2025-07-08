import math
import os
import sys
import torch

import madrona_basketball as mba

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.')))
from src.constants import *


class EnvWrapper:
    def __init__(self, num_worlds: int, use_gpu: bool, gpu_id: int = 0):
        self.world_width_meters = WORLD_WIDTH_M
        self.world_height_meters = WORLD_HEIGHT_M

        world_discrete_width = math.ceil(self.world_width_meters)
        world_discrete_height = math.ceil(self.world_height_meters)

        self.worlds = mba.SimpleGridworldSimulator(
            discrete_x=world_discrete_width,
            discrete_y=world_discrete_height,
            start_x=self.world_width_meters / 2.0,
            start_y=self.world_height_meters / 2.0,
            max_episode_length=39600,
            exec_mode=mba.madrona.ExecMode.CUDA if use_gpu
            else mba.madrona.ExecMode.CPU,
            num_worlds=num_worlds,
            gpu_id=gpu_id
        )

        # Store RL tensor references
        self.observations = self.worlds.observations_tensor().to_torch()
        self.actions = self.worlds.action_tensor().to_torch()
        self.dones = self.worlds.done_tensor().to_torch()
        self.rewards = self.worlds.reward_tensor().to_torch()
        self.resets = self.worlds.reset_tensor().to_torch()
        self.agent_idx = 0

        print("Obs shape:", self.observations.shape)
        print("Actions shape:", self.actions.shape)
        print("Dones shape:", self.dones.shape)
        print("Rewards shape:", self.rewards.shape)
        print("Resets shape:", self.resets.shape)

        # Move/don't move  [0, 1]
        # Move angle       [0, 7]
        # Rotate           [0, 2]
        # Grab             [0, 1]
        # Pass             [0, 1]
        # Shoot            [0, 1]
        self.action_buckets = [2, 8, 3, 2, 2, 2]

    def get_action_space_size(self):
        return len(self.action_buckets)

    def get_input_dim(self):
        return self.observations.shape[-1]

    def get_action_buckets(self):
        return self.action_buckets

    def set_agent_idx(self, agent_idx):
        self.agent_idx = agent_idx

    def step(self, actions):
        self.actions[:, self.agent_idx] = actions

        self.worlds.step()
        obs = self.observations[:, self.agent_idx].detach().clone()
        rew = self.rewards[:, self.agent_idx].detach().clone()
        done = self.dones[:, self.agent_idx].detach().clone()
        return obs, rew, done

    def reset(self):
        self.resets.fill_(1)
        dummy_actions = torch.zeros_like(self.actions[:, self.agent_idx])
        obs, rew, done = self.step(dummy_actions)
        self.resets.fill_(0)
        return obs, rew, done