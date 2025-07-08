import math
import os
import sys

import madrona_basketball as mba

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.')))
from src.constants import *

num_worlds = int(sys.argv[1])


class EnvWrapper:
    def __init__(self):
        self.world_width_meters = WORLD_WIDTH_M
        self.world_height_meters = WORLD_HEIGHT_M

        world_discrete_width = math.ceil(self.world_width_meters)
        world_discrete_height = math.ceil(self.world_height_meters)

        self.worlds = mba.SimpleGridworldSimulator(
            discrete_x = world_discrete_width,
            discrete_y = world_discrete_height,
            start_x=self.world_width_meters / 2.0,
            start_y=self.world_height_meters / 2.0,
            max_episode_length=39600,
            exec_mode=mba.madrona.ExecMode.CUDA,
            num_worlds=1,
            gpu_id=0
        )

        # Store RL tensor references
        self.observations = self.worlds.observations_tensor().to_torch()
        self.actions = self.worlds.action_tensor().to_torch()
        self.dones = self.worlds.done_tensor().to_torch()
        self.rewards = self.worlds.reward_tensor().to_torch()
        self.resets = self.worlds.reset_tensor().to_torch()


    def step(self):
        self.worlds.step()


env = EnvWrapper()
for i in range(5):
    env.step()
