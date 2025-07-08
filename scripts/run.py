import math
import os
import sys

import madrona_basketball as mba
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from constants import *

num_worlds = int(sys.argv[1])


class EnvWrapper:
    def __init__(self):
        self.world_width_meters = WORLD_WIDTH_M
        self.world_height_meters = WORLD_HEIGHT_M
        self.pixels_per_meter = PIXELS_PER_METER

        # 2. These are the TRUE pixel dimensions for drawing, calculated only ONCE.
        self.world_width_px = self.world_width_meters * PIXELS_PER_METER
        self.world_height_px = self.world_height_meters * PIXELS_PER_METER

        # 3. This is the TRUE pixel offset for centering the world view.
        self.world_offset_x = (WINDOW_WIDTH - self.world_width_px) / 2
        self.world_offset_y = (WINDOW_HEIGHT - self.world_height_px) / 2

        # 4. The discrete grid is only needed to create the legacy `walls` array.
        #    It no longer affects the drawing coordinates.
        world_discrete_width = math.ceil(self.world_width_meters)
        world_discrete_height = math.ceil(self.world_height_meters)
        walls = np.zeros((world_discrete_height, world_discrete_width),
                         dtype=bool)
        rewards = np.zeros((world_discrete_height, world_discrete_width),
                           dtype=float)

        self.worlds = mba.SimpleGridworldSimulator(
            walls=walls,
            rewards=rewards,
            start_x=self.world_width_meters / 2.0,  # Start in the true center
            start_y=self.world_height_meters / 2.0,
            max_episode_length=39600,
            exec_mode=mba.madrona.ExecMode.CPU,
            num_worlds=1,
            gpu_id=-1
        )

    def step(self):
        self.worlds.step()


env = EnvWrapper()
for i in range(5):
    env.step()
