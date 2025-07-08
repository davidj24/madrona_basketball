import math
import os
import sys
import time

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
            discrete_x=world_discrete_width,
            discrete_y=world_discrete_height,
            start_x=self.world_width_meters / 2.0,
            start_y=self.world_height_meters / 2.0,
            max_episode_length=39600,
            exec_mode=mba.madrona.ExecMode.CUDA,
            num_worlds=num_worlds,
            gpu_id=0
        )

        # Store RL tensor references
        self.observations = self.worlds.observations_tensor().to_torch()
        self.actions = self.worlds.action_tensor().to_torch()
        self.dones = self.worlds.done_tensor().to_torch()
        self.rewards = self.worlds.reward_tensor().to_torch()
        self.resets = self.worlds.reset_tensor().to_torch()

        print("Obs shape:", self.observations.shape)
        print("Actions shape:", self.actions.shape)
        print("Dones shape:", self.dones.shape)
        print("Rewards shape:", self.rewards.shape)
        print("Resets shape:", self.resets.shape)

    def step(self):
        self.worlds.step()


# Example use case
def main():
    env = EnvWrapper()

    n_steps = 1000
    start_time = time.time()
    for i in range(n_steps):
        env.step()

    end_time = time.time()
    elapsed_frames = n_steps * num_worlds
    print(
        f"Average FPS: {elapsed_frames / (end_time - start_time):.2f} frames/sec")


if __name__ == "__main__":
    main()
