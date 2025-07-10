import math
import os
import sys
import torch

import madrona_basketball as mba

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.')))
from src.constants import *
from viewer import ViewerClass


class EnvWrapper:
    def __init__(self, num_worlds: int, use_gpu: bool, gpu_id: int = 0, viewer: bool = False):
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
        
        print("✓ Simulation created and compiled successfully!")

        self.viewer = None
        if viewer:
            if num_worlds > 1:
                print("Viewer is enabled. Only rendering world 0")
            try:
                # CRITICAL: Wait for GPU compilation to complete before creating viewer
                if use_gpu:
                    print("🔧 GPU simulation ready, now initializing viewer...")
                    # Ensure GPU compilation is completely finished
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            print("✓ CUDA context ready for viewer integration")
                    except Exception as cuda_e:
                        print(f"⚠ CUDA sync warning before viewer: {cuda_e}")
                else:
                    print("🔧 CPU simulation ready, now initializing viewer...")
                
                # Test simulation access before creating viewer
                print("Testing simulation access...")
                test_obs = self.worlds.observations_tensor().to_torch()
                print(f"✓ Simulation accessible, obs shape: {test_obs.shape}")
                
                # Now it's safe to create the viewer
                print("Creating viewer...")
                # Create viewer with training mode flag to prevent action input conflicts
                self.viewer = ViewerClass(sim_instance=self.worlds, training_mode=True)
                print("✓ Viewer created successfully!")
                
            except Exception as e:
                print(f"❌ CRITICAL: Failed to create viewer: {e}")
                print("This is the likely source of the GPU compilation hang!")
                import traceback
                traceback.print_exc()
                print("Continuing without viewer to prevent crash")
                self.viewer = None

        # Store RL tensor references
        self.observations = self.worlds.observations_tensor().to_torch()
        self.actions = self.worlds.action_tensor().to_torch()
        self.dones = self.worlds.done_tensor().to_torch()
        self.rewards = self.worlds.reward_tensor().to_torch()
        self.resets = self.worlds.reset_tensor().to_torch()
        self.agent_idx = 0
        
        # Track if this is the first step to avoid calling viewer too early
        self.first_reset_done = False

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
        
        # Only call viewer after first reset is complete and tensors are ready
        if self.viewer is not None and self.first_reset_done:
            try:
                self.viewer.tick()
            except Exception as e:
                print(f"Warning: Viewer error: {e}")
                # Disable viewer on error to prevent crashes
                print("Disabling viewer due to error")
                self.viewer = None
            
        obs = self.observations[:, self.agent_idx].detach().clone()
        rew = self.rewards[:, self.agent_idx].detach().clone()
        done = self.dones[:, self.agent_idx].detach().clone()
        return obs, rew, done

    def get_blank_actions(self):
        return torch.zeros_like(self.actions[:, self.agent_idx])

    def reset(self):
        self.resets.fill_(1)
        dummy_actions = torch.zeros_like(self.actions[:, self.agent_idx])
        obs, rew, done = self.step(dummy_actions)
        self.resets.fill_(0)
        
        # Mark that first reset is complete - now safe to use viewer
        self.first_reset_done = True
        
        return obs, rew, done