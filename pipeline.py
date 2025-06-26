#!/usr/bin/env python3
"""
Simple Pipeline: Madrona C++ Simulation → Pygame Visualization
This connects to whatever simulation you build in Madrona and displays it
"""

import pygame
import sys
import numpy as np
import os

# Disable CUDA before importing anything else to avoid version conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Try to import and initialize PyTorch early to avoid issues later
try:
    import torch
    torch.set_num_threads(1)  # Limit CPU threads
    print("✓ PyTorch imported successfully")
except Exception as e:
    print(f"⚠ PyTorch import issue: {e}")

# Add build directory to path for the C++ module
sys.path.append('./build')

try:
    import _madrona_simple_example_cpp as madrona_sim
    from _madrona_simple_example_cpp.madrona import ExecMode
    print("✓ Successfully imported Madrona C++ module")
except ImportError as e:
    print(f"✗ Failed to import Madrona C++ module: {e}")
    print("Make sure you've built the project first with 'cmake --build build'")
    sys.exit(1)

# Simple visualization constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BACKGROUND_COLOR = (50, 50, 50)  # Dark gray
TEXT_COLOR = (255, 255, 255)     # White

class MadronaPipeline:
    """
    Simple pipeline that connects to your Madrona simulation and displays the data
    Modify your C++ code, and this will show you what's happening
    """
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Madrona Simulation Pipeline")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Initialize your Madrona simulation
        # (This uses the existing gridworld setup - you can modify this as needed)
        print("Initializing Madrona simulation...")
        
        # Create a simple 5x5 grid for the existing simulation
        walls = np.zeros((5, 5), dtype=bool)
        rewards = np.zeros((5, 5), dtype=float)
        end_cells = np.array([[4, 4]], dtype=np.int32)  # Goal at bottom-right
        
        self.sim = madrona_sim.SimpleGridworldSimulator(
            walls=walls,
            rewards=rewards, 
            end_cells=end_cells,
            start_x=0,
            start_y=0,
            max_episode_length=100,
            exec_mode=ExecMode.CPU,
            num_worlds=1,
            gpu_id=-1
        )
        
        print("✓ Madrona simulation initialized!")
        self.step_count = 0
        
    def get_simulation_data(self):
        """Get the current state from your Madrona simulation"""
        try:
            # Get tensors from your simulation
            obs_tensor = self.sim.observation_tensor()
            action_tensor = self.sim.action_tensor() 
            reward_tensor = self.sim.reward_tensor()
            done_tensor = self.sim.done_tensor()
            reset_tensor = self.sim.reset_tensor()
            basketballpos_tensor = self.sim.basketball_pos_tensor()


            
            # Convert to numpy arrays using torch backend (CPU only mode already set)
            obs_data = obs_tensor.to_torch().detach().cpu().numpy()      # Position data
            action_data = action_tensor.to_torch().detach().cpu().numpy() # Action data
            reward_data = reward_tensor.to_torch().detach().cpu().numpy() # Reward data
            done_data = done_tensor.to_torch().detach().cpu().numpy()     # Episode done flags
            reset_data = reset_tensor.to_torch().detach().cpu().numpy()   # Reset flags
            basketball_pos_data = basketballpos_tensor.to_torch().detach().cpu().numpy()  # Basketball position
            
            return {
                'observations': obs_data,
                'actions': action_data,
                'rewards': reward_data,
                'done': done_data,
                'reset': reset_data,
                'basketball_pos': basketball_pos_data
            }
            
        except Exception as e:
            print(f"Error getting simulation data: {e}")
            return None
        



    
    def draw_simulation_data(self, data):
        """Draw whatever data your simulation produces"""
        if data is None:
            return
            
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)
        
        # Display the raw data from your simulation
        y_offset = 20
        
        texts = [
            f"Madrona Simulation Pipeline - Step {self.step_count}",
            "",
            f"Observations: {data['observations'].flatten()}",
            f"Actions: {data['actions'].flatten()}",
            f"Rewards: {data['rewards'].flatten()}",
            f"Done: {data['done'].flatten()}",
            f"Reset: {data['reset'].flatten()}",
            "",
            "This shows the raw data from your C++ simulation.",
            "Modify the C++ code to see different values here!",
            "",
            "Controls:",
            "SPACE - Step simulation manually",
            "R - Reset simulation", 
            "ESC - Quit"
        ]
        
        for text in texts:
            if text:  # Skip empty lines
                surface = self.font.render(text, True, TEXT_COLOR)
                self.screen.blit(surface, (20, y_offset))
            y_offset += 25



        if 'basketball_pos' in data:
            positions = data['basketball_pos']
            
            for pos in positions:
                if len(pos) >= 2:  # Make sure we have x,y coordinates
                    # Convert grid coordinates to screen pixels
                    screen_x = pos[0] * 32 + 50  # Scale and offset
                    screen_y = pos[1] * 32 + 50
                    
                    # Draw circle (or whatever shape you want)
                    pygame.draw.circle(self.screen, (255, 0, 0), 
                                    (int(screen_x), int(screen_y)), 10)



    
    def step_simulation(self):
        """Step your Madrona simulation forward"""
        self.sim.step()
        self.step_count += 1
    
    def reset_simulation(self):
        """Reset your Madrona simulation"""
        try:
            reset_tensor = self.sim.reset_tensor()
            reset_data = reset_tensor.to_torch().detach().cpu().numpy()
            reset_data.fill(1)  # Trigger reset
            print(f"Simulation reset at step {self.step_count}")
            self.step_count = 0
        except Exception as e:
            print(f"Error resetting simulation: {e}")
    
    def run(self):
        """Main pipeline loop"""
        running = True
        auto_step = True  # Whether to step automatically
        
        print("Madrona Pipeline Started!")
        print("- The simulation will step automatically")
        print("- Press SPACE to toggle manual stepping")
        print("- Press R to reset")
        print("- Press ESC to quit")
        print()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_simulation()
                    elif event.key == pygame.K_SPACE:
                        auto_step = not auto_step
                        print(f"Auto-stepping: {'ON' if auto_step else 'OFF'}")
                        if not auto_step:
                            self.step_simulation()  # Manual step
            
            # Step simulation
            if auto_step:
                self.step_simulation()
            
            # Get and display data
            data = self.get_simulation_data()
            self.draw_simulation_data(data)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(10)  # 10 FPS for easy viewing
        
        pygame.quit()
        print(f"Pipeline ended after {self.step_count} steps")

if __name__ == "__main__":
    try:
        pipeline = MadronaPipeline()
        pipeline.run()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        print("\nMake sure you've:")
        print("1. Built the project: cmake --build build")
        print("2. The C++ simulation is working")
        sys.exit(1)
