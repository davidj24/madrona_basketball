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
WINDOW_WIDTH = 2000
WINDOW_HEIGHT = 1500
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
        
        # Configure world size (make it bigger!)
        self.world_width = 20   # Changed from 5 to 10
        self.world_height = 20   # Changed from 5 to 8
        self.cell_size = 40     # Size of each cell in pixels
        self.grid_offset_x = 50 # Offset from screen edge
        self.grid_offset_y = 100
        
        print("Initializing Madrona simulation...")
        
        # Create a larger grid
        walls = np.zeros((self.world_height, self.world_width), dtype=bool)
        rewards = np.zeros((self.world_height, self.world_width), dtype=float)
        end_cells = np.array([[self.world_height-1, self.world_width-1]], dtype=np.int32)  # Goal at bottom-right
        
        self.sim = madrona_sim.SimpleGridworldSimulator(
            walls=walls,
            rewards=rewards, 
            end_cells=end_cells,
            start_x=0,
            start_y=0,
            max_episode_length=1000,
            exec_mode=ExecMode.CPU,
            num_worlds=1,
            gpu_id=-1
        )
        
        print(f"✓ Madrona simulation initialized! World size: {self.world_width}x{self.world_height}")
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
        



    
    def draw_grid_boundaries(self):
        """Draw the world boundaries and grid"""
        # Draw grid lines
        for x in range(self.world_width + 1):
            start_pos = (self.grid_offset_x + x * self.cell_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + x * self.cell_size, 
                      self.grid_offset_y + self.world_height * self.cell_size)
            pygame.draw.line(self.screen, (100, 100, 100), start_pos, end_pos, 1)
        
        for y in range(self.world_height + 1):
            start_pos = (self.grid_offset_x, self.grid_offset_y + y * self.cell_size)
            end_pos = (self.grid_offset_x + self.world_width * self.cell_size, 
                      self.grid_offset_y + y * self.cell_size)
            pygame.draw.line(self.screen, (100, 100, 100), start_pos, end_pos, 1)
        
        # Draw boundary rectangle (thicker)
        boundary_rect = pygame.Rect(
            self.grid_offset_x, 
            self.grid_offset_y, 
            self.world_width * self.cell_size, 
            self.world_height * self.cell_size
        )
        pygame.draw.rect(self.screen, (200, 200, 200), boundary_rect, 3)
    
    def grid_to_screen(self, grid_x, grid_y):
        """Convert grid coordinates to screen pixels"""
        screen_x = self.grid_offset_x + (grid_x * self.cell_size) + (self.cell_size // 2)
        screen_y = self.grid_offset_y + (grid_y * self.cell_size) + (self.cell_size // 2)
        return int(screen_x), int(screen_y)
    
    def draw_simulation_data(self, data):
        """Draw whatever data your simulation produces"""
        if data is None:
            return
            
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw the grid boundaries first
        self.draw_grid_boundaries()
        
        # Display info text
        y_offset = 20
        info_texts = [
            f"Madrona Basketball Simulation - Step {self.step_count}",
            f"World Size: {self.world_width}x{self.world_height}",
            "",
            "Controls: WASD/Arrow Keys, SPACE=manual step, R=reset, ESC=quit"
        ]
        
        for text in info_texts:
            if text:
                surface = self.font.render(text, True, TEXT_COLOR)
                self.screen.blit(surface, (20, y_offset))
            y_offset += 20


        # Draw basketball positions
        if 'basketball_pos' in data:
            raw_positions = data['basketball_pos']  # Shape: (1, num_basketballs, 2)
            positions = raw_positions[0]  # Get first world, shape: (num_basketballs, 2)

            # Draw basketball positions with different colors
            colors = [(255, 100, 0), (255, 200, 0), (100, 255, 0)]  # Orange, Yellow, Green
            
            for i, pos in enumerate(positions):
                if len(pos) >= 2:
                    screen_x, screen_y = self.grid_to_screen(pos[0], pos[1])
                    color = colors[i % len(colors)]
                    pygame.draw.circle(self.screen, color, (screen_x, screen_y), 12)
                    # Add basketball pattern
                    pygame.draw.circle(self.screen, (200, 50, 0), (screen_x, screen_y), 12, 2)
                    
                    # Add a number to identify each basketball
                    font_small = pygame.font.Font(None, 16)
                    text_surface = font_small.render(str(i + 1), True, (255, 255, 255))
                    self.screen.blit(text_surface, (screen_x - 5, screen_y - 5))

        # Draw agent positions  
        if 'observations' in data:
            raw_positions = data['observations']  # Shape: (1, num_basketballs, 2)
            positions = raw_positions[0]  # Get first world, shape: (num_basketballs, 2)

            # Draw basketball positions with different colors
            colors = [(255, 100, 0), (255, 200, 0), (100, 255, 0)]  # Orange, Yellow, Green
            
            for i, pos in enumerate(positions):
                if len(pos) >= 2:
                    screen_x, screen_y = self.grid_to_screen(pos[0], pos[1])
                    color = colors[i % len(colors)]
                    pygame.draw.rect(self.screen, color, (screen_x, screen_y, 12, 12))
                    pygame.draw.rect(self.screen, (200, 50, 0), (screen_x, screen_y, 12, 12), 2)
                    font_small = pygame.font.Font(None, 16)
                    text_surface = font_small.render(str(i + 1), True, (255, 255, 255))
                    self.screen.blit(text_surface, (screen_x - 5, screen_y - 5))

    
    def step_simulation(self):
        """Step your Madrona simulation forward"""
        self.sim.step()
        self.step_count += 1
    
    def reset_simulation(self):
        """Reset your Madrona simulation"""
        try:
            # Use the proper input injection interface
            self.sim.trigger_reset(0)  # Reset world 0
            print(f"Simulation reset at step {self.step_count}")
            self.step_count = 0
        except Exception as e:
            print(f"Error resetting simulation: {e}")
    
    def handle_input(self):
        """Handle keyboard input and inject actions into simulation"""
        keys = pygame.key.get_pressed()
        
        # Agent 0 actions (WASD keys)
        agent0_action = 4  # Default to Action::None
        if keys[pygame.K_w]:
            agent0_action = 0  # Action::Up
        elif keys[pygame.K_s]:
            agent0_action = 1  # Action::Down
        elif keys[pygame.K_a]:
            agent0_action = 2  # Action::Left
        elif keys[pygame.K_d]:
            agent0_action = 3  # Action::Right
        
        # Agent 1 actions (Arrow keys)
        agent1_action = 4
        if keys[pygame.K_UP]:
            agent1_action = 0  # Action::Up
        elif keys[pygame.K_DOWN]:
            agent1_action = 1  # Action::Down
        elif keys[pygame.K_LEFT]:
            agent1_action = 2  # Action::Left
        elif keys[pygame.K_RIGHT]:
            agent1_action = 3  # Action::Right
        
        # Inject actions for both agents
        self.sim.set_action(0, 0, agent0_action)  # World 0, Agent 0
        self.sim.set_action(0, 1, agent1_action)  # World 0, Agent 1
    
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
            
            # Handle continuous input (following Madrona viewer pattern)
            self.handle_input()
            
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
