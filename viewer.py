#!/usr/bin/env python3
"""
Simple Pipeline: Madrona C++ Simulation → Pygame Visualization
This connects to whatever simulation you build in Madrona and displays it
"""

import pygame
import sys
import numpy as np
import os



# ================================ Config Constants ================================
WINDOW_WIDTH = 3000
WINDOW_HEIGHT = 1750
BACKGROUND_COLOR = (50, 50, 50)  # Dark gray
TEXT_COLOR = (255, 255, 255)     # White

WORLD_WIDTH = 51 
WORLD_HEIGHT = 35  
CELL_SIZE = 45     # For lab computer
# CELL_SIZE = 30     # For laptop
GRID_OFFSET_X = 250 # Offset from screen edge
GRID_OFFSET_Y = 100 # Offset from screen edge




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
        self.world_width = WORLD_WIDTH
        self.world_height = WORLD_HEIGHT
        self.cell_size = CELL_SIZE
        self.grid_offset_x = GRID_OFFSET_X
        self.grid_offset_y = GRID_OFFSET_Y

        print("Initializing Madrona simulation...")
        
        # Create a larger grid
        walls = np.zeros((self.world_height, self.world_width), dtype=bool)
        rewards = np.zeros((self.world_height, self.world_width), dtype=float)
        
        self.sim = madrona_sim.SimpleGridworldSimulator(
            walls=walls,
            rewards=rewards, 
            start_x=self.world_width // 2,
            start_y=self.world_height // 2,
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
            
            # Debug print shapes (only first few times)
            if self.step_count < 3:
                print(f"Debug - Step {self.step_count}:")
                print(f"  obs_data shape: {obs_data.shape}, values: {obs_data}")
                print(f"  action_data shape: {action_data.shape}, values: {action_data}")
                print(f"  basketball_pos_data shape: {basketball_pos_data.shape}, values: {basketball_pos_data}")
            
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
            import traceback
            traceback.print_exc()
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
    
    
    def draw_basketball_court(self):
        """Draws a basketball court that dynamically fits the grid."""
        # Define court colors
        COURT_ORANGE = (205, 133, 63)
        COURT_BLUE = (20, 40, 80)
        LINE_WHITE = (255, 255, 255)
        KEY_RED = (139, 0, 0)
        LINE_THICKNESS = 3

        # --- Dynamic Court Dimensions ---
        margin_cells = 1
        court_start_x = self.grid_offset_x + margin_cells * self.cell_size
        court_start_y = self.grid_offset_y + margin_cells * self.cell_size
        court_width = (self.world_width - 2 * margin_cells) * self.cell_size
        court_height = (self.world_height - 2 * margin_cells) * self.cell_size

        if court_width <= 0 or court_height <= 0: return

        court_rect = pygame.Rect(court_start_x, court_start_y, court_width, court_height)

        # --- Draw Court Background ---
        pygame.draw.rect(self.screen, COURT_ORANGE, court_rect)
        playable_rect = court_rect.inflate(-LINE_THICKNESS * 2, -LINE_THICKNESS * 2)
        pygame.draw.rect(self.screen, COURT_BLUE, playable_rect)

        # --- Draw Court Markings ---
        center_x, center_y = court_rect.centerx, court_rect.centery

        # Center line
        pygame.draw.line(self.screen, LINE_WHITE, (center_x, court_rect.top), (center_x, court_rect.bottom), LINE_THICKNESS)

        # Center circle
        center_circle_radius = int(court_height * 0.15)
        pygame.draw.circle(self.screen, LINE_WHITE, (center_x, center_y), center_circle_radius, LINE_THICKNESS)
        pygame.draw.circle(self.screen, KEY_RED, (center_x, center_y), center_circle_radius - LINE_THICKNESS)

        # --- Draw Features for Both Halves ---
        for side in [-1, 1]:  # -1 for left side, 1 for right side
            # Key (the "paint") - Adjusted height and restored width
            key_width = int(court_width * 0.25) # Restored from 0.20
            key_height = int(court_height * 0.25) # Changed from 0.4 to make it shorter
            key_x = court_rect.left if side == -1 else court_rect.right - key_width
            key_y = center_y - key_height / 2
            key_rect = pygame.Rect(key_x, key_y, key_width, key_height)
            pygame.draw.rect(self.screen, KEY_RED, key_rect)

            # Free-throw circle (semi-circle)
            ft_circle_radius = int(key_height / 2)
            ft_center_x = court_rect.left + key_width if side == -1 else court_rect.right - key_width
            arc_rect = pygame.Rect(ft_center_x - ft_circle_radius, center_y - ft_circle_radius, ft_circle_radius * 2, ft_circle_radius * 2)
            
            start_angle = -np.pi / 2 if side == -1 else np.pi / 2
            end_angle = np.pi / 2 if side == -1 else 3 * np.pi / 2
            pygame.draw.arc(self.screen, LINE_WHITE, arc_rect, start_angle, end_angle, LINE_THICKNESS)

            # Three-point line (recalculated to align with the key)
            hoop_offset_x = int(self.cell_size * 1.5)
            hoop_center_x = court_rect.left + hoop_offset_x if side == -1 else court_rect.right - hoop_offset_x
            
            # New radius calculation: distance from hoop to the top of the free-throw circle, plus a small gap.
            dist_hoop_to_ft_line = abs(ft_center_x - hoop_center_x)
            three_pt_radius = dist_hoop_to_ft_line + ft_circle_radius + int(self.cell_size * 0.5)

            three_pt_arc_rect = pygame.Rect(hoop_center_x - three_pt_radius, center_y - three_pt_radius, three_pt_radius * 2, three_pt_radius * 2)
            pygame.draw.arc(self.screen, LINE_WHITE, three_pt_arc_rect, start_angle, end_angle, LINE_THICKNESS)

            # Backboard and Hoop
            backboard_x = court_rect.left + hoop_offset_x if side == -1 else court_rect.right - hoop_offset_x
            pygame.draw.line(self.screen, LINE_WHITE, (backboard_x, center_y - 30), (backboard_x, center_y + 30), LINE_THICKNESS + 2)
            pygame.draw.circle(self.screen, COURT_ORANGE, (backboard_x, center_y), 8, 0) # The hoop
            pygame.draw.circle(self.screen, LINE_WHITE, (backboard_x, center_y), 8, 2) # Hoop rim
    


    def draw_simulation_data(self, data):
        """Draw whatever data your simulation produces"""
        if data is None:
            return
            
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_basketball_court()
        self.draw_grid_boundaries()
        
        # Display info text
        y_offset = 20
        info_texts = [
            f"Madrona Basketball Simulation - Step {self.step_count}",
            f"World Size: {self.world_width}x{self.world_height}",
            "",
            "Controls: WASD/Arrow Keys, SPACE=manual step, R=reset, ESC=quit"
        ]

        # Add action debugging for enhanced actions
        if 'actions' in data:
            actions = data['actions'][0]  # Get first world, shape: (num_agents, 4)
            for i, action_components in enumerate(actions):
                if len(action_components) >= 4:
                    move_speed, move_angle, rotate, grab = action_components
                    info_texts.append(f"Agent {i}: Speed={int(move_speed)} Angle={int(move_angle)} Rotate={int(rotate)} Grab={int(grab)}")
                else:
                    info_texts.append(f"Agent {i}: Invalid action data")

        
        # Rewards are now shape (1, 2) instead of (1, 1)
        if 'rewards' in data:
            rewards = data['rewards'][0]  # Shape: (num_agents,)
            for i, reward in enumerate(rewards):
                info_texts.append(f"Agent {i} Reward: {reward:.2f}")
        
        # Done flags are now shape (1, 2) instead of (1, 1)  
        if 'done' in data:
            done_flags = data['done'][0]  # Shape: (num_agents,)
            for i, done in enumerate(done_flags):
                info_texts.append(f"Agent {i} Done: {done}")
            

            for text in info_texts:
                if text:
                    surface = self.font.render(text, True, TEXT_COLOR)
                    self.screen.blit(surface, (20, y_offset))
                y_offset += 20


        # Draw basketball positions
        if 'basketball_pos' in data:
            raw_positions = data['basketball_pos']  # Shape: (1, num_basketballs, 3) - x,y,z
            positions = raw_positions[0]  # Get first world, shape: (num_basketballs, 3)

            # Draw basketball positions with different colors
            colors = [(255, 100, 0), (255, 200, 0), (100, 255, 0)]  # Orange, Yellow, Green
            
            for i, pos in enumerate(positions):
                if len(pos) >= 2:  # Use x,y from the 3-component position (x,y,z)
                    screen_x, screen_y = self.grid_to_screen(pos[0], pos[1])
                    color = colors[i % len(colors)]
                    pygame.draw.circle(self.screen, color, (screen_x, screen_y), 12)
                    # Add basketball pattern
                    pygame.draw.circle(self.screen, (200, 50, 0), (screen_x, screen_y), 12, 2)
                    
                    # Add a number to identify each basketball
                    font_small = pygame.font.Font(None, 16)
                    text_surface = font_small.render(f"B{i + 1}", True, (255, 255, 255))
                    self.screen.blit(text_surface, (screen_x - 8, screen_y - 5))

        # Draw agent positions  
        if 'observations' in data:
            raw_positions = data['observations']  # Shape: (1, num_agents, 3) - x,y,z
            positions = raw_positions[0]  # Get first world, shape: (num_agents, 3)

            # Draw agents with distinct colors (different from basketballs)
            agent_colors = [(0, 100, 255), (255, 0, 100)]  # Blue, Pink - distinct from basketball orange
            
            for i, pos in enumerate(positions):
                if len(pos) >= 2:  # Use x,y from the 3-component position (x,y,z)
                    screen_x, screen_y = self.grid_to_screen(pos[0], pos[1])
                    color = agent_colors[i % len(agent_colors)]
                    # Draw agents as rectangles to distinguish from circular basketballs
                    pygame.draw.rect(self.screen, color, (screen_x - 8, screen_y - 8, 16, 16))
                    pygame.draw.rect(self.screen, (255, 255, 255), (screen_x - 8, screen_y - 8, 16, 16), 2)
                    
                    # Add agent number
                    font_small = pygame.font.Font(None, 16)
                    text_surface = font_small.render(f"A{i + 1}", True, (255, 255, 255))
                    self.screen.blit(text_surface, (screen_x - 8, screen_y - 20))

    
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
        """Handle keyboard input and inject enhanced actions into simulation"""
        keys = pygame.key.get_pressed()
        
        # Agent 0 actions (WASD keys)
        move_speed = 0
        move_angle = 0
        rotate = 0
        grab = 0
        
        # Movement (WASD)
        if keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d]:
            move_speed = 1  # Normal speed
            
            # 8-directional movement
            if keys[pygame.K_w] and keys[pygame.K_d]:      # NE
                move_angle = 1
            elif keys[pygame.K_d] and keys[pygame.K_s]:    # SE  
                move_angle = 3
            elif keys[pygame.K_s] and keys[pygame.K_a]:    # SW
                move_angle = 5
            elif keys[pygame.K_a] and keys[pygame.K_w]:    # NW
                move_angle = 7
            elif keys[pygame.K_w]:                         # N
                move_angle = 0
            elif keys[pygame.K_d]:                         # E
                move_angle = 2
            elif keys[pygame.K_s]:                         # S
                move_angle = 4
            elif keys[pygame.K_a]:                         # W
                move_angle = 6
        
        # Rotation (Q/E)
        if keys[pygame.K_q]:
            rotate = -1  # Turn left
        elif keys[pygame.K_e]:
            rotate = 1   # Turn right
        
        # Grab (Space)
        if keys[pygame.K_SPACE]:
            grab = 1
        
        # Agent 1 actions (Arrow keys + other keys)
        move_speed1 = 0
        move_angle1 = 0  
        rotate1 = 0
        grab1 = 0
        
        if keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
            move_speed1 = 1
            
            if keys[pygame.K_UP]:
                move_angle1 = 0
            elif keys[pygame.K_RIGHT]:
                move_angle1 = 2
            elif keys[pygame.K_DOWN]:
                move_angle1 = 4
            elif keys[pygame.K_LEFT]:
                move_angle1 = 6
        
        if keys[pygame.K_COMMA]:  # '<' key
            rotate1 = -1
        elif keys[pygame.K_PERIOD]:  # '>' key  
            rotate1 = 1
        
        if keys[pygame.K_RSHIFT]:
            grab1 = 1
        
        # Send enhanced actions to simulation
        self.sim.set_action(0, 0, move_speed, move_angle, rotate, grab)
        self.sim.set_action(0, 1, move_speed1, move_angle1, rotate1, grab1)
    
    def run(self):
        """Main pipeline loop"""
        running = True
        auto_step = True  # Whether to step automatically
        
        print("Madrona Pipeline Started!")
        print("- The simulation will step automatically")
        print("- Press F to toggle manual stepping")
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
                    elif event.key == pygame.K_f:
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
