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

# World dimensions in meters (continuous coordinates)
WORLD_WIDTH_METERS = 71.0  
WORLD_HEIGHT_METERS = 51.0  
METERS_TO_PIXELS = 30      # Pixels per meter for visualization
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

def rotate_vec(q, v):
    """
    A direct Python translation of the C++ Quat::rotateVec function.
    Rotates a vector v by a quaternion q.
    
    Args:
        q (list or np.array): The quaternion in [w, x, y, z] order.
        v (list or np.array): The 3D vector [x, y, z] to rotate.
    
    Returns:
        np.array: The rotated 3D vector.
    """
    # Extract scalar and vector ('pure') parts of the quaternion
    scalar = q[0]
    pure = np.array([q[1], q[2], q[3]])
    
    # Ensure v is a numpy array
    v_vec = np.asarray(v)

    # C++: Vector3 pure_x_v = cross(pure, v);
    pure_x_v = np.cross(pure, v_vec)
    
    # C++: Vector3 pure_x_pure_x_v = cross(pure, pure_x_v);
    pure_x_pure_x_v = np.cross(pure, pure_x_v)
    
    # C++: return v + 2.f * ((pure_x_v * scalar) + pure_x_pure_x_v);
    return v_vec + 2.0 * ((pure_x_v * scalar) + pure_x_pure_x_v)

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
        
        # Configure world size in continuous coordinates (meters)
        self.world_width_meters = WORLD_WIDTH_METERS
        self.world_height_meters = WORLD_HEIGHT_METERS
        self.meters_to_pixels = METERS_TO_PIXELS
        self.grid_offset_x = GRID_OFFSET_X
        self.grid_offset_y = GRID_OFFSET_Y

        print("Initializing Madrona simulation...")
        
        # Create a discrete grid for simulation initialization (legacy support)
        # We'll use 1 cell per meter resolution
        discrete_width = int(self.world_width_meters)
        discrete_height = int(self.world_height_meters)
        walls = np.zeros((discrete_height, discrete_width), dtype=bool)
        rewards = np.zeros((discrete_height, discrete_width), dtype=float)
        
        self.sim = madrona_sim.SimpleGridworldSimulator(
            walls=walls,
            rewards=rewards, 
            start_x=self.world_width_meters / 2.0,  # Center of court
            start_y=self.world_height_meters / 2.0,  # Center of court
            max_episode_length=1000,
            exec_mode=ExecMode.CPU,
            num_worlds=1,
            gpu_id=-1
        )
        
        print(f"✓ Madrona simulation initialized! World size: {self.world_width_meters}x{self.world_height_meters} meters")
        self.step_count = 0
        
    def get_simulation_data(self):
        """Get the current state from your Madrona simulation"""
        try:
            # Get tensors from your simulation
            obs_tensor = self.sim.observation_tensor()
            agent_team_tensor = self.sim.agent_team_tensor()
            action_tensor = self.sim.action_tensor() 
            reward_tensor = self.sim.reward_tensor()
            done_tensor = self.sim.done_tensor()
            reset_tensor = self.sim.reset_tensor()
            basketball_pos_tensor = self.sim.basketball_pos_tensor()
            ball_physics_tensor = self.sim.ball_physics_tensor()
            hoop_pos_tensor = self.sim.hoop_pos_tensor()
            agent_possession_tensor = self.sim.agent_possession_tensor()
            ball_grabbed_tensor = self.sim.ball_grabbed_tensor()
            agent_entity_id_tensor = self.sim.agent_entity_id_tensor()
            ball_entity_id_tensor = self.sim.ball_entity_id_tensor()
            agent_team_tensor = self.sim.agent_team_tensor()
            game_state_inbounding_tensor = self.sim.game_state_inbounding_tensor()
            orientation_tensor = self.sim.orientation_tensor()

            
            # Convert to numpy arrays using torch backend (CPU only mode already set)
            obs_data = obs_tensor.to_torch().detach().cpu().numpy()      # Position data
            agent_team_data = agent_team_tensor.to_torch().detach().cpu().numpy()  # Agent team data
            action_data = action_tensor.to_torch().detach().cpu().numpy() # Action data
            reward_data = reward_tensor.to_torch().detach().cpu().numpy() # Reward data
            done_data = done_tensor.to_torch().detach().cpu().numpy()     # Episode done flags
            reset_data = reset_tensor.to_torch().detach().cpu().numpy()   # Reset flags
            basketball_pos_data = basketball_pos_tensor.to_torch().detach().cpu().numpy()  # Basketball position
            ball_physics_data = ball_physics_tensor.to_torch().detach().cpu().numpy()
            hoop_pos_data = hoop_pos_tensor.to_torch().detach().cpu().numpy()  # Basketball position
            agent_possession_data = agent_possession_tensor.to_torch().detach().cpu().numpy()  # Agent possession
            ball_grabbed_data = ball_grabbed_tensor.to_torch().detach().cpu().numpy()  # Ball grabbed status
            agent_entity_id_data = agent_entity_id_tensor.to_torch().detach().cpu().numpy()  # Agent entity IDs
            ball_entity_id_data = ball_entity_id_tensor.to_torch().detach().cpu().numpy()  # Ball entity IDs
            game_state_inbounding_data = game_state_inbounding_tensor.to_torch().detach().cpu().numpy()
            orientation_data = orientation_tensor.to_torch().detach().cpu().numpy()  # Orientation data
            
            return {
                'observations': obs_data,
                'agent_team': agent_team_data,
                'actions': action_data,
                'rewards': reward_data,
                'done': done_data,
                'reset': reset_data,
                'basketball_pos': basketball_pos_data,
                'ball_physics' : ball_physics_data,
                'hoop_pos' : hoop_pos_data,
                'agent_possession': agent_possession_data,
                'ball_grabbed': ball_grabbed_data,
                'agent_entity_ids': agent_entity_id_data,
                'ball_entity_ids': ball_entity_id_data,
                'agent_teams': agent_team_data,
                'game_state_inbounding' : game_state_inbounding_data,
                'orientation': orientation_data
            }
            
        except Exception as e:
            print(f"Error getting simulation data: {e}")
            import traceback
            traceback.print_exc()
            return None
        



    
    def draw_basketball_court(self):
        """Draws a basketball court that dynamically fits the grid."""
        # --- NBA standard dimensions (meters) ---
        NBA_COURT_W = 28.65
        NBA_COURT_H = 15.24
        KEY_W = 4.88
        KEY_H = 5.79
        FT_CIRCLE_R = 1.8
        THREE_PT_R = 6.75
        HOOP_OFFSET = 1.575  # Distance from baseline to hoop center
        LINE_THICKNESS = 3

        # --- Dynamic Court Rectangle ---
        margin_meters = 2.0
        court_start_x = self.grid_offset_x + margin_meters * self.meters_to_pixels
        court_start_y = self.grid_offset_y + margin_meters * self.meters_to_pixels
        court_width = (self.world_width_meters - 2 * margin_meters) * self.meters_to_pixels
        court_height = (self.world_height_meters - 2 * margin_meters) * self.meters_to_pixels
        if court_width <= 0 or court_height <= 0: return
        court_rect = pygame.Rect(court_start_x, court_start_y, court_width, court_height)

        # --- Calculate scale: pixels per meter for this court rectangle ---
        px_per_meter_x = court_width / NBA_COURT_W
        px_per_meter_y = court_height / NBA_COURT_H
        px_per_meter = min(px_per_meter_x, px_per_meter_y)  # Use min to avoid overflow

        # --- Draw World Boundary ---
        # Draw world boundary (full world, not just court)
        world_rect = pygame.Rect(
            self.grid_offset_x,
            self.grid_offset_y,
            self.world_width_meters * self.meters_to_pixels,
            self.world_height_meters * self.meters_to_pixels
        )
        pygame.draw.rect(self.screen, (255, 255, 255), world_rect, 4)  # Yellow, 4px thick

        # --- Draw Court Background ---
        pygame.draw.rect(self.screen, (205, 133, 63), court_rect)
        playable_rect = court_rect.inflate(-LINE_THICKNESS * 2, -LINE_THICKNESS * 2)
        pygame.draw.rect(self.screen, (20, 40, 80), playable_rect)

        # --- Draw Court Markings ---
        center_x, center_y = court_rect.centerx, court_rect.centery

        # Center line
        pygame.draw.line(self.screen, (255, 255, 255), (center_x, court_rect.top), (center_x, court_rect.bottom), LINE_THICKNESS)

        # Center circle (3 meter radius in basketball)
        center_circle_radius = int(3.0 * px_per_meter)
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), center_circle_radius, LINE_THICKNESS)
        pygame.draw.circle(self.screen, (139, 0, 0), (center_x, center_y), center_circle_radius - LINE_THICKNESS)

        # --- Draw Features for Both Halves ---
        for side in [-1, 1]:  # -1 for left side, 1 for right side
            # Key (the "paint") - 5.8m x 4.9m in real basketball
            key_w = KEY_W * px_per_meter
            key_h = KEY_H * px_per_meter
            key_x = court_rect.left if side == -1 else court_rect.right - key_w
            key_y = center_y - key_h / 2
            key_rect = pygame.Rect(key_x, key_y, key_w, key_h)
            pygame.draw.rect(self.screen, (139, 0, 0), key_rect)

            # Free-throw circle (1.8m radius)
            ft_circle_radius = int(FT_CIRCLE_R * px_per_meter)
            ft_center_x = key_x + key_w if side == -1 else key_x
            arc_rect = pygame.Rect(ft_center_x - ft_circle_radius, center_y - ft_circle_radius, ft_circle_radius * 2, ft_circle_radius * 2)
            
            start_angle = -np.pi / 2 if side == -1 else np.pi / 2
            end_angle = np.pi / 2 if side == -1 else 3 * np.pi / 2
            pygame.draw.arc(self.screen, (255, 255, 255), arc_rect, start_angle, end_angle, LINE_THICKNESS)

            # Three-point line (6.75m from basket in NBA)
            hoop_center_x = court_rect.left + HOOP_OFFSET * px_per_meter if side == -1 else court_rect.right - HOOP_OFFSET * px_per_meter
            
            three_pt_radius = int(THREE_PT_R * px_per_meter)
            three_pt_arc_rect = pygame.Rect(hoop_center_x - three_pt_radius, center_y - three_pt_radius, three_pt_radius * 2, three_pt_radius * 2)
            pygame.draw.arc(self.screen, (255, 255, 255), three_pt_arc_rect, start_angle, end_angle, LINE_THICKNESS)
    


    def draw_simulation_data(self, data):
        """Draw whatever data your simulation produces"""
        if data is None:
            return
            
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_basketball_court()
        # self.draw_grid_boundaries()  # Removed: no more grid lines
        
        # Display info text
        y_offset = 20
        info_texts = [
            f"Madrona Basketball Simulation - Step {self.step_count}",
            f"World Size: {self.world_width_meters:.1f}x{self.world_height_meters:.1f} meters",
            "",
            "Controls: WASD/Arrow Keys, SPACE=manual step, R=reset, ESC=quit"
        ]

        # Add action debugging for enhanced actions
        if 'actions' in data:
            actions = data['actions'][0]  # Get first world, shape: (num_agents, N)
            for i, action_components in enumerate(actions):
                if len(action_components) >= 8:
                    move_speed, move_angle, rotate, grab, pass_ball, shoot_ball, steal, contest = action_components[:8]
                    info_texts.append(f"Agent {i}: Speed={int(move_speed)} Angle={int(move_angle)} Rotate={int(rotate)} Grab={int(grab)} Pass={int(pass_ball)} Shoot={int(shoot_ball)} Steal={int(steal)} Contest={int(contest)}")
                elif len(action_components) >= 6:
                    move_speed, move_angle, rotate, grab, pass_ball, shoot_ball = action_components[:6]
                    info_texts.append(f"Agent {i}: Speed={int(move_speed)} Angle={int(move_angle)} Rotate={int(rotate)} Grab={int(grab)} Pass={int(pass_ball)} Shoot={int(shoot_ball)}")
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
        


        # Draw basketball positions
        if 'basketball_pos' in data:
            raw_positions = data['basketball_pos']  # Shape: (1, num_basketballs, 3) - x,y,z
            positions = raw_positions[0]  # Get first world, shape: (num_basketballs, 3)

            # Draw basketball positions with different colors
            colors = [(255, 100, 0), (255, 200, 0), (100, 255, 0)]  # Orange, Yellow, Green
            
            for i, pos in enumerate(positions):
                if len(pos) >= 2:  # Use x,y from the 3-component position (x,y,z)
                    screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                    color = colors[i % len(colors)]
                    pygame.draw.circle(self.screen, color, (screen_x, screen_y), 12)
                    # Add basketball pattern
                    pygame.draw.circle(self.screen, (200, 50, 0), (screen_x, screen_y), 12, 2)
                    
                    # Add a number to identify each basketball
                    font_small = pygame.font.Font(None, 16)
                    text_surface = font_small.render(f"B{i + 1}", True, (255, 255, 255))
                    self.screen.blit(text_surface, (screen_x - 8, screen_y - 5))

        # Draw hoop positions
        if 'hoop_pos' in data:
            raw_positions = data['hoop_pos']  # Shape: (1, num_hoops, 3) - x,y,z
            positions = raw_positions[0]  # Get first world, shape: (num_hoops, 3)
            
            for i, pos in enumerate(positions):
                if len(pos) >= 2:  # Use x,y from the 3-component position (x,y,z)
                    screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                    
                    # Draw backboard (vertical line behind hoop)
                    backboard_color = (255, 255, 255)  # White backboard
                    pygame.draw.line(self.screen, backboard_color, 
                                (screen_x - 2, screen_y - 15), 
                                (screen_x - 2, screen_y + 15), 4)
                    
                    # Draw hoop base (orange circle)
                    hoop_base_color = (255, 140, 0)  # Orange hoop base
                    pygame.draw.circle(self.screen, hoop_base_color, (screen_x, screen_y), 10, 0)
                    
                    # Draw red rim (circle outline)
                    rim_color = (220, 20, 20)  # Red rim
                    pygame.draw.circle(self.screen, rim_color, (screen_x, screen_y), 10, 3)
                    
                    # Add hoop number/label
                    font_small = pygame.font.Font(None, 16)
                    text_surface = font_small.render(f"H{i + 1}", True, (255, 255, 255))
                    self.screen.blit(text_surface, (screen_x - 8, screen_y + 15))

        # Draw agent positions  
        if 'observations' in data and 'agent_teams' in data and 'orientation' in data:
            raw_positions = data['observations']  # Shape: (1, num_agents, 3) - x,y,z
            team_data = data['agent_teams'][0]  # Shape: (num_agents, 4) - teamIndex, colorR, colorG, colorB
            positions = raw_positions[0]  # Get first world, shape: (num_agents, 3)
            orientations = data['orientation'][0]  # (num_agents, 4) [w, x, y, z]
            
            for i, pos in enumerate(positions):
                if len(pos) >= 2:  # Use x,y from the 3-component position (x,y,z)
                    screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                    
                    # Extract team color from team data (teamIndex, then RGB components)
                    if i < len(team_data) and len(team_data[i]) >= 4:
                        team_index, color_r_raw, color_g_raw, color_b_raw = team_data[i]
                        
                        # The color values are stored as int32, but they're actually float32 bit patterns
                        # Convert them back to floats first
                        import struct
                        color_r_float = struct.unpack('f', struct.pack('I', np.uint32(color_r_raw)))[0]
                        color_g_float = struct.unpack('f', struct.pack('I', np.uint32(color_g_raw)))[0]
                        color_b_float = struct.unpack('f', struct.pack('I', np.uint32(color_b_raw)))[0]
                        
                        # Ensure color values are valid integers in range [0, 255]
                        color_r = max(0, min(255, int(color_r_float)))
                        color_g = max(0, min(255, int(color_g_float)))
                        color_b = max(0, min(255, int(color_b_float))) 
                        agent_color = (color_r, color_g, color_b)
                    else:
                        # Fallback colors if team data is missing or malformed
                        fallback_colors = [(0, 100, 255), (255, 0, 100)]  # Blue, Pink
                        agent_color = fallback_colors[i % len(fallback_colors)]
                    
                    # Draw agents as rectangles to distinguish from circular basketballs
                    pygame.draw.rect(self.screen, agent_color, (screen_x - 8, screen_y - 8, 16, 16))
                    pygame.draw.rect(self.screen, (255, 255, 255), (screen_x - 8, screen_y - 8, 16, 16), 2)
                    
                    # Add agent number and team info
                    font_small = pygame.font.Font(None, 16)
                    text_surface = font_small.render(f"A{i + 1}", True, (255, 255, 255))
                    self.screen.blit(text_surface, (screen_x - 8, screen_y - 20))
                    
                    # Add team index below agent
                    if i < len(team_data):
                        team_text = font_small.render(f"T{int(team_data[i][0])}", True, (255, 255, 255))
                        self.screen.blit(team_text, (screen_x - 8, screen_y + 10))
                    


                    # Drawing the yellow orientation line
                    # Get the current agent's orientation quaternion from the data
                    q = orientations[i]
                    
                    # Define the single, consistent "forward" vector for the agent model.
                    # This MUST match the vector used in your C++ passSystem: Vector3{1, 0, 0}
                    BASE_FORWARD_VECTOR = np.array([0.0, 2.0, 0.0])

                    # Calculate the current direction vector by rotating the base vector
                    # using the function you provided.
                    direction_3d = rotate_vec(q, BASE_FORWARD_VECTOR)
                    
                    # We only need the x and y components for our 2D visualization
                    dx = direction_3d[0]
                    dy = direction_3d[1]
                    
                    # Set the screen length of the orientation line (0.8 meters)
                    arrow_length_meters = 0.8
                    arrow_length_pixels = arrow_length_meters * self.meters_to_pixels
                    
                    # Calculate the end point of the orientation line
                    arrow_end = (screen_x + arrow_length_pixels * dx,
                                 screen_y + arrow_length_pixels * dy)
                    
                    # Draw the yellow line that now correctly represents the agent's orientation
                    pygame.draw.line(self.screen, (255, 255, 0), (screen_x, screen_y), arrow_end, 3)


                    


        if 'ball_physics' in data:
            physics_data = data['ball_physics'][0]
            for i, physics in enumerate(physics_data):
                if len(physics) >= 5:
                    in_flight, vel_x, vel_y, vel_z, lastTouchedByID = physics
                    info_texts.append(f"Ball {i+1}: Flight={bool(in_flight)} Vel=({vel_x:.1f},{vel_y:.1f},{vel_z:.1f}) Last Touched By ID={int(lastTouchedByID)}")

        # Debug: Display ball grabbed status with detailed info
        if 'ball_grabbed' in data:
            grabbed_data = data['ball_grabbed'][0]  # Shape: (num_basketballs, 2)
            for i, grabbed in enumerate(grabbed_data):
                if len(grabbed) >= 2:
                    is_grabbed, holder_entity_id = grabbed
                    info_texts.append(f"Ball {i+1}: isGrabbed={bool(is_grabbed)} holderID={int(holder_entity_id)}")
        
        # Debug: Display agent possession status
        if 'agent_possession' in data:
            possession_data = data['agent_possession'][0]  # Shape: (num_agents, 2)
            for i, possession in enumerate(possession_data):
                if len(possession) >= 2:
                    has_ball, ball_entity_id = possession
                    info_texts.append(f"Agent {i}: hasBall={bool(has_ball)} ballEntityID={int(ball_entity_id)}")
        
        # Debug: Show what we think the entity IDs are
        if 'agent_entity_ids' in data:
            agent_ids = data['agent_entity_ids'][0]
            info_texts.append(f"Agent Entity IDs: {[int(x) for x in agent_ids]}")
        
        if 'ball_entity_ids' in data:
            ball_ids = data['ball_entity_ids'][0]
            info_texts.append(f"Ball Entity IDs: {[int(x) for x in ball_ids]}")

        for text in info_texts:
            if text:
                surface = self.font.render(text, True, TEXT_COLOR)
                self.screen.blit(surface, (20, y_offset))
            y_offset += 20

    
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
        pass_ball = 0
        shoot_ball = 0
        
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
        if keys[pygame.K_LSHIFT]:
            grab = 1


        if keys[pygame.K_SPACE]:
            pass_ball = 1

        if keys[pygame.K_h]:
            shoot_ball = 1
        
        
        # Agent 1 actions (Arrow keys + other keys)
        move_speed1 = 0
        move_angle1 = 0  
        rotate1 = 0
        grab1 = 0
        pass_ball1 = 0
        shoot_ball1 = 0
        
        if keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
            move_speed1 = 1  # ← Fix: use move_speed1, not move_speed
            
            # 8-directional movement with corrected key mappings
            if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:          # NE
                move_angle1 = 1
            elif keys[pygame.K_RIGHT] and keys[pygame.K_DOWN]:      # SE  
                move_angle1 = 3
            elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:       # SW ← Fix: DOWN + LEFT
                move_angle1 = 5
            elif keys[pygame.K_LEFT] and keys[pygame.K_UP]:         # NW ← Fix: LEFT + UP
                move_angle1 = 7
            elif keys[pygame.K_UP]:                                 # N
                move_angle1 = 0
            elif keys[pygame.K_RIGHT]:                              # E
                move_angle1 = 2
            elif keys[pygame.K_DOWN]:                               # S
                move_angle1 = 4
            elif keys[pygame.K_LEFT]:                               # W ← Fix: LEFT key
                move_angle1 = 6
        
        # Agent 1 rotation and grab (different keys to avoid conflicts)
        if keys[pygame.K_COMMA]:      # ',' key for Agent 1 turn left
            rotate1 = -1
        elif keys[pygame.K_PERIOD]:   # '.' key for Agent 1 turn right
            rotate1 = 1
        
        if keys[pygame.K_KP0]:        # Numpad 0 for Agent 1 grab
            grab1 = 1

        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:  # Left or Right Ctrl for Agent 1 pass
            pass_ball1 = 1

        
        # Send enhanced actions to simulation
        self.sim.set_action(0, 0, move_speed, move_angle, rotate, grab, pass_ball, shoot_ball)
        self.sim.set_action(0, 1, move_speed1, move_angle1, rotate1, grab1, pass_ball1, shoot_ball1)

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

    def meters_to_screen(self, meter_x, meter_y):
        """Convert meter coordinates to screen pixels"""
        screen_x = self.grid_offset_x + (meter_x * self.meters_to_pixels)
        screen_y = self.grid_offset_y + (meter_y * self.meters_to_pixels)
        return int(screen_x), int(screen_y)

if __name__ == "__main__":
    try:
        pipeline = MadronaPipeline()
        pipeline.run()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
