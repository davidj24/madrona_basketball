#!/usr/bin/env python3
"""
Simple Pipeline: Madrona C++ Simulation → Pygame Visualization
This connects to whatever simulation you build in Madrona and displays it
"""

import pygame
import sys
import numpy as np
import os
import math
import argparse

# Import constants
from src.constants import *

# CRITICAL: Set environment variables before any CUDA/OpenGL operations
import os
# Prevent CUDA from initializing OpenGL context that conflicts with pygame
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
# Force CUDA to use a specific device context
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# Note: CUDA_LAUNCH_BLOCKING=1 causes very slow GPU compilation, so we only set it during tensor operations

# Try to import and initialize PyTorch early to avoid issues later
try:
    import torch
    torch.set_num_threads(1)  # Limit CPU threads
    print("✓ PyTorch imported successfully")
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"⚠ PyTorch import issue: {e}")
    torch = None
    TORCH_AVAILABLE = False  # Fallback for systems without torch

# Add build directory to path for the C++ module
sys.path.append('./build')

try:
    import madrona_basketball as mba
    from madrona_basketball.madrona import ExecMode
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
        np.array: The ro    tated 3D vector.
    """
    scalar = q[0]
    pure = np.array([q[1], q[2], q[3]])
    v_vec = np.asarray(v)
    pure_x_v = np.cross(pure, v_vec)
    pure_x_pure_x_v = np.cross(pure, pure_x_v)
    return v_vec + 2.0 * ((pure_x_v * scalar) + pure_x_pure_x_v)

class ViewerClass:
    """
    Simple pipeline that connects to your Madrona simulation and displays the data
    """
    
    def handle_audio_events(self, data):
        """Checks for new audio events from the simulation and plays sounds."""
        if data is None or 'game_state' not in data:
            return

        world_idx = min(self.debug_world_index, len(data['game_state']) - 1) if hasattr(self, 'debug_world_index') else 0
        game_state = data['game_state'][world_idx]
        
        # We assume scoreCount is the 11th field (index 10)
        # and outOfBoundsCount is the 12th field (index 11).
        # Adjust these indices if struct is different.
        current_score_count = int(game_state[10])
        current_oob_count = int(game_state[11])

        # Check if the score count has increased since the last frame
        if self.score_sound and current_score_count > self.last_score_count:
            self.score_sound.play()
        
        # Check if the out-of-bounds count has increased
        if self.whistle_sound and current_oob_count > self.last_oob_count:
            self.whistle_sound.play()

        # Update the last known counts for the next frame
        self.last_score_count = current_score_count
        self.last_oob_count = current_oob_count

    def __init__(self, sim_instance=None, training_mode=False):
        print("🔧 Initializing viewer with GPU safety measures...")
        
        # Store simulation instance but don't access it
        self.sim = sim_instance
        self.simulation_ready = False  # Track if simulation is ready for data access
        self.ready_check_attempts = 0  # Count attempts to check if simulation is ready
        
        self.disable_action_input = False  # Allow interactive input for human control
        self.training_mode = training_mode
        if training_mode:
            print("✓ Training mode detected - interactive human control enabled")
        
        # Interactive training support
        self.controller_manager = None
        self.training_paused = False
        self.selected_world = 0
        self.selected_agent = 0
        # Initialize human action 
        if TORCH_AVAILABLE and torch is not None:
            self.human_action = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int32)
        else:
            self.human_action = [0, 0, 0, 0, 0, 0]  # Fallback to list
        self.action_changed = False
        
        # Add a flag to detect if we're running on GPU
        self.is_gpu_simulation = False
        self.gpu_device = None
        
        self.debug_world_index = 0  # Default to world 0
        self.max_worlds_available = 1  # Will be updated when data is available
        
        # Initialize pygame with extensive error handling and CUDA isolation
        try:
            # Force CUDA to not interfere with graphics
            import os
            os.environ['SDL_VIDEODRIVER'] = 'x11'  # Force specific video driver
            os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')  # Ensure display is set
            
            # Initialize pygame
            pygame.init()
            print("✓ Pygame core initialized")
            
            # Initialize mixer separately
            try:
                pygame.mixer.init()
                print("✓ Pygame mixer initialized")
            except pygame.error as mixer_e:
                print(f"⚠ Pygame mixer failed: {mixer_e}, continuing without audio")
            
        except Exception as e:
            print(f"✗ Pygame initialization failed: {e}")
            raise RuntimeError(f"Cannot initialize pygame: {e}")
        
        # Create display with error handling
        try:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Basketball Simulation")
            print("✓ Pygame display created")
        except Exception as e:
            print(f"✗ Pygame display creation failed: {e}")
            raise RuntimeError(f"Cannot create pygame display: {e}")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.active_agent_idx = 0

        try:
            pygame.mixer.init()  # Initialize the audio mixer
            self.score_sound = pygame.mixer.Sound("assets/swish.wav")
            self.whistle_sound = pygame.mixer.Sound("assets/whistle.wav")
            print("✓ Audio files loaded successfully.")
        except pygame.error as e:
            print(f"⚠ Warning: Could not initialize audio. Running in silent mode. Error: {e}")
            self.score_sound = None
            self.whistle_sound = None


        self.world_width_meters = WORLD_WIDTH_M
        self.world_height_meters = WORLD_HEIGHT_M
        self.pixels_per_meter = PIXELS_PER_METER

        self.world_width_px = self.world_width_meters * PIXELS_PER_METER
        self.world_height_px = self.world_height_meters * PIXELS_PER_METER

        self.world_offset_x = (WINDOW_WIDTH - self.world_width_px) / 2
        self.world_offset_y = (WINDOW_HEIGHT - self.world_height_px) / 2
        
        print("Madrona simulation initialized!")
        print(f"✓ World is {self.world_width_meters:.2f}m x {self.world_height_meters:.2f}m")
        
        self.step_count = 0
        self.last_score_count = 0
        self.last_oob_count = 0
        
    def check_simulation_status(self):
        """Check simulation status and print device information on first successful access"""
        try:
            # Get tensor shapes to determine number of environments
            obs_tensor = self.sim.observations_tensor()
            obs_torch = obs_tensor.to_torch()
            num_envs = obs_torch.shape[0] if len(obs_torch.shape) > 0 else 1
            
            if num_envs > 1000:
                print(f"⚠ Warning: Running viewer with {num_envs} environments")
                print(f"   Viewer will only display data from the first environment to prevent memory issues")
                
            # Check if running on GPU and store the information
            device = obs_torch.device
            if device.type == 'cuda':
                self.is_gpu_simulation = True
                self.gpu_device = device
                print(f"✓ GPU simulation detected (device: {device})")
                print(f"  Viewer memory optimization enabled for {num_envs} environments")
                
                # Additional GPU safety measures
                try:
                    import torch
                    torch.cuda.synchronize()
                    print("✓ CUDA context verified and synchronized")
                except Exception as e:
                    print(f"⚠ CUDA verification warning: {e}")
            else:
                self.is_gpu_simulation = False
                print(f"✓ CPU simulation detected (device: {device})")
                
            return True
                
        except Exception as e:
            print(f"⚠ Could not determine simulation configuration: {e}")
            return False

    def get_simulation_data(self):
        """Get the current state from your Madrona simulation"""
        # Check if simulation is ready - reduce wait time to get data faster
        if not self.simulation_ready:
            self.ready_check_attempts += 1
            if self.ready_check_attempts < 3:  # Reduced from 10 to 3 for faster startup
                return None
            else:
                # Try to check simulation status once
                if self.check_simulation_status():
                    self.simulation_ready = True
                    print("✓ Simulation data access ready!")
                else:
                    # Reset counter to try again later
                    self.ready_check_attempts = 0
                    return None
        
        try:
            def safe_tensor_to_numpy(tensor_torch):
                """Safely convert tensor to numpy with GPU handling"""
                if tensor_torch is None:
                    return None
                try:
                    if hasattr(tensor_torch, 'is_cuda') and tensor_torch.is_cuda:
                        return tensor_torch.detach().cpu().numpy()
                    else:
                        return tensor_torch.detach().numpy()
                except Exception as e:
                    print(f"Warning: Tensor conversion failed: {e}")
                    return None
            
            # Access all the data that the viewer needs to actually display something
            try:
                # Basic training data (safe)
                observations = safe_tensor_to_numpy(self.sim.observations_tensor().to_torch())
                actions = safe_tensor_to_numpy(self.sim.action_tensor().to_torch())
                rewards = safe_tensor_to_numpy(self.sim.reward_tensor().to_torch())
                done = safe_tensor_to_numpy(self.sim.done_tensor().to_torch())
                reset = safe_tensor_to_numpy(self.sim.reset_tensor().to_torch())
                
                # Essential viewer data (needed for visualization)
                agent_pos = safe_tensor_to_numpy(self.sim.agent_pos_tensor().to_torch())
                agent_teams = safe_tensor_to_numpy(self.sim.agent_team_tensor().to_torch())
                basketball_pos = safe_tensor_to_numpy(self.sim.basketball_pos_tensor().to_torch())
                hoop_pos = safe_tensor_to_numpy(self.sim.hoop_pos_tensor().to_torch())
                orientation = safe_tensor_to_numpy(self.sim.orientation_tensor().to_torch())
                
                # Optional data (with fallback to None if not available)
                try:
                    ball_physics = safe_tensor_to_numpy(self.sim.ball_physics_tensor().to_torch())
                except:
                    ball_physics = None
                    
                try:
                    agent_possession = safe_tensor_to_numpy(self.sim.agent_possession_tensor().to_torch())
                except:
                    agent_possession = None
                    
                try:
                    ball_grabbed = safe_tensor_to_numpy(self.sim.ball_grabbed_tensor().to_torch())
                except:
                    ball_grabbed = None
                    
                try:
                    game_state_tensor = self.sim.game_state_tensor().to_torch()
                    game_state = safe_tensor_to_numpy(game_state_tensor)
                except Exception as e:
                    print(f"Warning: Could not access game_state_tensor: {e}")
                    game_state = None
                    
                try:
                    agent_entity_ids = safe_tensor_to_numpy(self.sim.agent_entity_id_tensor().to_torch())
                except:
                    agent_entity_ids = None
                    
                try:
                    ball_entity_ids = safe_tensor_to_numpy(self.sim.ball_entity_id_tensor().to_torch())
                except:
                    ball_entity_ids = None
                    
                try:
                    agent_stats = safe_tensor_to_numpy(self.sim.agent_stats_tensor().to_torch())
                except:
                    agent_stats = None
                
            except Exception as e:
                print(f"Warning: Tensor access failed: {e}")
                return None
            
            # Return the complete data needed for visualization
            result = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'done': done,
                'reset': reset,
                'agent_pos': agent_pos,
                'agent_teams': agent_teams,
                'basketball_pos': basketball_pos,
                'ball_physics': ball_physics,
                'hoop_pos': hoop_pos,
                'agent_possession': agent_possession,
                'ball_grabbed': ball_grabbed,
                'agent_entity_ids': agent_entity_ids,
                'ball_entity_ids': ball_entity_ids,
                'game_state': game_state,
                'orientation': orientation,
                'agent_stats': agent_stats
            }
            

            
            return result
            
        except Exception as e:
            print(f"Error getting simulation data: {e}")
            return None

    def draw_basketball_court(self):
        """
        Draws an accurate, regulation NBA court, with corrected 3-point line geometry
        that perfectly matches the C++ game logic.
        """
        # 1. Define Colors and Dimensions (matching the C++ "Blueprint")
        COURT_COLOR = (0, 0, 0)
        PAINT_RED = (120, 20, 20)
        LINE_WHITE = (255, 255, 255)
        ORANGE_BORDER = (255, 255, 255)
        
        # 2. Scaling and Positioning
        scale = self.pixels_per_meter
        line_thickness = max(2, int(scale * 0.05))

        court_w_px = COURT_LENGTH_M * scale
        court_h_px = COURT_WIDTH_M * scale
        court_offset_x = self.world_offset_x + (self.world_width_px - court_w_px) / 2
        court_offset_y = self.world_offset_y + (self.world_height_px - court_h_px) / 2
        court_rect = pygame.Rect(court_offset_x, court_offset_y, court_w_px, court_h_px)
        center_x, center_y = court_rect.centerx, court_rect.centery

        # 3. Draw Floor and Borders
        pygame.draw.rect(self.screen, ORANGE_BORDER, court_rect.inflate(line_thickness, line_thickness))
        pygame.draw.rect(self.screen, COURT_COLOR, court_rect)

        # 4. Draw Features for Both Halves
        for side in [-1, 1]:
            hoop_x = court_rect.left + HOOP_FROM_BASELINE_M * scale if side == -1 else court_rect.right - HOOP_FROM_BASELINE_M * scale

            # Key and Free-throw circle
            key_w_px = KEY_HEIGHT_M * scale
            key_h_px = KEY_WIDTH_M * scale
            key_x = court_rect.left if side == -1 else court_rect.right - key_w_px
            key_rect = pygame.Rect(key_x, center_y - key_h_px / 2, key_w_px, key_h_px)
            pygame.draw.rect(self.screen, PAINT_RED, key_rect)
            pygame.draw.rect(self.screen, LINE_WHITE, key_rect, line_thickness)
            ft_radius_px = FREE_THROW_CIRCLE_RADIUS_M * scale
            ft_center_x = court_rect.left + KEY_HEIGHT_M * scale if side == -1 else court_rect.right - KEY_HEIGHT_M * scale
            ft_arc_rect = pygame.Rect(ft_center_x - ft_radius_px, center_y - ft_radius_px, ft_radius_px * 2, ft_radius_px * 2)
            start_angle, end_angle = (-np.pi / 2, np.pi / 2) if side == -1 else (np.pi / 2, 3 * np.pi / 2)
            pygame.draw.arc(self.screen, LINE_WHITE, ft_arc_rect, start_angle, end_angle, line_thickness)
            
            # --- Corrected 3-Point Line Drawing ---
            three_pt_radius_px = ARC_RADIUS_M * scale
            
            # A) Corner straight lines
            corner_y_offset_px = (COURT_WIDTH_M / 2 - CORNER_3_FROM_SIDELINE_M) * scale
            y1 = center_y - corner_y_offset_px
            y2 = center_y + corner_y_offset_px
            
            x_line_start = court_rect.left if side == -1 else court_rect.right
            x_line_end = court_rect.left + CORNER_3_LENGTH_FROM_BASELINE_M * scale if side == -1 else court_rect.right - CORNER_3_LENGTH_FROM_BASELINE_M * scale
            
            pygame.draw.line(self.screen, LINE_WHITE, (x_line_start, y1), (x_line_end, y1), line_thickness)
            pygame.draw.line(self.screen, LINE_WHITE, (x_line_start, y2), (x_line_end, y2), line_thickness)

            # B) 3-Point Arc connecting the two straight lines
            arc_rect = pygame.Rect(hoop_x - three_pt_radius_px, center_y - three_pt_radius_px, three_pt_radius_px * 2, three_pt_radius_px * 2)
            
            # Simplified and corrected angle logic
            x_diff = abs(x_line_end - hoop_x)
            y_diff = abs(y1 - center_y)

            if three_pt_radius_px > 0:
                # This angle is now always a simple positive value in the first quadrant
                clip_angle = np.arctan2(y_diff, x_diff)

                if side == -1: # Left hoop (arc is on the right of the hoop)
                    # Starts from the bottom-right quadrant, goes to the top-right
                    start_angle = -clip_angle
                    end_angle = clip_angle
                else: # Right hoop (arc is on the left of the hoop)
                    # Starts from the top-left quadrant, goes to thebottom-left
                    start_angle = np.pi - clip_angle
                    end_angle = np.pi + clip_angle
                
                pygame.draw.arc(self.screen, LINE_WHITE, arc_rect, start_angle, end_angle, line_thickness)

        # 5. Center Line & Circle
        pygame.draw.line(self.screen, LINE_WHITE, (center_x, court_rect.top), (center_x, court_rect.bottom), line_thickness)
        center_circle_radius = FREE_THROW_CIRCLE_RADIUS_M * scale
        pygame.draw.circle(self.screen, LINE_WHITE, (center_x, center_y), center_circle_radius, line_thickness)

        # 6. World Border
        world_border_rect = pygame.Rect(self.world_offset_x, self.world_offset_y, self.world_width_px, self.world_height_px)
        pygame.draw.rect(self.screen, ORANGE_BORDER, world_border_rect, 3)

    def draw_inbound_clock(self, data):
        """Draws the inbound clock only when an inbound pass is in progress."""
        if data is None or 'game_state' not in data:
            return

        world_idx = min(self.debug_world_index, len(data['game_state']) - 1)
        game_state = data['game_state'][world_idx]
        # These indices MUST match your C++ GameState struct.
        # From your function, it looks like inboundingInProgress is at index 0
        # and inbound_clock is at index 12.
        inbounding_in_progress = game_state[0] > 0.5
        inbound_clock = float(game_state[12])

        # Only draw if inbounding is active
        if not inbounding_in_progress:
            return

        # 1. Define appearance and dimensions in meters for proper scaling
        box_width_meters = 2.5
        box_height_meters = 1.2
        box_vertical_offset_from_top_px = 20 # How far from the top of the window

        # 2. Scale dimensions to pixels using the pixels_per_meter variable
        box_width_px = int(box_width_meters * self.pixels_per_meter)
        box_height_px = int(box_height_meters * self.pixels_per_meter)

        # 3. Calculate position (centered horizontally, near the top)
        box_center_x = WINDOW_WIDTH // 2
        box_top_y = box_vertical_offset_from_top_px
        box_rect = pygame.Rect(
            box_center_x - box_width_px // 2,
            box_top_y,
            box_width_px,
            box_height_px
        )

        # 4. Draw the box
        pygame.draw.rect(self.screen, (0, 0, 0), box_rect) # Black background
        pygame.draw.rect(self.screen, (255, 255, 255), box_rect, 3) # White outline

        # 5. Draw the text, scaling the font size with the box size
        font_size = int(box_height_px * 0.8)
        inbound_font = pygame.font.Font(None, font_size)

        # Format the clock to show one decimal place
        clock_text = f"{inbound_clock:.1f}"
        text_surface = inbound_font.render(clock_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=box_rect.center)

        self.screen.blit(text_surface, text_rect)

    def draw_score_display(self, data):
        """Draw the team scores and game info at the bottom center of the screen"""
        if data is None or 'game_state' not in data:
            return

        world_idx = min(self.debug_world_index, len(data['game_state']) - 1)
        game_state = data['game_state'][world_idx]
        inboundingInProgress = float(game_state[0])
        period = int(game_state[2])
        team0_score = int(game_state[5])
        team1_score = int(game_state[7])
        game_clock = float(game_state[8])
        shot_clock = float(game_state[9])
        inbound_clock = float(game_state[12])

        # Use team color constants
        team_colors = { 0: TEAM0_COLOR, 1: TEAM1_COLOR }
        if 'agent_teams' in data:
            agent_teams = data['agent_teams'][world_idx]
            for i, team_data in enumerate(agent_teams):
                if len(team_data) >= 4:
                    team_index = int(team_data[0])
                    if team_index not in team_colors:
                        # Fallback: use constants
                        team_colors[team_index] = TEAM0_COLOR if team_index == 0 else TEAM1_COLOR

        # Score display dimensions and positioning
        display_width = 600
        display_height = 120
        display_x = (WINDOW_WIDTH - display_width) // 2
        display_y = WINDOW_HEIGHT - display_height - 10

        # Draw main score display background 
        score_bg_rect = pygame.Rect(display_x, display_y, display_width, display_height)
        pygame.draw.rect(self.screen, (30, 30, 30), score_bg_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), score_bg_rect, 3)

        # Team score sections 
        team_section_width = display_width // 3
        team0_rect = pygame.Rect(display_x, display_y, team_section_width, display_height)
        pygame.draw.rect(self.screen, team_colors.get(0, (0,100,255)), team0_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), team0_rect, 2)
        team1_rect = pygame.Rect(display_x + 2 * team_section_width, display_y, team_section_width, display_height)
        pygame.draw.rect(self.screen, team_colors.get(1, (255,0,100)), team1_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), team1_rect, 2)
        middle_rect = pygame.Rect(display_x + team_section_width, display_y, team_section_width, display_height)
        pygame.draw.rect(self.screen, (60, 60, 60), middle_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), middle_rect, 2)

        # Fonts 
        score_font = pygame.font.Font(None, 64)
        label_font = pygame.font.Font(None, 24)
        time_font = pygame.font.Font(None, 36)

        # Draw team scores 
        team0_score_text = score_font.render(str(team0_score), True, (255, 255, 255))
        team0_score_rect = team0_score_text.get_rect(center=(team0_rect.centerx, team0_rect.centery - 10))
        self.screen.blit(team0_score_text, team0_score_rect)
        team1_score_text = score_font.render(str(team1_score), True, (255, 255, 255))
        team1_score_rect = team1_score_text.get_rect(center=(team1_rect.centerx, team1_rect.centery - 10))
        self.screen.blit(team1_score_text, team1_score_rect)

        # Draw team labels 
        team0_label = label_font.render("TEAM 0", True, (255, 255, 255))
        team0_label_rect = team0_label.get_rect(center=(team0_rect.centerx, team0_rect.bottom - 15))
        self.screen.blit(team0_label, team0_label_rect)
        team1_label = label_font.render("TEAM 1", True, (255, 255, 255))
        team1_label_rect = team1_label.get_rect(center=(team1_rect.centerx, team1_rect.bottom - 15))
        self.screen.blit(team1_label, team1_label_rect)

        # Draw period, game clock, and shot clock 
        period_text = time_font.render(f"Q{period}", True, (255, 255, 255))
        period_rect = period_text.get_rect(center=(middle_rect.centerx, middle_rect.top + 25))
        self.screen.blit(period_text, period_rect)
        game_minutes = int(game_clock // 60)
        game_seconds = int(game_clock % 60)
        game_time_text = time_font.render(f"{game_minutes:02d}:{game_seconds:02d}", True, (255, 255, 255))
        game_time_rect = game_time_text.get_rect(center=(middle_rect.centerx, middle_rect.centery))
        self.screen.blit(game_time_text, game_time_rect)
        shot_clock_seconds = int(shot_clock)
        shot_clock_text = label_font.render(f"Shot: {shot_clock_seconds}", True, (255, 255, 0))
        shot_clock_rect = shot_clock_text.get_rect(center=(middle_rect.centerx, middle_rect.bottom - 20))
        self.screen.blit(shot_clock_text, shot_clock_rect)

    def draw_simulation_data(self, data):
        """Draw whatever data your simulation produces"""
        if data is None:
            # Draw a simple message if no data is available
            self.screen.fill(BACKGROUND_COLOR)
            error_text = self.font.render("Waiting for simulation data...", True, TEXT_COLOR)
            self.screen.blit(error_text, (10, 10))
            return

        self.screen.fill(BACKGROUND_COLOR)

        # Draw the main gameplay elements
        self.draw_basketball_court()
        self.draw_score_display(data)

        # Draw the new inbound clock (it will only appear when needed)
        self.draw_inbound_clock(data)

        # Draw world index indicator
        world_indicator_text = f"World {self.debug_world_index}/{self.max_worlds_available-1} (Press 1/2/3 to switch)"
        world_indicator_surface = self.font.render(world_indicator_text, True, (255, 255, 0))
        self.screen.blit(world_indicator_surface, (10, WINDOW_HEIGHT - 30))

        y_offset = 20
        info_texts = [f"Madrona Basketball Simulation - Step {self.step_count}"]

        # --- Info Text (Preserved from your file) ---
        if 'actions' in data and data['actions'] is not None:
            world_idx = min(self.debug_world_index, len(data['actions']) - 1)
            actions = data['actions'][world_idx]
            for i, action_components in enumerate(actions):
                if len(action_components) >= 8:
                    info_texts.append(f"Agent {i}: Speed={int(action_components[0])} Angle={int(action_components[1])} Rotate={int(action_components[2])} Grab={int(action_components[3])} Pass={int(action_components[4])} Shoot={int(action_components[5])} Steal={int(action_components[6])} Contest={int(action_components[7])}")
                elif len(action_components) >= 6:
                    info_texts.append(f"Agent {i}: Speed={int(action_components[0])} Angle={int(action_components[1])} Rotate={int(action_components[2])} Grab={int(action_components[3])} Pass={int(action_components[4])} Shoot={int(action_components[5])}")

        if 'rewards' in data and data['rewards'] is not None:
            world_idx = min(self.debug_world_index, len(data['rewards']) - 1)
            for i, reward in enumerate(data['rewards'][world_idx]): info_texts.append(f"Agent {i} Reward: {reward:.2f}")
        if 'done' in data and data['done'] is not None:
            world_idx = min(self.debug_world_index, len(data['done']) - 1)
            for i, done in enumerate(data['done'][world_idx]): info_texts.append(f"Agent {i} Done: {done}")

        # --- Agent Rendering (Corrected Colors and Text) ---
        if ('agent_pos' in data and data['agent_pos'] is not None and
            'agent_teams' in data and data['agent_teams'] is not None and
            'orientation' in data and data['orientation'] is not None):
            # Update max worlds available for debugging
            self.max_worlds_available = len(data['agent_pos'])

            # Use debug world index (can be changed for testing)
            world_idx = min(self.debug_world_index, len(data['agent_pos']) - 1)
            positions, team_data, orientations = data['agent_pos'][world_idx], data['agent_teams'][world_idx], data['orientation'][world_idx]

            # Use team color constants (Assuming TEAM0_COLOR and TEAM1_COLOR are defined in constants.py)
            for i, pos in enumerate(positions):
                screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                team_index = int(team_data[i][0]) if i < len(team_data) else 0
                agent_color = TEAM0_COLOR if team_index == 0 else TEAM1_COLOR

                # Get orientation to calculate rectangle vertices
                q = orientations[i]
                forward_vec_3d = rotate_vec(q, np.array([0.0, 1.0, 0.0]))

                # We only need the 2D projection for drawing
                forward_vec = np.array([forward_vec_3d[0], forward_vec_3d[1]])
                # The "right" vector is perpendicular to the forward vector
                right_vec = np.array([forward_vec[1], -forward_vec[0]])

                # Get dimensions in pixels
                shoulder_width_px = AGENT_SHOULDER_WIDTH * self.pixels_per_meter
                depth_px = AGENT_DEPTH * self.pixels_per_meter

                # Calculate the four corner points of the rectangle
                center_point = np.array([screen_x, screen_y])

                half_width_vec = right_vec * (shoulder_width_px / 2)
                half_depth_vec = forward_vec * (depth_px / 2)

                p1 = center_point - half_depth_vec + half_width_vec # Front-right
                p2 = center_point - half_depth_vec - half_width_vec # Front-left
                p3 = center_point + half_depth_vec - half_width_vec # Back-left
                p4 = center_point + half_depth_vec + half_width_vec # Back-right

                agent_points = [p1, p2, p3, p4]

                # Draw the agent as a polygon
                pygame.draw.polygon(self.screen, agent_color, agent_points)

                # Draw a special highlight for the active agent
                if i == self.active_agent_idx:
                    pygame.draw.polygon(self.screen, (0, 255, 255), agent_points, 3) # Bright yellow outline
                else:
                    pygame.draw.polygon(self.screen, (255, 255, 255), agent_points, 1) # Standard white outline

                # Draw the agent's number (not team ID) inside the rectangle
                # Using a smaller fixed font size to ensure it fits
                font_small = pygame.font.Font(None, 20)
                text_surface = font_small.render(str(i), True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(screen_x, screen_y))
                self.screen.blit(text_surface, text_rect)

                # Draw orientation line (still useful for debugging)
                arrow_len_px = self.pixels_per_meter * AGENT_ORIENTATION_ARROW_LENGTH_M
                arrow_end = (screen_x + arrow_len_px * forward_vec[0], screen_y + arrow_len_px * forward_vec[1])
                pygame.draw.line(self.screen, (255, 255, 0), (screen_x, screen_y), arrow_end, 3)

        # --- Basketball Rendering (Scaled to Real Size) ---
        if 'basketball_pos' in data and data['basketball_pos'] is not None:
            world_idx = min(self.debug_world_index, len(data['basketball_pos']) - 1)
            basketball_positions = data['basketball_pos'][world_idx]  # Extract basketball positions from debug world

            # Use real basketball dimensions from constants
            ball_radius_px = BALL_RADIUS_M * self.pixels_per_meter
            for i, pos in enumerate(basketball_positions):
                screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                # Draw filled orange ball
                pygame.draw.circle(self.screen, (255, 100, 0), (screen_x, screen_y), int(ball_radius_px))
                # Draw a darker orange outline
                pygame.draw.circle(self.screen, (200, 50, 0), (screen_x, screen_y), int(ball_radius_px), max(2, int(ball_radius_px * 0.15)))
                # Ball label
                font_ball = pygame.font.Font(None, max(12, int(ball_radius_px * 0.7)))
                label_surface = font_ball.render(f"B{i + 1}", True, (255, 255, 255))
                label_rect = label_surface.get_rect(center=(screen_x, screen_y))
                self.screen.blit(label_surface, label_rect)

        # --- Hoop Rendering ---
        if 'hoop_pos' in data:
            world_idx = min(self.debug_world_index, len(data['hoop_pos']) - 1)
            hoop_positions = data['hoop_pos'][world_idx]  # Extract hoop positions from debug world

            for i, pos in enumerate(hoop_positions):
                screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                backboard_width_px = BACKBOARD_WIDTH_M * self.pixels_per_meter
                rim_radius_px = (RIM_DIAMETER_M / 2) * self.pixels_per_meter
                rim_thickness_px = max(2, int(self.pixels_per_meter * 0.02))
                backboard_thickness_px = max(3, int(self.pixels_per_meter * 0.05))
                backboard_offset_px = BACKBOARD_OFFSET_FROM_HOOP_M * self.pixels_per_meter
                backboard_x = screen_x - backboard_offset_px if pos[0] < self.world_width_meters / 2 else screen_x + backboard_offset_px
                pygame.draw.line(self.screen, (255, 255, 255), (backboard_x, screen_y - backboard_width_px / 2), (backboard_x, screen_y + backboard_width_px / 2), backboard_thickness_px)
                pygame.draw.circle(self.screen, (255, 100, 0), (screen_x, screen_y), rim_radius_px, rim_thickness_px)

        # --- Ball Physics Debug Info ---
        if 'ball_physics' in data and data['ball_physics'] is not None:
            world_idx = min(self.debug_world_index, len(data['ball_physics']) - 1)
            ball_physics = data['ball_physics'][world_idx]
            for i, ball in enumerate(ball_physics):
                # ball: [inFlight, vx, vy, vz, lastTouchedByID, pointsWorth] (may be shorter)
                in_flight = ball[0] if len(ball) > 0 else None
                velocity = tuple(ball[1:4]) if len(ball) > 3 else (None, None, None)
                last_touched = int(ball[4]) if len(ball) > 4 else None
                points_worth = int(ball[5]) if len(ball) > 5 else None
                info_texts.append(f"Ball {i}: inFlight={in_flight}, velocity={velocity}, lastTouchedByID={last_touched}, pointsWorth={points_worth}")

        # Display all the info text at the end
        for text in info_texts:
            if text: self.screen.blit(self.font.render(text, True, TEXT_COLOR), (20, y_offset)); y_offset += 20
    
    def meters_to_screen(self, meter_x, meter_y):
        """Convert world coordinates in meters to screen coordinates in pixels."""
        screen_x = self.world_offset_x + (meter_x * self.pixels_per_meter)
        screen_y = self.world_offset_y + (meter_y * self.pixels_per_meter)
        return int(screen_x), int(screen_y)

    def step_simulation(self):
        self.sim.step(); self.step_count += 1
        


    def reset_simulation(self):
        # Don't reset when in training mode - let the training handle resets
        if hasattr(self, 'disable_action_input') and self.disable_action_input:
            print("Reset disabled during training mode")
            return
            
        try:
            self.sim.trigger_reset(0)
            print(f"Simulation reset at step {self.step_count}")
            self.step_count = 0
        except Exception as e: print(f"Error resetting simulation: {e}")

    def tick(self):
        self.step_count += 1
        data = self.get_simulation_data() # Get data at the start of the loop

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    self.reset_simulation()
                elif event.key == pygame.K_1:
                    self.debug_world_index = 0
                    print(f"Switched to displaying World {self.debug_world_index}")
                elif event.key == pygame.K_2:
                    if self.max_worlds_available > 1:
                        self.debug_world_index = 1
                        print(f"Switched to displaying World {self.debug_world_index}")
                    else:
                        print(f"World 1 not available (only {self.max_worlds_available} worlds)")
                elif event.key == pygame.K_3:
                    if self.max_worlds_available > 2:
                        self.debug_world_index = 2
                        print(f"Switched to displaying World {self.debug_world_index}")
                    else:
                        print(f"World 2 not available (only {self.max_worlds_available} worlds)")
                elif event.key == pygame.K_4:
                    if self.max_worlds_available > 3:
                        self.debug_world_index = 3
                        print(f"Switched to displaying World {self.debug_world_index}")
                    else:
                        print(f"World 3 not available (only {self.max_worlds_available} worlds)")
                elif event.key == pygame.K_5:
                    if self.max_worlds_available > 4:
                        self.debug_world_index = 4
                        print(f"Switched to displaying World {self.debug_world_index}")
                    else:
                        print(f"World 4 not available (only {self.max_worlds_available} worlds)")
                elif event.key == pygame.K_6:
                    if self.max_worlds_available > 5:
                        self.debug_world_index = 5
                        print(f"Switched to displaying World {self.debug_world_index}")
                    else:
                        print(f"World 5 not available (only {self.max_worlds_available} worlds)")
                elif event.key == pygame.K_7:
                    if self.max_worlds_available > 6:
                        self.debug_world_index = 6
                        print(f"Switched to displaying World {self.debug_world_index}")
                    else:
                        print(f"World 6 not available (only {self.max_worlds_available} worlds)")
                elif event.key == pygame.K_8:
                    if self.max_worlds_available > 7:
                        self.debug_world_index = 7
                        print(f"Switched to displaying World {self.debug_world_index}")
                    else:
                        print(f"World 7 not available (only {self.max_worlds_available} worlds)")
                elif event.key == pygame.K_9:
                    if self.max_worlds_available > 8:
                        self.debug_world_index = 8
                        print(f"Switched to displaying World {self.debug_world_index}")
                    else:
                        print(f"World 8 not available (only {self.max_worlds_available} worlds)")
                elif event.key == pygame.K_0:
                    if self.max_worlds_available > 9:
                        self.debug_world_index = 9
                        print(f"Switched to displaying World {self.debug_world_index}")
                    else:
                        print(f"World 9 not available (only {self.max_worlds_available} worlds)")
                # Interactive training controls
                elif event.key == pygame.K_h and self.controller_manager is not None:
                    # Toggle human control for current agent
                    current_state = self.controller_manager.is_human_control_active()
                    self.controller_manager.set_human_control(not current_state)
                    state_msg = "enabled" if not current_state else "disabled"
                    print(f"🎮 Human control {state_msg} for Agent {self.active_agent_idx} in world 0")
                elif event.key == pygame.K_PAUSE or (event.key == pygame.K_p and pygame.key.get_pressed()[pygame.K_LCTRL]):
                    # Toggle training pause (Ctrl+P to avoid conflict with pass action)
                    self.training_paused = not self.training_paused
                    print(f"⏸ Training {'paused' if self.training_paused else 'resumed'}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left mouse click
                    mouse_x, mouse_y = event.pos
                    # Check if the click was on any agent
                    if data and 'agent_pos' in data and data['agent_pos'] is not None:
                        world_idx = min(self.debug_world_index, len(data['agent_pos']) - 1)
                        agent_positions = data['agent_pos'][world_idx]
                        for i, pos in enumerate(agent_positions):
                            screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                            agent_rect = pygame.Rect(screen_x - 10, screen_y - 10, 200, 200) # A small clickable area
                            if agent_rect.collidepoint(mouse_x, mouse_y):
                                self.active_agent_idx = i
                                print(f"Switched control to Agent {i}")
                                break # Stop after finding the first clicked agent

        self.handle_interactive_input()  # Handle human control input
        
        self.handle_audio_events(data)
        self.draw_simulation_data(data)

        pygame.display.flip()
        # self.clock.tick(60)

    def set_controller_manager(self, controller_manager):
        """Set the controller manager for interactive training"""
        self.controller_manager = controller_manager
    
    def set_training_paused(self, paused: bool):
        """Set training pause state"""
        self.training_paused = paused
    
    def get_human_action(self):
        """Get the current human action for the HumanController"""
        # Always return a torch tensor to ensure compatibility
        if TORCH_AVAILABLE and torch is not None:
            if hasattr(self.human_action, 'clone'):
                return self.human_action.clone()
            elif isinstance(self.human_action, list):
                return torch.tensor(self.human_action, dtype=torch.int32)
            else:
                # Fallback: convert whatever we have to tensor
                return torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int32)
        else:
            # If torch not available, return list (should not happen in training)
            return list(self.human_action) if hasattr(self.human_action, '__iter__') else [0, 0, 0, 0, 0, 0]
    
    def handle_interactive_input(self):
        """Handle keyboard input for interactive training control"""
        if self.controller_manager is None:
            # Debug: Print when controller manager is missing
            if not hasattr(self, '_controller_manager_warning_shown'):
                print("🔧 Debug: No controller manager set, interactive input disabled")
                self._controller_manager_warning_shown = True
            return
            
        keys = pygame.key.get_pressed()
        
        # Reset action
        move_speed = 0
        move_angle = 0  # Default angle
        rotate = 0
        grab = 0
        pass_ball = 0
        shoot_ball = 0
        
        # Movement controls (WASD) - Allow diagonal movement
        if keys[pygame.K_w] and keys[pygame.K_d]:
            move_speed = 1
            move_angle = 1  # Northeast
        elif keys[pygame.K_d] and keys[pygame.K_s]:
            move_speed = 1
            move_angle = 3  # Southeast
        elif keys[pygame.K_s] and keys[pygame.K_a]:
            move_speed = 1
            move_angle = 5  # Southwest
        elif keys[pygame.K_a] and keys[pygame.K_w]:
            move_speed = 1
            move_angle = 7  # Northwest
        elif keys[pygame.K_w]:
            move_speed = 1
            move_angle = 0  # North
        elif keys[pygame.K_d]:
            move_speed = 1
            move_angle = 2  # East
        elif keys[pygame.K_s]:
            move_speed = 1
            move_angle = 4  # South
        elif keys[pygame.K_a]:
            move_speed = 1
            move_angle = 6  # West
        
        # Rotation (Q/E or comma/period)
        if keys[pygame.K_COMMA] or keys[pygame.K_q]:
            rotate = -1  # Counter Clockwise
        elif keys[pygame.K_PERIOD] or keys[pygame.K_e]:
            rotate = 1  # Clockwise
        else:
            rotate = 0  # No rotation
        
        # Actions
        if keys[pygame.K_SPACE]:
            pass_ball = 1
        if keys[pygame.K_LSHIFT]:  
            grab = 1
        if keys[pygame.K_RETURN] or keys[pygame.K_RSHIFT]:
            shoot_ball = 1
        
        # Convert to tensor format [move_speed, move_angle, rotate, grab, pass_ball, shoot_ball]
        # Always use tensor for consistency
        if TORCH_AVAILABLE and torch is not None:
            new_action = torch.tensor([move_speed, move_angle, rotate, grab, pass_ball, shoot_ball], dtype=torch.int32)
            
            # Check if action changed
            if hasattr(self.human_action, 'equal') and isinstance(self.human_action, torch.Tensor):
                action_changed = not torch.equal(self.human_action, new_action)
            else:
                action_changed = True  # Force update if not tensor
        else:
            new_action = [move_speed, move_angle, rotate, grab, pass_ball, shoot_ball]
            action_changed = self.human_action != new_action
            
        if action_changed:
            self.human_action = new_action
            self.action_changed = True
            # Debug: Print when any input is detected (reduced frequency)
            if (hasattr(self, '_last_debug_frame') and 
                (not hasattr(self, '_last_debug_frame') or self.step_count - getattr(self, '_last_debug_frame', 0) > 10)):
                print(f"🎮 Human input detected for Agent {self.active_agent_idx}: move={move_speed}, angle={move_angle}, rot={rotate}, grab={grab}, pass={pass_ball}, shoot={shoot_ball}")
                self._last_debug_frame = self.step_count

    def get_selected_agent_index(self):
        """Get the index of the currently selected agent for human control"""
        return self.active_agent_idx

    def draw_events(self, parsed_events, event_def, current_episode, display_mode):
        if display_mode == "Off" or not event_def:
            return

        if event_def and parsed_events:
            for event in parsed_events:
                if display_mode == "Current Episode" and event['episode_idx'] != current_episode:
                    continue
                
                screen_x, screen_y = self.meters_to_screen(event['pos'][0], event['pos'][1])
                visual_info = event_def["visuals"].get(event['outcome'])

                if visual_info:
                    if visual_info["shape"] == "circle":
                        pygame.draw.circle(self.screen, visual_info["color"], (screen_x, screen_y), visual_info["size"])
                    elif visual_info["shape"] == "x":
                        size = visual_info["size"]
                        thickness = visual_info.get("thickness", 2)  # Default thickness if not specified
                        pygame.draw.line(self.screen, visual_info["color"], (screen_x - size, screen_y - size), (screen_x + size, screen_y + size), thickness)
                        pygame.draw.line(self.screen, visual_info["color"], (screen_x + size, screen_y - size), (screen_x - size, screen_y + size), thickness)
                    elif visual_info["shape"] == "square":
                        size = visual_info["size"]
                        pygame.draw.rect(self.screen, visual_info["color"], (screen_x - size, screen_y - size, size * 2, size * 2))

    def draw_scene_static(self, hoop_pos):
        """
        Creates and returns static surfaces for background and events.
        This should only be called once or when static elements change.
        """
        # Create background surface with court and hoops
        background = pygame.Surface(self.screen.get_size())
        background.fill(BACKGROUND_COLOR)
        
        original_screen = self.screen
        self.screen = background
        self.draw_basketball_court()
        
        if hoop_pos is not None:
            hoop_positions = hoop_pos[0]
            for pos in hoop_positions:
                screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                backboard_width_px = BACKBOARD_WIDTH_M * self.pixels_per_meter
                rim_radius_px = (RIM_DIAMETER_M / 2) * self.pixels_per_meter
                rim_thickness_px = max(2, int(self.pixels_per_meter * 0.02))
                backboard_thickness_px = max(3, int(self.pixels_per_meter * 0.05))
                backboard_offset_px = BACKBOARD_OFFSET_FROM_HOOP_M * self.pixels_per_meter
                backboard_x = screen_x - backboard_offset_px if pos[0] < self.world_width_meters / 2 else screen_x + backboard_offset_px
                pygame.draw.line(self.screen, (255, 255, 255), (backboard_x, screen_y - backboard_width_px / 2), (backboard_x, screen_y + backboard_width_px / 2), backboard_thickness_px)
                pygame.draw.circle(self.screen, (255, 100, 0), (screen_x, screen_y), rim_radius_px, rim_thickness_px)
        
        self.screen = original_screen
        return background

    def draw_scene_dynamic(self, agent_pos_frame, ball_pos_frame, orientation_frame, episodes_completed_at_this_step_for_world, current_playback_episode, trail_surface, world_num, fading_trails=False, current_step=0, max_episode_length=1):
        """
        Draws dynamic elements (agents and balls) during trajectory playback,
        but only for worlds that have not yet completed the current target episode number.
        """
        num_agents, _ = agent_pos_frame.shape
        
        if episodes_completed_at_this_step_for_world == current_playback_episode:
            for agent_idx in range(num_agents):
                pos = agent_pos_frame[agent_idx]
                q = orientation_frame[agent_idx]
                screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])

                if fading_trails and current_step > 0:
                    base_trail_color = TEAM0_COLOR if agent_idx % 2 == 0 else TEAM1_COLOR
                    x = current_step / max_episode_length if max_episode_length > 0 else 0
                    faded_trail_color = tuple(int(x * 0.5 * c + (1 - x) * c) for c in base_trail_color)
                    pygame.draw.circle(trail_surface, faded_trail_color, (screen_x, screen_y), 3)
                elif not fading_trails:
                    trail_color = TEAM0_COLOR if agent_idx % 2 == 0 else TEAM1_COLOR
                    pygame.draw.circle(trail_surface, trail_color, (screen_x, screen_y), 3)

                if not fading_trails or current_step == 0:
                    agent_color = TEAM0_COLOR if agent_idx % 2 == 0 else TEAM1_COLOR
                    forward_vec_3d = rotate_vec(q, np.array([0.0, 1.0, 0.0]))
                    forward_vec = np.array([forward_vec_3d[0], forward_vec_3d[1]])
                    right_vec = np.array([forward_vec[1], -forward_vec[0]])
                    shoulder_width_px = AGENT_SHOULDER_WIDTH * self.pixels_per_meter
                    depth_px = AGENT_DEPTH * self.pixels_per_meter
                    center_point = np.array([screen_x, screen_y])
                    half_width_vec = right_vec * (shoulder_width_px / 2)
                    half_depth_vec = forward_vec * (depth_px / 2)
                    p1 = center_point - half_depth_vec + half_width_vec
                    p2 = center_point - half_depth_vec - half_width_vec
                    p3 = center_point + half_depth_vec - half_width_vec
                    p4 = center_point + half_depth_vec + half_width_vec
                    agent_points = [p1, p2, p3, p4]
                    pygame.draw.polygon(self.screen, agent_color, agent_points)
                    pygame.draw.polygon(self.screen, (255, 255, 255), agent_points, 1)
                    font_world_num = pygame.font.Font(None, 22)
                    world_num_surface = font_world_num.render(str(world_num), True, (255, 255, 255))
                    world_num_rect = world_num_surface.get_rect(center=(screen_x, screen_y))
                    self.screen.blit(world_num_surface, world_num_rect)

                    arrow_len_px = self.pixels_per_meter * AGENT_ORIENTATION_ARROW_LENGTH_M
                    arrow_end = (screen_x + arrow_len_px * forward_vec[0], screen_y + arrow_len_px * forward_vec[1])
                    pygame.draw.line(self.screen, (255, 255, 0), (screen_x, screen_y), arrow_end, 2)
            
            if not fading_trails or current_step == 0:
                ball_pos = ball_pos_frame[0]
                screen_x, screen_y = self.meters_to_screen(ball_pos[0], ball_pos[1])
                ball_radius_px = BALL_RADIUS_M * self.pixels_per_meter
                pygame.draw.circle(self.screen, (255, 100, 0), (screen_x, screen_y), int(ball_radius_px))
                font_world_num = pygame.font.Font(None, 22)
                world_num_surface = font_world_num.render(str(world_num), True, (255, 255, 255))
                world_num_rect = world_num_surface.get_rect(center=(screen_x, screen_y))
                self.screen.blit(world_num_surface, world_num_rect)

    def run_trajectory_playback(self, log_path, fading_trails=False, event_to_track="shoot"):
        """
        Loads a trajectory log file and plays back multiple episodes, pausing
        between each one and waiting for user input to continue.
        """
        print(f"Loading trajectory data from {log_path}...")
        try:
            log_data = np.load(log_path)
            agent_pos_log = log_data.get('agent_pos')
            ball_pos_log = log_data.get('ball_pos')
            hoop_pos = log_data.get('hoop_pos')
            orientation_log = log_data.get('orientation')
            ball_physics_log = log_data.get('ball_physics')
            actions_log = log_data.get('actions')
            done_log = log_data.get('done')

            if any(data is None for data in [agent_pos_log, ball_pos_log, hoop_pos, orientation_log, done_log]):
                print("FATAL: Log file is missing one or more required data arrays.")
                return
            
            if 'num_episodes' in log_data:
                total_episodes_in_log = int(log_data['num_episodes']) - 1
                print(f"Log file contains data for up to {total_episodes_in_log} episodes.")
            else:
                total_episodes_in_log = 0
        
            num_steps, num_worlds, num_agents, _ = agent_pos_log.shape
            print(f"Playing back {total_episodes_in_log} episodes.")

            episode_breaks = [[] for _ in range(num_worlds)]
            parsed_events = []
            event_def = EVENT_DEFINITIONS.get(event_to_track)
            episodes_completed_log = np.cumsum(done_log, axis=0)

            print("Parsing episodes and events from log...")
            
            # Debug: Print agent_possession array info
            if 'agent_possession' in log_data:
                possession_shape = log_data['agent_possession'].shape
                print(f"DEBUG: agent_possession shape: {possession_shape}")
                print(f"DEBUG: Sample possession values at step 10:")
                if len(possession_shape) >= 3:
                    for w in range(min(3, possession_shape[1])):  # Show first 3 worlds
                        for a in range(min(2, possession_shape[2])):  # Show first 2 agents
                            if len(possession_shape) == 4:
                                # Show all components to understand the structure
                                for c in range(possession_shape[3]):
                                    val = log_data['agent_possession'][10, w, a, c] if possession_shape[0] > 10 else "N/A"
                                    print(f"  World {w}, Agent {a}, Component {c}: {val}")
                            else:
                                val = log_data['agent_possession'][10, w, a] if possession_shape[0] > 10 else "N/A"
                                print(f"  World {w}, Agent {a}: {val}")
                print("") # Empty line for readability
            
            for world_num in range(num_worlds):
                last_done_step = -1
                for step_num in range(1, num_steps):
                    if (done_log[step_num][world_num] > 0):
                        episode_breaks[world_num].append({'start': last_done_step + 1, 'end': step_num})
                        print(f"World {world_num}: Episode {len(episode_breaks[world_num])-1} from step {last_done_step + 1} to {step_num}")
                        last_done_step = step_num
                
                print(f"World {world_num} final episode breaks: {episode_breaks[world_num]}")
                
                # Now check for events in this world
                for step_num in range(1, num_steps):
                    if event_def and actions_log is not None and ball_physics_log is not None:
                        action_idx = event_def["action_idx"]
                        
                        # Check if this step is within any episode for this world
                        step_in_valid_episode = False
                        for ep_info in episode_breaks[world_num]:
                            if ep_info['start'] <= step_num <= ep_info['end']:
                                step_in_valid_episode = True
                                break
                        
                        if not step_in_valid_episode:
                            continue  # Skip steps that aren't in any episode for this world
                        
                        if len(actions_log.shape) == 4:  # [steps, worlds, agents, actions] - multi-agent
                            for agent_num in range(num_agents):
                                if (step_num < len(actions_log) and actions_log[step_num, world_num, agent_num, action_idx] == 1): # Checks if action happened this timestep
                                    if event_def["conditions"](log_data, step_num, world_num, agent_num):
                                        outcome = event_def["outcome_func"](log_data, step_num, world_num)
                                        
                                        # Debug: Print possession values for pass events
                                        if event_to_track == "pass":
                                            if 'agent_possession' in log_data:
                                                # Get possession values for steps before, during, and after the event
                                                possession_before = log_data['agent_possession'][step_num-1, world_num, agent_num] if step_num > 0 else "N/A"
                                                possession_during = log_data['agent_possession'][step_num, world_num, agent_num] if step_num < len(log_data['agent_possession']) else "N/A"
                                                possession_after = log_data['agent_possession'][step_num+1, world_num, agent_num] if step_num+1 < len(log_data['agent_possession']) else "N/A"
                                                
                                                print(f"PASS DEBUG: Step {step_num}, World {world_num}, Agent {agent_num}")
                                                print(f"  Possession[{step_num-1}] (before): {possession_before}")
                                                print(f"  Possession[{step_num}] (during): {possession_during}")
                                                print(f"  Possession[{step_num+1}] (after): {possession_after}")
                                                print(f"  Action value: {actions_log[step_num, world_num, agent_num, action_idx]}")
                                                
                                                # Check if this is the problematic first step of an episode
                                                is_episode_start = False
                                                for ep_info in episode_breaks[world_num]:
                                                    if step_num == ep_info['start']:
                                                        is_episode_start = True
                                                        break
                                                if is_episode_start:
                                                    print(f"  ⚠️  WARNING: This is the FIRST STEP of an episode!")
                                        
                                        # Find which episode this step belongs to by checking episode_breaks
                                        current_episode = 0
                                        found_episode = False
                                        for ep_idx, ep_info in enumerate(episode_breaks[world_num]):
                                            if ep_info['start'] <= step_num <= ep_info['end']:
                                                current_episode = ep_idx
                                                found_episode = True
                                                break
                                        
                                        if not found_episode:
                                            print(f"WARNING: Step {step_num} in World {world_num} not found in any episode!")
                                            print(f"Available episodes for World {world_num}: {episode_breaks[world_num]}")
                                            continue  # Skip this event as it's not in any valid episode
                                        
                                        # Calculate episode step
                                        episode_step_in_episode = step_num - episode_breaks[world_num][current_episode]['start'] if len(episode_breaks[world_num]) > current_episode else step_num
                                        
                                        event_data = {
                                            'pos': agent_pos_log[step_num, world_num, agent_num],
                                            'outcome': outcome,
                                            'episode_idx' : current_episode,
                                            'world_num': world_num,
                                            'step_num': step_num,
                                            'episode_step': episode_step_in_episode,
                                            'agent_num': agent_num
                                        }
                                        parsed_events.append(event_data)
                                        
                                        print(f"DEBUG EVENT: World {world_num}, Agent {agent_num}, Step {step_num}, Episode {current_episode}, Episode Step {episode_step_in_episode}, Outcome: {outcome}, Pos: ({event_data['pos'][0]:.2f}, {event_data['pos'][1]:.2f})")
                        elif len(actions_log.shape) == 3:  # [steps, worlds, actions] - single agent per world
                            if (step_num < len(actions_log) and actions_log[step_num, world_num, action_idx] == 1):
                                # Use agent 0 since we only have one agent per world in inference
                                agent_num = 0
                                if event_def["conditions"](log_data, step_num, world_num, agent_num):
                                    outcome = event_def["outcome_func"](log_data, step_num, world_num)
                                    
                                    # Debug: Print possession values for pass events
                                    if event_to_track == "pass":
                                        if 'agent_possession' in log_data:
                                            # Get possession values for steps before, during, and after the event
                                            possession_before = log_data['agent_possession'][step_num-1, world_num, agent_num] if step_num > 0 else "N/A"
                                            possession_during = log_data['agent_possession'][step_num, world_num, agent_num] if step_num < len(log_data['agent_possession']) else "N/A"
                                            possession_after = log_data['agent_possession'][step_num+1, world_num, agent_num] if step_num+1 < len(log_data['agent_possession']) else "N/A"
                                            
                                            print(f"PASS DEBUG: Step {step_num}, World {world_num}, Agent {agent_num}")
                                            print(f"  Possession[{step_num-1}] (before): {possession_before}")
                                            print(f"  Possession[{step_num}] (during): {possession_during}")
                                            print(f"  Possession[{step_num+1}] (after): {possession_after}")
                                            print(f"  Action value: {actions_log[step_num, world_num, action_idx]}")
                                            
                                            # Check if this is the problematic first step of an episode
                                            is_episode_start = False
                                            for ep_info in episode_breaks[world_num]:
                                                if step_num == ep_info['start']:
                                                    is_episode_start = True
                                                    break
                                            if is_episode_start:
                                                print(f"  ⚠️  WARNING: This is the FIRST STEP of an episode!")
                                    
                                    # Find which episode this step belongs to by checking episode_breaks
                                    current_episode = 0
                                    found_episode = False
                                    for ep_idx, ep_info in enumerate(episode_breaks[world_num]):
                                        if ep_info['start'] <= step_num <= ep_info['end']:
                                            current_episode = ep_idx
                                            found_episode = True
                                            break
                                    
                                    if not found_episode:
                                        print(f"WARNING: Step {step_num} in World {world_num} not found in any episode!")
                                        print(f"Available episodes for World {world_num}: {episode_breaks[world_num]}")
                                        continue  # Skip this event as it's not in any valid episode
                                    
                                    # Calculate episode step
                                    episode_step_in_episode = step_num - episode_breaks[world_num][current_episode]['start'] if len(episode_breaks[world_num]) > current_episode else step_num
                                    
                                    event_data = {
                                        'pos': agent_pos_log[step_num, world_num, 0],  
                                        'outcome': outcome,
                                        'episode_idx' : current_episode,
                                        'world_num': world_num,
                                        'step_num': step_num,
                                        'episode_step': episode_step_in_episode,
                                        'agent_num': agent_num
                                    }
                                    parsed_events.append(event_data)
                                    
                                    # print(f"DEBUG EVENT: World {world_num}, Agent {agent_num}, Step {step_num}, Episode {current_episode}, Episode Step {episode_step_in_episode}, Outcome: {outcome}, Pos: ({event_data['pos'][0]:.2f}, {event_data['pos'][1]:.2f})")
                        else:
                            print(f"WARNING: Unexpected actions_log shape: {actions_log.shape}")

            # Print summary of all detected events
            print(f"\n=== EVENT SUMMARY ===")
            print(f"Total {event_to_track} events detected: {len(parsed_events)}")
            
            # Group events by episode
            events_by_episode = {}
            for event in parsed_events:
                ep = event['episode_idx']
                if ep not in events_by_episode:
                    events_by_episode[ep] = []
                events_by_episode[ep].append(event)
            
            for episode_idx in sorted(events_by_episode.keys()):
                events_in_episode = events_by_episode[episode_idx]
                print(f"Episode {episode_idx}: {len(events_in_episode)} events")
                for event in events_in_episode:
                    print(f"  World {event['world_num']}, Step {event['step_num']}, Episode Step {event['episode_step']}, Outcome: {event['outcome']}")
            
            print("=== END EVENT SUMMARY ===\n")

            background = self.draw_scene_static(hoop_pos)

            trail_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

            episode_step = 0
            current_playback_episode = 0
            is_paused_for_next_episode = False
            paused = False
            running = True
            show_trails = False
            event_display_modes = ['Off', 'Current Episode', 'All Episodes']
            event_display_mode_idx = 0
            
            episode_lengths = [
                world_info[current_playback_episode]['end'] - world_info[current_playback_episode]['start'] 
                for world_info in episode_breaks 
                if len(world_info) > current_playback_episode
            ]

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                            print("Paused" if paused else "Playing")
                        if event.key == pygame.K_b:
                            if current_playback_episode > 0:
                                current_playback_episode -= 1
                                if not fading_trails:
                                    trail_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                                episode_lengths = [
                                    world_info[current_playback_episode]['end'] - world_info[current_playback_episode]['start'] 
                                    for world_info in episode_breaks 
                                    if len(world_info) > current_playback_episode
                                ]
                                print(f"Now, current episode is: {current_playback_episode} and the max lengths are: {episode_lengths}")
                                is_paused_for_next_episode = False
                                paused = False
                                episode_step = 0
                                print(f"Playing Episode {current_playback_episode}")
                        if event.key == pygame.K_n:
                            if current_playback_episode < total_episodes_in_log:
                                current_playback_episode += 1
                                if not fading_trails:
                                    trail_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                                episode_lengths = [
                                    world_info[current_playback_episode]['end'] - world_info[current_playback_episode]['start'] 
                                    for world_info in episode_breaks 
                                    if len(world_info) > current_playback_episode
                                ]
                                print(f"Now, current episode is: {current_playback_episode} and the max lengths are: {episode_lengths}")
                                is_paused_for_next_episode = False
                                paused = False
                                episode_step = 0
                                print(f"Playing Episode {current_playback_episode}")
                        if event.key == pygame.K_t:
                            show_trails = not show_trails
                        if event.key == pygame.K_c:  # Toggle event chart with 'C' key
                            keys = pygame.key.get_pressed()
                            if keys[pygame.K_LSHIFT or pygame.K_RSHIFT]:
                                event_display_mode_idx = 2 if event_display_mode_idx != 2 else 0
                            else:
                                event_display_mode_idx = 1 if event_display_mode_idx != 1 else 0
                            print(f"Event chart mode: {event_display_modes[event_display_mode_idx]}")
                        if event.key == pygame.K_r:
                            episode_step = 0
                        if event.key == pygame.K_PERIOD:
                            episode_step += 1 if episode_step < max(episode_lengths) else 0
                        if event.key == pygame.K_COMMA:
                            episode_step -= 1 if episode_step > 0 else 0
                
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    is_paused_for_next_episode = False
                    episode_step = max(0, episode_step-1 if not keys[pygame.K_LSHIFT] else episode_step-5)
                if keys[pygame.K_RIGHT]:
                    episode_step = min(max(episode_lengths), episode_step+1 if not keys[pygame.K_LSHIFT] else episode_step+5)

                self.screen.blit(background, (0, 0))
                
                current_display_mode = event_display_modes[event_display_mode_idx]
                
                self.draw_events(parsed_events, event_def, current_playback_episode, current_display_mode)
                
                if show_trails:
                    if fading_trails:
                        max_episode_length = max(episode_lengths) if episode_lengths else 1
                        
                        # Draw fading trails
                        for trail_step in range(max(0, episode_step - max_episode_length + 1), episode_step + 1):
                            if trail_step < 0:
                                continue
                            for world_num in range(num_worlds):
                                if trail_step + episode_breaks[world_num][current_playback_episode]['start'] >= num_steps:
                                    continue
                                trail_agent_pos = agent_pos_log[episode_breaks[world_num][current_playback_episode]['start'] + trail_step][world_num]
                                trail_episodes_completed = episodes_completed_log[episode_breaks[world_num][current_playback_episode]['start'] + trail_step]
                                
                                # Draw fading trail points directly for this step
                                if trail_episodes_completed[world_num] == current_playback_episode:
                                    num_agents, _ = trail_agent_pos.shape
                                    for agent_idx in range(num_agents):
                                        pos = trail_agent_pos[agent_idx]
                                        screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                                        
                                        # Calculate fading color
                                        base_trail_color = TEAM0_COLOR if agent_idx % 2 == 0 else TEAM1_COLOR
                                        x = (episode_step - trail_step) / max_episode_length if max_episode_length > 0 else 0
                                        faded_trail_color = tuple(int(x * 0.5 * c + (1 - x) * c) for c in base_trail_color)
                                        pygame.draw.circle(self.screen, faded_trail_color, (screen_x, screen_y), 3)
                    else:
                        self.screen.blit(trail_surface, (0, 0))

                for world_num in range(num_worlds):
                    step_index = episode_breaks[world_num][current_playback_episode]['start'] + episode_step
                    if step_index >= num_steps:
                        continue  # Skip this world if we're beyond available data
                    
                    # Ensure this world has the current episode
                    if len(episode_breaks[world_num]) <= current_playback_episode:
                        continue  # Skip this world if it doesn't have this episode
                        
                    agent_pos_frame = agent_pos_log[step_index][world_num]
                    ball_pos_frame = ball_pos_log[step_index][world_num]
                    orientation_frame = orientation_log[step_index][world_num]

                    episodes_completed_at_this_step = episodes_completed_log[step_index]
                    self.draw_scene_dynamic(agent_pos_frame, ball_pos_frame, orientation_frame, episodes_completed_at_this_step[world_num], current_playback_episode, trail_surface, world_num, fading_trails, 0, max(episode_lengths) if episode_lengths else 1)

                # Draw status text
                status_text = f"Viewing Episode: {current_playback_episode}/{total_episodes_in_log} | step: {episode_step}"
                if is_paused_for_next_episode:
                    status_text += f" | Press 'N' for Next Episode || max episode length is: {max(episode_lengths)}"
                elif paused:
                    status_text += " | Paused"
                status_text += f" | Trails: {'On' if show_trails else 'Off'}"
                status_text += f" | Events: {current_display_mode}"
                text_surface = self.font.render(status_text, True, (255, 255, 0))
                self.screen.blit(text_surface, (10, 10))
                controls_text = f"Pause: Space | FF: shift + right | Trails: T | Events: C/ShiftC | frame by frame: , or ."
                controls_surface = self.font.render(controls_text, True, (255, 255, 255))
                self.screen.blit(controls_surface, (10, WINDOW_HEIGHT-30))

                pygame.display.flip()

                if not paused and not is_paused_for_next_episode:
                    episode_step += 1

                max_episode_length = max(episode_lengths) if episode_lengths else 0
                
                if episode_step >= max_episode_length:
                    paused = True
                    is_paused_for_next_episode = True

                # self.clock.tick(60)

            pygame.quit()
            sys.exit()

        except Exception as e:
            print(f"Error in trajectory playback: {e}")
            import traceback
            traceback.print_exc()
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Madrona Basketball Viewer: Live Sim or Trajectory Playback")
    parser.add_argument('--playback-log', type=str, default="logs/trajectories.npz", help="Path to an .npz trajectory log file for playback.")
    parser.add_argument('--track-event', type=str, default="shoot", help="name of event to track: shoot, pass, grab, etc. Events defined in constants.py")
    parser.add_argument('--fading-trails', action='store_true', default=False) # Darkening trails slow down performance, only use if necessary
    args = parser.parse_args()

    if args.playback_log:
        viewer = ViewerClass()
        viewer.run_trajectory_playback(args.playback_log, args.fading_trails, args.track_event)
    else:
        print("No playback log provided. To view trajectories, run with:")
        print("python viewer.py --playback-log path/to/your/log.npz")
        print("Optional: --track-event shoot|pass|grab --fading-trails")
