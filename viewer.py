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
except Exception as e:
    print(f"⚠ PyTorch import issue: {e}")

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

    def __init__(self, sim_instance, training_mode=False):
        # CRITICAL: Initialize pygame BEFORE any CUDA operations to avoid context conflicts
        print("🔧 Initializing viewer with GPU safety measures...")
        
        # Store simulation instance but don't access it yet - this is crucial!
        self.sim = sim_instance
        self.simulation_ready = False  # Track if simulation is ready for data access
        self.ready_check_attempts = 0  # Count attempts to check if simulation is ready
        
        # CRITICAL FIX: Disable action input during training to prevent segfaults
        self.disable_action_input = training_mode
        if training_mode:
            print("✓ Training mode detected - disabling viewer action input to prevent conflicts")
        
        # Add a flag to detect if we're running on GPU
        self.is_gpu_simulation = False
        self.gpu_device = None
        
        # DEBUG: Add ability to override which world index to display (for debugging)
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
            pygame.display.set_caption("Madrona Simulation Pipeline")
            print("✓ Pygame display created")
        except Exception as e:
            print(f"✗ Pygame display creation failed: {e}")
            raise RuntimeError(f"Cannot create pygame display: {e}")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.active_agent_idx = 0

        # --- Load Sound Effects ---
        try:
            pygame.mixer.init()  # Initialize the audio mixer
            self.score_sound = pygame.mixer.Sound("assets/swish.wav")
            self.whistle_sound = pygame.mixer.Sound("assets/whistle.wav")
            print("✓ Audio files loaded successfully.")
        except pygame.error as e:
            print(f"⚠ Warning: Could not initialize audio. Running in silent mode. Error: {e}")
            self.score_sound = None
            self.whistle_sound = None

        # --- Correctly set up the coordinate system ---

        # 1. These are the TRUE world dimensions in meters, based on your constants.
        self.world_width_meters = WORLD_WIDTH_M
        self.world_height_meters = WORLD_HEIGHT_M
        self.pixels_per_meter = PIXELS_PER_METER

        # 2. These are the TRUE pixel dimensions for drawing, calculated only ONCE.
        self.world_width_px = self.world_width_meters * PIXELS_PER_METER
        self.world_height_px = self.world_height_meters * PIXELS_PER_METER

        # 3. This is the TRUE pixel offset for centering the world view.
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
                    game_state = safe_tensor_to_numpy(self.sim.game_state_tensor().to_torch())
                except:
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
        COURT_BLUE = (10, 30, 70)
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
        pygame.draw.rect(self.screen, COURT_BLUE, court_rect)

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
                    # Starts from the top-left quadrant, goes to the bottom-left
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

        # Get team colors 
        team_colors = { 0: (0, 100, 255), 1: (255, 50, 50) }
        if 'agent_teams' in data:
            agent_teams = data['agent_teams'][world_idx]
            for i, team_data in enumerate(agent_teams):
                if len(team_data) >= 4:
                    team_index = int(team_data[0])
                    if team_index not in team_colors:
                        import struct
                        color_r = max(0, min(255, int(struct.unpack('f', struct.pack('I', np.uint32(team_data[1])))[0])))
                        color_g = max(0, min(255, int(struct.unpack('f', struct.pack('I', np.uint32(team_data[2])))[0])))
                        color_b = max(0, min(255, int(struct.unpack('f', struct.pack('I', np.uint32(team_data[3])))[0])))
                        team_colors[team_index] = (color_r, color_g, color_b)

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
            
            team_colors = { 0: (0, 100, 255), 1: (255, 50, 50) } 
            
            for i, pos in enumerate(positions):
                screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                team_index = int(team_data[i][0]) if i < len(team_data) else 0
                agent_color = team_colors.get(team_index, (128, 128, 128))
                
                agent_size_px = self.pixels_per_meter * AGENT_SIZE_M
                agent_rect = pygame.Rect(screen_x - agent_size_px / 2, screen_y - agent_size_px / 2, agent_size_px, agent_size_px)
                pygame.draw.rect(self.screen, agent_color, agent_rect)

                # Draw a special highlight for the active agent
                if i == self.active_agent_idx:
                    pygame.draw.rect(self.screen, (0, 255, 255), agent_rect, 3) # Bright yellow outline
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), agent_rect, 1) # Standard white outline

                # Draw the agent's number (not team ID) inside the rectangle
                font_small = pygame.font.Font(None, int(agent_size_px * 0.8))
                text_surface = font_small.render(str(i), True, (255, 255, 255))
                self.screen.blit(text_surface, text_surface.get_rect(center=agent_rect.center))
                
                # Draw orientation line
                q = orientations[i]
                BASE_FORWARD_VECTOR = np.array([0.0, 1.0, 0.0])
                direction_3d = rotate_vec(q, BASE_FORWARD_VECTOR)
                dx, dy = direction_3d[0], direction_3d[1]
                arrow_len_px = self.pixels_per_meter * AGENT_ORIENTATION_ARROW_LENGTH_M
                arrow_end = (screen_x + arrow_len_px * dx, screen_y + arrow_len_px * dy)
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
        if 'ball_physics' in data:
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
    
    def handle_input(self):
        # CRITICAL FIX: Don't call set_action when viewer is used with GPU simulation
        # This prevents segfaults when the viewer is used during training or with GPU
        if hasattr(self, 'disable_action_input') and self.disable_action_input:
            return
            
        keys = pygame.key.get_pressed()
        move_speed, move_angle, rotate, grab, pass_ball, shoot_ball = 0,0,0,0,0,0

        # This logic now controls the selected agent
        if keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d]:
            move_speed = 1
            if keys[pygame.K_w] and keys[pygame.K_d]: move_angle = 1
            elif keys[pygame.K_d] and keys[pygame.K_s]: move_angle = 3
            elif keys[pygame.K_s] and keys[pygame.K_a]: move_angle = 5
            elif keys[pygame.K_a] and keys[pygame.K_w]: move_angle = 7
            elif keys[pygame.K_w]: move_angle = 0
            elif keys[pygame.K_d]: move_angle = 2
            elif keys[pygame.K_s]: move_angle = 4
            elif keys[pygame.K_a]: move_angle = 6
        if keys[pygame.K_q]: rotate = -1
        elif keys[pygame.K_e]: rotate = 1
        if keys[pygame.K_LSHIFT]: grab = 1
        if keys[pygame.K_SPACE]: pass_ball = 1
        if keys[pygame.K_h]: shoot_ball = 1

        # CRITICAL: Only call set_action if we're in CPU mode or confirmed safe mode
        # Check if we're running on GPU and disable action input accordingly
        if hasattr(self, 'is_gpu_simulation') and self.is_gpu_simulation:
            if not hasattr(self, 'gpu_warning_shown'):
                print("⚠ GPU simulation detected - disabling interactive controls to prevent crashes")
                print("  The viewer will display the simulation but won't accept keyboard input")
                self.gpu_warning_shown = True
                self.disable_action_input = True
            return
        
        # Only call set_action if we're in a safe environment (CPU simulation)
        try:
            self.sim.set_action(0, self.active_agent_idx, move_speed, move_angle, rotate, grab, pass_ball, shoot_ball)
        except Exception as e:
            print(f"Warning: Could not set action: {e}")
            print("Disabling interactive controls to prevent further errors")
            # Disable future action input attempts to prevent repeated errors
            self.disable_action_input = True
            return

        # You can keep a second set of controls for another agent if you wish,
        # or have all other agents be controlled by the hard-coded AI.
        # For now, we'll leave the second player controls as they are.
        move_speed1, move_angle1, rotate1, grab1, pass_ball1, shoot_ball1 = 0,0,0,0,0,0
        if keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
            move_speed1 = 1
            if keys[pygame.K_UP] and keys[pygame.K_RIGHT]: move_angle1 = 1
        # move_speed1, move_angle1, rotate1, grab1, pass_ball1, shoot_ball1 = 0,0,0,0,0,0
        # if keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
        #     move_speed1 = 1
        #     if keys[pygame.K_UP] and keys[pygame.K_RIGHT]: move_angle1 = 1
        #     elif keys[pygame.K_RIGHT] and keys[pygame.K_DOWN]: move_angle1 = 3
        #     elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]: move_angle1 = 5
        #     elif keys[pygame.K_LEFT] and keys[pygame.K_UP]: move_angle1 = 7
        #     elif keys[pygame.K_UP]: move_angle1 = 0
        #     elif keys[pygame.K_RIGHT]: move_angle1 = 2
        #     elif keys[pygame.K_DOWN]: move_angle1 = 4
        #     elif keys[pygame.K_LEFT]: move_angle1 = 6
        # if keys[pygame.K_COMMA]: rotate1 = -1
        # elif keys[pygame.K_PERIOD]: rotate1 = 1
        # if keys[pygame.K_KP0]: grab1 = 1
        # if keys[pygame.K_RCTRL] : pass_ball1 = 1
        # if keys[pygame.K_RSHIFT] : shoot_ball1 = 1
        # self.sim.set_action(0, 0, move_speed, move_angle, rotate, grab, pass_ball, shoot_ball)
        # self.sim.set_action(0, 1, move_speed1, move_angle1, rotate1, grab1, pass_ball1, shoot_ball1)

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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left mouse click
                    mouse_x, mouse_y = event.pos
                    # Check if the click was on any agent
                    if data and 'agent_pos' in data and data['agent_pos'] is not None:
                        world_idx = min(self.debug_world_index, len(data['agent_pos']) - 1)
                        agent_positions = data['agent_pos'][world_idx]
                        for i, pos in enumerate(agent_positions):
                            screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                            agent_rect = pygame.Rect(screen_x - 10, screen_y - 10, 20, 20) # A small clickable area
                            if agent_rect.collidepoint(mouse_x, mouse_y):
                                self.active_agent_idx = i
                                print(f"Switched control to Agent {i}")
                                break # Stop after finding the first clicked agent

        self.handle_input()
        
        self.handle_audio_events(data)
        self.draw_simulation_data(data)

        pygame.display.flip()
        self.clock.tick(60)




