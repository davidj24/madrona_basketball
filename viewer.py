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



# ================================ Config Constants ================================
WINDOW_WIDTH = 3000
WINDOW_HEIGHT = 1750
BACKGROUND_COLOR = (50, 50, 50)  # Dark gray
TEXT_COLOR = (255, 255, 255)     # White

# World/court dimensions in meters (NBA standard)
NBA_COURT_WIDTH = 28.65
NBA_COURT_HEIGHT = 15.24
NBA_3_POINT_LINE = 7.24 # The 3 point line in the corner is a stright line,and the distnace is shorter, roughly 6.71m
NBA_KEY_WIDTH = 5.79
NBA_KEY_HEIGHT = 4.88
NBA_TOP_OF_KEY_RADIUS = 1.22
NBA_HALFCOURT_CIRCLE_RADIUS = 1.33

COURT_MARGIN_FACTOR = 1.1  # World area will be 1.1x the court area
PIXELS_PER_METER = 80      # This is the single source of truth for scaling. Change this to zoom in/out.


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
    scalar = q[0]
    pure = np.array([q[1], q[2], q[3]])
    v_vec = np.asarray(v)
    pure_x_v = np.cross(pure, v_vec)
    pure_x_pure_x_v = np.cross(pure, pure_x_v)
    return v_vec + 2.0 * ((pure_x_v * scalar) + pure_x_pure_x_v)

class MadronaPipeline:
    """
    Simple pipeline that connects to your Madrona simulation and displays the data
    """
    
    def handle_audio_events(self, data):
        """Checks for new audio events from the simulation and plays sounds."""
        if data is None or 'game_state' not in data:
            return

        game_state = data['game_state'][0]
        
        # NOTE: These indices MUST match the order in your C++ GameState struct.
        # Based on our previous discussion, we assume scoreCount is the 11th field (index 10)
        # and outOfBoundsCount is the 12th field (index 11).
        # Adjust these indices if your struct is different.
        current_score_count = int(game_state[12])
        current_oob_count = int(game_state[13])

        # Check if the score count has increased since the last frame
        if self.score_sound and current_score_count > self.last_score_count:
            self.score_sound.play()
        
        # Check if the out-of-bounds count has increased
        if self.whistle_sound and current_oob_count > self.last_oob_count:
            self.whistle_sound.play()

        # Update the last known counts for the next frame
        self.last_score_count = current_score_count
        self.last_oob_count = current_oob_count

    def __init__(self):
        pygame.init()
        pygame.mixer.init()  # Initialize the audio mixer

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Madrona Simulation Pipeline")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # --- Load Sound Effects ---
        try:
            self.score_sound = pygame.mixer.Sound("assets/swish.wav")
            self.whistle_sound = pygame.mixer.Sound("assets/whistle.wav")
            print("✓ Audio files loaded successfully.")
        except pygame.error as e:
            print(f"⚠ Warning: Could not load audio files from 'assets/' folder. Error: {e}")
            self.score_sound = None
            self.whistle_sound = None

        # Keep track of the last count to detect when a new event occurs
        self.last_score_count = 0
        self.last_oob_count = 0
        
        # --- Centralized Coordinate System Setup (your existing code is preserved) ---
        self.pixels_per_meter = PIXELS_PER_METER
        margin_factor = 1.10
        NBA_COURT_WIDTH = 28.65
        NBA_COURT_HEIGHT = 15.24
        self.world_width_meters = NBA_COURT_WIDTH * margin_factor
        self.world_height_meters = NBA_COURT_HEIGHT * margin_factor
        self.world_width_px = self.world_width_meters * self.pixels_per_meter
        self.world_height_px = self.world_height_meters * self.pixels_per_meter
        self.world_offset_x = (WINDOW_WIDTH - self.world_width_px) / 2
        self.world_offset_y = (WINDOW_HEIGHT - self.world_height_px) / 2

        print("Initializing Madrona simulation...")
        
        import math
        self.world_discrete_width = math.ceil(self.world_width_meters)
        self.world_discrete_height = math.ceil(self.world_height_meters)
        walls = np.zeros((self.world_discrete_height, self.world_discrete_width), dtype=bool)
        rewards = np.zeros((self.world_discrete_height, self.world_discrete_width), dtype=float)
        
        self.sim = madrona_sim.SimpleGridworldSimulator(
            walls=walls,
            rewards=rewards, 
            start_x=self.world_discrete_width / 2.0,
            start_y=self.world_discrete_height / 2.0,
            max_episode_length=10000,
            exec_mode=ExecMode.CPU,
            num_worlds=1,
            gpu_id=-1
        )
        self.world_width_px = self.world_discrete_width * self.pixels_per_meter
        self.world_height_px = self.world_discrete_height * self.pixels_per_meter
        self.world_offset_x = (WINDOW_WIDTH - self.world_width_px) / 2
        self.world_offset_y = (WINDOW_HEIGHT - self.world_height_px) / 2
        print(f"✓ Madrona simulation initialized! World size: {self.world_discrete_width}x{self.world_discrete_height} meters (discrete grid)")
        self.step_count = 0
        
    def get_simulation_data(self):
        """Get the current state from your Madrona simulation"""
        try:
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
            game_state_tensor = self.sim.game_state_tensor()
            orientation_tensor = self.sim.orientation_tensor()
            
            return {
                'observations': obs_tensor.to_torch().detach().cpu().numpy(),
                'agent_teams': agent_team_tensor.to_torch().detach().cpu().numpy(),
                'actions': action_tensor.to_torch().detach().cpu().numpy(),
                'rewards': reward_tensor.to_torch().detach().cpu().numpy(),
                'done': done_tensor.to_torch().detach().cpu().numpy(),
                'reset': reset_tensor.to_torch().detach().cpu().numpy(),
                'basketball_pos': basketball_pos_tensor.to_torch().detach().cpu().numpy(),
                'ball_physics' : ball_physics_tensor.to_torch().detach().cpu().numpy(),
                'hoop_pos' : hoop_pos_tensor.to_torch().detach().cpu().numpy(),
                'agent_possession': agent_possession_tensor.to_torch().detach().cpu().numpy(),
                'ball_grabbed': ball_grabbed_tensor.to_torch().detach().cpu().numpy(),
                'agent_entity_ids': agent_entity_id_tensor.to_torch().detach().cpu().numpy(),
                'ball_entity_ids': ball_entity_id_tensor.to_torch().detach().cpu().numpy(),
                'orientation': orientation_tensor.to_torch().detach().cpu().numpy(),
                'game_state': game_state_tensor.to_torch().detach().cpu().numpy()
            }
        except Exception as e:
            print(f"Error getting simulation data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def draw_basketball_court(self):
        """Draws a basketball court centered in the window, scaled by pixels_per_meter, in horizontal (landscape) orientation."""
        # --- NBA standard dimensions (meters) ---
        COURT_LENGTH = 28.65  # x-axis
        COURT_WIDTH = 15.24   # y-axis
        KEY_WIDTH = 4.88      # y-axis
        KEY_HEIGHT = 5.79     # x-axis
        FT_CIRCLE_RADIUS = 1.80
        HALFCOURT_CIRCLE_RADIUS = 1.33
        THREE_PT_RADIUS = 7.24
        THREE_PT_LINE_DIST = 6.71
        BACKBOARD_TO_BASELINE = 1.22
        BASKET_DIAMETER = 0.45
        RESTRICTED_RADIUS = 1.22
        HOOP_CENTER_FROM_BASELINE = 1.575
        LINE_THICKNESS = 3

        scale = self.pixels_per_meter

        # Calculate court dimensions in pixels
        court_length_px = COURT_LENGTH * scale
        court_width_px = COURT_WIDTH * scale
        court_offset_x = self.world_offset_x + (self.world_width_px - court_length_px) / 2
        court_offset_y = self.world_offset_y + (self.world_height_px - court_width_px) / 2
        court_rect = pygame.Rect(court_offset_x, court_offset_y, court_length_px, court_width_px)

        # Draw court background
        pygame.draw.rect(self.screen, (205, 133, 63), court_rect)
        playable_rect = court_rect.inflate(-LINE_THICKNESS * 2, -LINE_THICKNESS * 2)
        pygame.draw.rect(self.screen, (20, 40, 80), playable_rect)

        # Center line (vertical, at half court)
        center_x = court_rect.left + court_rect.width / 2
        pygame.draw.line(self.screen, (255, 255, 255), (center_x, court_rect.top), (center_x, court_rect.bottom), LINE_THICKNESS)

        # Halfcourt circle
        center_y = court_rect.top + court_rect.height / 2
        halfcourt_radius_px = HALFCOURT_CIRCLE_RADIUS * scale
        pygame.draw.circle(self.screen, (255, 255, 255), (int(center_x), int(center_y)), int(halfcourt_radius_px), LINE_THICKNESS)

        # Keys, free throw circles, restricted areas, backboards, hoops, three-point lines
        for side in [-1, 1]:
            # Baseline x
            if side == -1:
                baseline_x = court_rect.left
                key_x = baseline_x
                hoop_x = baseline_x + HOOP_CENTER_FROM_BASELINE * scale
                backboard_x = baseline_x + BACKBOARD_TO_BASELINE * scale
                three_pt_arc_center_x = hoop_x
                arc_restricted_start = math.radians(270)
                arc_restricted_end = math.radians(90)
                arc_ft_start = math.radians(270)
                arc_ft_end = math.radians(90)
                arc_3pt_start = math.radians(292)
                arc_3pt_end = math.radians(68)
                three_pt_line_x = court_rect.left
                arc_sign = 1
            else:
                baseline_x = court_rect.right
                key_x = baseline_x - KEY_HEIGHT * scale
                hoop_x = baseline_x - HOOP_CENTER_FROM_BASELINE * scale
                backboard_x = baseline_x - BACKBOARD_TO_BASELINE * scale
                three_pt_arc_center_x = hoop_x
                arc_restricted_start = math.radians(90)
                arc_restricted_end = math.radians(270)
                arc_ft_start = math.radians(90)
                arc_ft_end = math.radians(270)
                arc_3pt_start = math.radians(112)
                arc_3pt_end = math.radians(248)
                three_pt_line_x = court_rect.right
                arc_sign = -1

            # Key (paint)
            key_top = center_y - KEY_WIDTH / 2 * scale
            key_rect = pygame.Rect(key_x if side == -1 else key_x, key_top, KEY_HEIGHT * scale, KEY_WIDTH * scale)
            pygame.draw.rect(self.screen, (139, 0, 0), key_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), key_rect, LINE_THICKNESS)

            # Free throw semicircle (only the half facing center court)
            ft_circle_center = (key_x + KEY_HEIGHT * scale if side == -1 else key_x, center_y)
            ft_circle_rect = pygame.Rect(
                ft_circle_center[0] - FT_CIRCLE_RADIUS * scale,
                center_y - FT_CIRCLE_RADIUS * scale,
                2 * FT_CIRCLE_RADIUS * scale,
                2 * FT_CIRCLE_RADIUS * scale)
            pygame.draw.arc(self.screen, (255, 255, 255), ft_circle_rect, arc_ft_start, arc_ft_end, LINE_THICKNESS)

            # Restricted area semicircle (only the half facing center court)
            restricted_rect = pygame.Rect(
                hoop_x - RESTRICTED_RADIUS * scale,
                center_y - RESTRICTED_RADIUS * scale,
                2 * RESTRICTED_RADIUS * scale,
                2 * RESTRICTED_RADIUS * scale)
            pygame.draw.arc(self.screen, (255, 255, 255), restricted_rect, arc_restricted_start, arc_restricted_end, LINE_THICKNESS)

            # Backboard
            pygame.draw.line(self.screen, (255, 255, 255), (backboard_x, center_y - 0.91 * scale), (backboard_x, center_y + 0.91 * scale), LINE_THICKNESS)

            # Hoop
            pygame.draw.circle(self.screen, (255, 140, 0), (int(hoop_x), int(center_y)), int(BASKET_DIAMETER * scale / 2), 0)
            pygame.draw.circle(self.screen, (220, 20, 20), (int(hoop_x), int(center_y)), int(BASKET_DIAMETER * scale / 2), LINE_THICKNESS)

            # Three-point arc
            arc_rect = pygame.Rect(three_pt_arc_center_x - THREE_PT_RADIUS * scale, center_y - THREE_PT_RADIUS * scale, 2 * THREE_PT_RADIUS * scale, 2 * THREE_PT_RADIUS * scale)
            pygame.draw.arc(self.screen, (255, 255, 255), arc_rect, arc_3pt_start, arc_3pt_end, LINE_THICKNESS)

            # Three-point straight lines (corners) - extend to meet arc smoothly
            y1 = center_y - THREE_PT_LINE_DIST * scale
            y2 = center_y + THREE_PT_LINE_DIST * scale
            # Find arc endpoints for smooth join
            arc_y1 = center_y - math.sin(math.radians(68)) * THREE_PT_RADIUS * scale if side == -1 else center_y - math.sin(math.radians(112)) * THREE_PT_RADIUS * scale
            arc_y2 = center_y + math.sin(math.radians(68)) * THREE_PT_RADIUS * scale if side == -1 else center_y + math.sin(math.radians(112)) * THREE_PT_RADIUS * scale
            if side == -1:
                pygame.draw.line(self.screen, (255, 255, 255), (three_pt_line_x, y1), (three_pt_arc_center_x + math.cos(math.radians(68)) * THREE_PT_RADIUS * scale, arc_y1))
                pygame.draw.line(self.screen, (255, 255, 255), (three_pt_line_x, y2), (three_pt_arc_center_x + math.cos(math.radians(68)) * THREE_PT_RADIUS * scale, arc_y2))
            else:
                pygame.draw.line(self.screen, (255, 255, 255), (three_pt_line_x, y1), (three_pt_arc_center_x - math.cos(math.radians(68)) * THREE_PT_RADIUS * scale, arc_y1))
                pygame.draw.line(self.screen, (255, 255, 255), (three_pt_line_x, y2), (three_pt_arc_center_x - math.cos(math.radians(68)) * THREE_PT_RADIUS * scale, arc_y2))

        # Draw world border (larger than the court)
        world_border_rect = pygame.Rect(self.world_offset_x, self.world_offset_y, self.world_width_px, self.world_height_px)
        pygame.draw.rect(self.screen, (255, 255, 255), world_border_rect, 3)

    def draw_score_display(self, data):
        """Draw the team scores and game info at the bottom center of the screen"""
        if data is None or 'game_state' not in data:
            return

        game_state = data['game_state'][0]
        
        # --- CORRECTED INDICES ---
        # These now match the C++ GameState struct order
        period = int(game_state[4])
        team0_score = int(game_state[7])
        team1_score = int(game_state[9])
        game_clock = float(game_state[10])
        shot_clock = float(game_state[11])

        # Get team colors - this part of your code is fine
        team_colors = { 0: (0, 100, 255), 1: (255, 50, 50) }
        if 'agent_teams' in data:
            agent_teams = data['agent_teams'][0]
            for i, team_data in enumerate(agent_teams):
                if len(team_data) >= 4:
                    team_index = int(team_data[0])
                    if team_index not in team_colors:
                        import struct
                        color_r = max(0, min(255, int(struct.unpack('f', struct.pack('I', np.uint32(team_data[1])))[0])))
                        color_g = max(0, min(255, int(struct.unpack('f', struct.pack('I', np.uint32(team_data[2])))[0])))
                        color_b = max(0, min(255, int(struct.unpack('f', struct.pack('I', np.uint32(team_data[3])))[0])))
                        team_colors[team_index] = (color_r, color_g, color_b)

        # Score display dimensions and positioning (your code is fine)
        display_width = 600
        display_height = 120
        display_x = (WINDOW_WIDTH - display_width) // 2
        display_y = WINDOW_HEIGHT - display_height - 50

        # Draw main score display background (your code is fine)
        score_bg_rect = pygame.Rect(display_x, display_y, display_width, display_height)
        pygame.draw.rect(self.screen, (30, 30, 30), score_bg_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), score_bg_rect, 3)

        # Team score sections (your code is fine)
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

        # Fonts (your code is fine)
        score_font = pygame.font.Font(None, 64)
        label_font = pygame.font.Font(None, 24)
        time_font = pygame.font.Font(None, 36)

        # Draw team scores (your code is fine)
        team0_score_text = score_font.render(str(team0_score), True, (255, 255, 255))
        team0_score_rect = team0_score_text.get_rect(center=(team0_rect.centerx, team0_rect.centery - 10))
        self.screen.blit(team0_score_text, team0_score_rect)
        team1_score_text = score_font.render(str(team1_score), True, (255, 255, 255))
        team1_score_rect = team1_score_text.get_rect(center=(team1_rect.centerx, team1_rect.centery - 10))
        self.screen.blit(team1_score_text, team1_score_rect)

        # Draw team labels (your code is fine)
        team0_label = label_font.render("TEAM 0", True, (255, 255, 255))
        team0_label_rect = team0_label.get_rect(center=(team0_rect.centerx, team0_rect.bottom - 15))
        self.screen.blit(team0_label, team0_label_rect)
        team1_label = label_font.render("TEAM 1", True, (255, 255, 255))
        team1_label_rect = team1_label.get_rect(center=(team1_rect.centerx, team1_rect.bottom - 15))
        self.screen.blit(team1_label, team1_label_rect)

        # Draw period, game clock, and shot clock (your code is fine)
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
        if data is None: return
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_basketball_court()
        self.draw_score_display(data)
        y_offset = 20
        info_texts = [f"Madrona Basketball Simulation - Step {self.step_count}", f"World Size: {self.world_width_meters:.1f}x{self.world_height_meters:.1f} meters", "", "Controls: WASD/Arrow Keys, SPACE=manual step, R=reset, ESC=quit"]
        if 'actions' in data:
            for i, action_components in enumerate(data['actions'][0]):
                if len(action_components) >= 8:
                    info_texts.append(f"Agent {i}: Speed={int(action_components[0])} Angle={int(action_components[1])} Rotate={int(action_components[2])} Grab={int(action_components[3])} Pass={int(action_components[4])} Shoot={int(action_components[5])} Steal={int(action_components[6])} Contest={int(action_components[7])}")
                elif len(action_components) >= 6:
                    info_texts.append(f"Agent {i}: Speed={int(action_components[0])} Angle={int(action_components[1])} Rotate={int(action_components[2])} Grab={int(action_components[3])} Pass={int(action_components[4])} Shoot={int(action_components[5])}")
        if 'rewards' in data:
            for i, reward in enumerate(data['rewards'][0]): info_texts.append(f"Agent {i} Reward: {reward:.2f}")
        if 'done' in data:
            for i, done in enumerate(data['done'][0]): info_texts.append(f"Agent {i} Done: {done}")
        if 'basketball_pos' in data:
            for i, pos in enumerate(data['basketball_pos'][0]):
                screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                pygame.draw.circle(self.screen, (255, 100, 0), (screen_x, screen_y), 12)
                pygame.draw.circle(self.screen, (200, 50, 0), (screen_x, screen_y), 12, 2)
                self.screen.blit(pygame.font.Font(None, 16).render(f"B{i + 1}", True, (255, 255, 255)), (screen_x - 8, screen_y - 5))
        if 'hoop_pos' in data:
            for i, pos in enumerate(data['hoop_pos'][0]):
                screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                pygame.draw.line(self.screen, (255, 255, 255), (screen_x - 2, screen_y - 15), (screen_x - 2, screen_y + 15), 4)
                pygame.draw.circle(self.screen, (255, 140, 0), (screen_x, screen_y), 10, 0)
                pygame.draw.circle(self.screen, (220, 20, 20), (screen_x, screen_y), 10, 3)
                self.screen.blit(pygame.font.Font(None, 16).render(f"H{i + 1}", True, (255, 255, 255)), (screen_x - 8, screen_y + 15))
        if 'observations' in data and 'agent_teams' in data and 'orientation' in data:
            positions, team_data, orientations = data['observations'][0], data['agent_teams'][0], data['orientation'][0]
            team_colors = { 0: (0, 100, 255), 1: (255, 50, 50) }
            for i, pos in enumerate(positions):
                screen_x, screen_y = self.meters_to_screen(pos[0], pos[1])
                team_index = int(team_data[i][0]) if i < len(team_data) else 0
                agent_color = team_colors.get(team_index, (128, 128, 128))
                pygame.draw.rect(self.screen, agent_color, (screen_x - 8, screen_y - 8, 16, 16))
                pygame.draw.rect(self.screen, (255, 255, 255), (screen_x - 8, screen_y - 8, 16, 16), 2)
                font_small = pygame.font.Font(None, 16)
                self.screen.blit(font_small.render(f"A{i + 1}", True, (255, 255, 255)), (screen_x - 8, screen_y - 20))
                self.screen.blit(font_small.render(f"T{team_index}", True, (255, 255, 255)), (screen_x - 8, screen_y + 10))
                q = orientations[i]
                BASE_FORWARD_VECTOR = np.array([0.0, 2.0, 0.0])
                direction_3d = rotate_vec(q, BASE_FORWARD_VECTOR)
                dx, dy = direction_3d[0], direction_3d[1]
                arrow_length_pixels = 0.2 * self.pixels_per_meter
                arrow_end = (screen_x + arrow_length_pixels * dx, screen_y + arrow_length_pixels * dy)
                pygame.draw.line(self.screen, (255, 255, 0), (screen_x, screen_y), arrow_end, 3)
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
        try:
            self.sim.trigger_reset(0)
            print(f"Simulation reset at step {self.step_count}")
            self.step_count = 0
        except Exception as e: print(f"Error resetting simulation: {e}")
    
    def handle_input(self):
        keys = pygame.key.get_pressed()
        move_speed, move_angle, rotate, grab, pass_ball, shoot_ball = 0,0,0,0,0,0
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
        move_speed1, move_angle1, rotate1, grab1, pass_ball1, shoot_ball1 = 0,0,0,0,0,0
        if keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
            move_speed1 = 1
            if keys[pygame.K_UP] and keys[pygame.K_RIGHT]: move_angle1 = 1
            elif keys[pygame.K_RIGHT] and keys[pygame.K_DOWN]: move_angle1 = 3
            elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]: move_angle1 = 5
            elif keys[pygame.K_LEFT] and keys[pygame.K_UP]: move_angle1 = 7
            elif keys[pygame.K_UP]: move_angle1 = 0
            elif keys[pygame.K_RIGHT]: move_angle1 = 2
            elif keys[pygame.K_DOWN]: move_angle1 = 4
            elif keys[pygame.K_LEFT]: move_angle1 = 6
        if keys[pygame.K_COMMA]: rotate1 = -1
        elif keys[pygame.K_PERIOD]: rotate1 = 1
        if keys[pygame.K_KP0]: grab1 = 1
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]: pass_ball1 = 1
        self.sim.set_action(0, 0, move_speed, move_angle, rotate, grab, pass_ball, shoot_ball)
        self.sim.set_action(0, 1, move_speed1, move_angle1, rotate1, grab1, pass_ball1, shoot_ball1)

    def run(self):
        running, auto_step = True, True
        print("Madrona Pipeline Started!\n- Press F to toggle auto-stepping | R to reset | ESC to quit")
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: self.reset_simulation()
                    if event.key == pygame.K_f: auto_step = not auto_step
            self.handle_input()
            if auto_step: self.step_simulation()
            
            data = self.get_simulation_data()
            self.handle_audio_events(data)  # Call the new audio handler
            self.draw_simulation_data(data)

            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        print(f"Pipeline ended after {self.step_count} steps")

if __name__ == "__main__":
    try:
        pipeline = MadronaPipeline()
        pipeline.run()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)