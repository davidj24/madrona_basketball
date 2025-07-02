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
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Madrona Simulation Pipeline")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # --- Centralized Coordinate System Setup ---
        self.pixels_per_meter = PIXELS_PER_METER

        # --- Court dimensions (do not change these) ---
        court_width_m = 28.65
        court_height_m = 15.24

        # --- World is slightly larger than the court ---
        margin_factor = 1.10  # 10% larger in each dimension
        self.world_width_meters = court_width_m * margin_factor
        self.world_height_meters = court_height_m * margin_factor

        # Calculate all pixel dimensions from the meter values and the single scale factor
        self.world_width_px = self.world_width_meters * self.pixels_per_meter
        self.world_height_px = self.world_height_meters * self.pixels_per_meter
        
        # Calculate the offset needed to center the entire world in the window
        self.world_offset_x = (WINDOW_WIDTH - self.world_width_px) / 2
        self.world_offset_y = (WINDOW_HEIGHT - self.world_height_px) / 2

        print("Initializing Madrona simulation...")
        
        # Use math.ceil to ensure the grid fully covers the world size
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
        # For all world border and coordinate calculations, use the discrete grid size
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
        """Draws a basketball court centered in the window, scaled by pixels_per_meter."""
        # --- NBA standard dimensions (meters) ---
        NBA_COURT_W = 28.65
        NBA_COURT_H = 15.24
        KEY_W = 4.88
        KEY_H = 5.79
        FT_CIRCLE_R = 1.8
        THREE_PT_R = 6.75
        HOOP_OFFSET = 1.575
        LINE_THICKNESS = 3

        # --- Use the single scaling factor from __init__ ---
        scale = self.pixels_per_meter
        
        # --- Calculate court dimensions and center it in the world view ---
        court_width_px = NBA_COURT_W * scale
        court_height_px = NBA_COURT_H * scale
        court_offset_x = self.world_offset_x + (self.world_width_px - court_width_px) / 2
        court_offset_y = self.world_offset_y + (self.world_height_px - court_height_px) / 2
        
        court_rect = pygame.Rect(court_offset_x, court_offset_y, court_width_px, court_height_px)
        
        # --- Draw Court Background ---
        pygame.draw.rect(self.screen, (205, 133, 63), court_rect)
        playable_rect = court_rect.inflate(-LINE_THICKNESS * 2, -LINE_THICKNESS * 2)
        pygame.draw.rect(self.screen, (20, 40, 80), playable_rect)

        # --- Draw Court Markings ---
        center_x, center_y = court_rect.centerx, court_rect.centery
        pygame.draw.line(self.screen, (255, 255, 255), (center_x, court_rect.top), (center_x, court_rect.bottom), LINE_THICKNESS)
        center_circle_radius = int(3.0 * scale)
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), center_circle_radius, LINE_THICKNESS)
        pygame.draw.circle(self.screen, (139, 0, 0), (center_x, center_y), center_circle_radius - LINE_THICKNESS)

        # --- Draw Features for Both Halves ---
        for side in [-1, 1]:
            key_w = KEY_W * scale
            key_h = KEY_H * scale
            key_x = court_rect.left if side == -1 else court_rect.right - key_w
            key_y = center_y - key_h / 2
            key_rect = pygame.Rect(key_x, key_y, key_w, key_h)
            pygame.draw.rect(self.screen, (139, 0, 0), key_rect)

            ft_circle_radius = int(FT_CIRCLE_R * scale)
            ft_center_x = key_x + key_w if side == -1 else key_x
            arc_rect = pygame.Rect(ft_center_x - ft_circle_radius, center_y - ft_circle_radius, ft_circle_radius * 2, ft_circle_radius * 2)
            start_angle, end_angle = (-np.pi / 2, np.pi / 2) if side == -1 else (np.pi / 2, 3 * np.pi / 2)
            pygame.draw.arc(self.screen, (255, 255, 255), arc_rect, start_angle, end_angle, LINE_THICKNESS)

            hoop_center_x = court_rect.left + HOOP_OFFSET * scale if side == -1 else court_rect.right - HOOP_OFFSET * scale
            three_pt_radius = int(THREE_PT_R * scale)
            three_pt_arc_rect = pygame.Rect(hoop_center_x - three_pt_radius, center_y - three_pt_radius, three_pt_radius * 2, three_pt_radius * 2)
            pygame.draw.arc(self.screen, (255, 255, 255), three_pt_arc_rect, start_angle, end_angle, LINE_THICKNESS)
        
        # --- Draw World Border (larger than the court) ---
        world_border_rect = pygame.Rect(self.world_offset_x, self.world_offset_y, self.world_width_px, self.world_height_px)
        pygame.draw.rect(self.screen, (255, 255, 255), world_border_rect, 3)

    def draw_score_display(self, data):
        """Draw the team scores and game info at the bottom center of the screen"""
        if data is None or 'game_state' not in data: return
        game_state = data['game_state'][0]
        team0_score, team1_score = int(game_state[5]), int(game_state[7])
        game_clock, shot_clock, period = float(game_state[8]), float(game_state[9]), int(game_state[2])
        team_colors = { 0: (0, 100, 255), 1: (255, 50, 50) }
        display_width, display_height = 600, 120
        display_x, display_y = (WINDOW_WIDTH - display_width) // 2, WINDOW_HEIGHT - display_height - 50
        score_bg_rect = pygame.Rect(display_x, display_y, display_width, display_height)
        pygame.draw.rect(self.screen, (30, 30, 30), score_bg_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), score_bg_rect, 3)
        team_section_width = display_width // 3
        team0_rect = pygame.Rect(display_x, display_y, team_section_width, display_height)
        team1_rect = pygame.Rect(display_x + 2 * team_section_width, display_y, team_section_width, display_height)
        middle_rect = pygame.Rect(display_x + team_section_width, display_y, team_section_width, display_height)
        pygame.draw.rect(self.screen, team_colors.get(0, (0,100,255)), team0_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), team0_rect, 2)
        pygame.draw.rect(self.screen, team_colors.get(1, (255,0,100)), team1_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), team1_rect, 2)
        pygame.draw.rect(self.screen, (60, 60, 60), middle_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), middle_rect, 2)
        score_font = pygame.font.Font(None, 64)
        label_font = pygame.font.Font(None, 24)
        time_font = pygame.font.Font(None, 36)
        team0_score_text = score_font.render(str(team0_score), True, (255, 255, 255))
        team0_score_rect = team0_score_text.get_rect(center=(team0_rect.centerx, team0_rect.centery - 10))
        self.screen.blit(team0_score_text, team0_score_rect)
        team1_score_text = score_font.render(str(team1_score), True, (255, 255, 255))
        team1_score_rect = team1_score_text.get_rect(center=(team1_rect.centerx, team1_rect.centery - 10))
        self.screen.blit(team1_score_text, team1_score_rect)
        team0_label = label_font.render("TEAM 0", True, (255, 255, 255))
        team0_label_rect = team0_label.get_rect(center=(team0_rect.centerx, team0_rect.bottom - 15))
        self.screen.blit(team0_label, team0_label_rect)
        team1_label = label_font.render("TEAM 1", True, (255, 255, 255))
        team1_label_rect = team1_label.get_rect(center=(team1_rect.centerx, team1_rect.bottom - 15))
        self.screen.blit(team1_label, team1_label_rect)
        period_text = time_font.render(f"Q{period}", True, (255, 255, 255))
        period_rect = period_text.get_rect(center=(middle_rect.centerx, middle_rect.top + 25))
        self.screen.blit(period_text, period_rect)
        game_minutes = int(game_clock // 60)
        game_seconds = int(game_clock % 60)
        shot_clock_seconds = int(shot_clock)
        game_time_text = time_font.render(f"{game_minutes:02d}:{game_seconds:02d}", True, (255, 255, 255))
        game_time_rect = game_time_text.get_rect(center=(middle_rect.centerx, middle_rect.centery))
        self.screen.blit(game_time_text, game_time_rect)
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
                arrow_length_pixels = 0.5 * self.pixels_per_meter
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