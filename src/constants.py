# Python constants that mirror constants.hpp
# These should be kept in sync with the C++ constants

# ======================================== Entity Counts ========================================
NUM_AGENTS = 2
NUM_BASKETBALLS = 1
NUM_HOOPS = 2

# ======================================== Simulation Parameters ========================================
SIMULATION_HZ = 62.0  # How many timesteps are in one second
TIMESTEPS_TO_SECONDS_FACTOR = 1.0 / SIMULATION_HZ
TIME_PER_PERIOD = 300.0  # 5 minutes per quarter (in seconds)



# ======================================== Gameplay Constants ========================================
HOOP_SCORE_ZONE_SIZE = 0.1  # Radius for scoring detection
IN_COURT_OFFSET = 0.1  # Buffer to ensure players are placed inside court lines
SHOT_CLOCK_DURATION = 24.0  # Shot clock in seconds


# ======================================== Agent Properties ========================================
AGENT_SIZE_M = 0.25  # Agent visual size for rendering
AGENT_SHOULDER_WIDTH = 0.4290 # Average shoulder width of agent
AGENT_DEPTH = .1
AGENT_ORIENTATION_ARROW_LENGTH_M = 0.5  # Length of orientation arrow
EVENT_DEFINITIONS = {
    "shoot" : {
        "action_idx" : 5,
        "conditions" : lambda log_data_tensor, step_num, world_num, agent_num=0: int(log_data_tensor.get('ball_physics')[step_num, world_num, 0][0]) == 1 and int(log_data_tensor.get('ball_physics')[step_num-1, world_num, 0][0]) == 0,
        "outcome_func" : lambda log_data, step_num, world_num: log_data.get('ball_physics')[step_num, world_num, 0][6] == 1,
        "visuals" : {
            True : {"shape" : "circle", "color" : (0, 255, 0), "size" : 7},
            False : {"shape" : "x", "color" : (255, 0, 0), "size" : 5},
        }
    },

    "pass" : {
        "action_idx" : 4,
        "conditions" : lambda log_data_tensor, step_num, world_num, agent_num=0: (int(log_data_tensor.get('agent_possession')[step_num-1, world_num, agent_num, 0]) == 1) and #Agent had the ball
                                                                                 (step_num+1 < len(log_data_tensor.get('ball_vel')) and # there is a next step
                                                                                 (abs(log_data_tensor.get('ball_vel')[step_num+1, world_num, 0][0]) > 0.001 or # ball has x velocity on next step
                                                                                  abs(log_data_tensor.get('ball_vel')[step_num+1, world_num, 0][1]) > 0.001)) and # ball has y velocity on next step
                                                                                  int(log_data_tensor.get('ball_physics')[step_num+1, world_num, 0][0]) == 0, # ball is not in flight
        "outcome_func" : lambda log_data, step_num, world_num: True,
        "visuals" : {
            True : {"shape" : "circle", "color" : (0, 255, 0), "size" : 7},
        }
    },

    "grab" : {
        "action_idx" : 3,
        "conditions" : lambda log_data_tensor, step_num, world_num, agent_num=0: (int(log_data_tensor.get('agent_possession')[step_num, world_num, agent_num, 0]) != int(log_data_tensor.get('agent_possession')[step_num-1, world_num, agent_num, 0])),  # Possession state changed
        "outcome_func" : lambda log_data, step_num, world_num: True,
        "visuals" : {
            True : {"shape" : "circle", "color" : (0, 255, 0), "size" : 7},
        }
    }
}






# ======================================== Basketball Physical Properties ========================================
BALL_DIAMETER_M = 0.242  # Official basketball diameter
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0
BALL_CIRCUMFERENCE_M = 0.749  # Official basketball circumference







# ======================================== Basketball Court Dimensions (NBA Standard) ========================================
COURT_LENGTH_M = 28.65  # Full court length
COURT_WIDTH_M = 15.24   # Full court width

# World dimensions (court + margin)
WORLD_MARGIN_FACTOR = 1.1
WORLD_WIDTH_M = COURT_LENGTH_M * WORLD_MARGIN_FACTOR
WORLD_HEIGHT_M = COURT_WIDTH_M * WORLD_MARGIN_FACTOR

# Court positioning within world
COURT_MIN_X = (WORLD_WIDTH_M - COURT_LENGTH_M) / 2.0
COURT_MAX_X = COURT_MIN_X + COURT_LENGTH_M
COURT_MIN_Y = (WORLD_HEIGHT_M - COURT_WIDTH_M) / 2.0
COURT_MAX_Y = COURT_MIN_Y + COURT_WIDTH_M

# Court features
KEY_WIDTH_M = 4.88   # Width of the paint/key
KEY_HEIGHT_M = 5.79  # Height of the paint/key (from baseline to free throw line)
HOOP_FROM_BASELINE_M = 1.575  # Distance from baseline to center of hoop
FREE_THROW_CIRCLE_RADIUS_M = 1.8
CENTER_CIRCLE_RADIUS_M = 1.8
TOP_OF_KEY_RADIUS_M = 1.22
HALFCOURT_CIRCLE_RADIUS_M = 1.33

# 3-Point Line
ARC_RADIUS_M = 7.24  # 3-point arc radius
CORNER_3_FROM_SIDELINE_M = 0.91  # Distance from sideline to corner 3-point line
CORNER_3_LENGTH_FROM_BASELINE_M = 4.27  # Length of corner 3-point line

# Backboard and rim
BACKBOARD_WIDTH_M = 1.829
RIM_DIAMETER_M = 0.4572
BACKBOARD_OFFSET_FROM_HOOP_M = (HOOP_FROM_BASELINE_M - 1.22)  # Distance from hoop center to backboard



# ======================================== Rendering & Scaling ========================================
PIXELS_PER_METER = 54.0
# PIXELS_PER_METER = 110
WINDOW_WIDTH = PIXELS_PER_METER * 32.3
WINDOW_HEIGHT = PIXELS_PER_METER * 18.2
BACKGROUND_COLOR = (50, 50, 50)  # Dark gray
TEXT_COLOR = (255, 255, 255)     # White
TEAM0_COLOR = (0, 100, 255)
TEAM1_COLOR = (128, 0, 128)