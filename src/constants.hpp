#pragma once

namespace madsimple {
    // ======================================== Entity Counts ========================================
    constexpr int32_t NUM_AGENTS = 10;
    constexpr int32_t NUM_BASKETBALLS = 1;
    constexpr int32_t NUM_HOOPS = 2;
    constexpr uint32_t ENTITY_ID_PLACEHOLDER = UINT32_MAX;  // Use max value as invalid/null entity ID

    // ======================================== Simulation Parameters ========================================
    constexpr float SIMULATION_HZ = 62.0f; // How many timesteps are in one second
    constexpr float TIMESTEPS_TO_SECONDS_FACTOR = 1.0f / SIMULATION_HZ;
    constexpr float TIME_PER_PERIOD = 300.f; // 5 minutes per quarter (in seconds)

    // ======================================== Rendering & Scaling ========================================
    constexpr float PIXELS_PER_METER = 110.f; // Single source of truth for visualization scaling


        // ======================================== Gameplay Constants ========================================
    constexpr float HOOP_SCORE_ZONE_SIZE = 0.1f; // Radius for scoring detection
    constexpr float IN_COURT_OFFSET = 0.1f; // Buffer to ensure players are placed inside court lines
    constexpr float SHOT_CLOCK_DURATION = 24.0f; // Shot clock in seconds
    


    // ======================================== Basketball Physical Properties ========================================
    constexpr float BALL_DIAMETER_M = 0.242f; // Official basketball diameter
    constexpr float BALL_RADIUS_M = BALL_DIAMETER_M / 2.0f;
    constexpr float BALL_CIRCUMFERENCE_M = 0.749f; // Official basketball circumference
    


    // ======================================== Agent Properties ========================================
    constexpr float AGENT_SIZE_M = 0.2f; // Agent visual size for rendering
    constexpr float AGENT_ORIENTATION_ARROW_LENGTH_M = 0.5f; // Length of orientation arrow
    constexpr uint32_t NUM_OBSERVATIONS_PER_AGENT = 10; 
    
    // Movement
    constexpr float ANGLE_BETWEEN_DIRECTIONS = madrona::math::pi / 4.0f; // π/4 for 8-directional movement




    // ======================================== Basketball Court Dimensions (NBA Standard) ========================================
    constexpr float COURT_LENGTH_M = 28.65f; // Full court length
    constexpr float COURT_WIDTH_M = 15.24f;  // Full court width
    
    // World dimensions (court + margin)
    constexpr float WORLD_MARGIN_FACTOR = 1.1f;
    constexpr float WORLD_WIDTH_M = COURT_LENGTH_M * WORLD_MARGIN_FACTOR;
    constexpr float WORLD_HEIGHT_M = COURT_WIDTH_M * WORLD_MARGIN_FACTOR;
    
    // Court positioning within world
    constexpr float COURT_MIN_X = (WORLD_WIDTH_M - COURT_LENGTH_M) / 2.0f;
    constexpr float COURT_MAX_X = COURT_MIN_X + COURT_LENGTH_M;
    constexpr float COURT_MIN_Y = (WORLD_HEIGHT_M - COURT_WIDTH_M) / 2.0f;
    constexpr float COURT_MAX_Y = COURT_MIN_Y + COURT_WIDTH_M;
    
    // Court features
    constexpr float KEY_WIDTH_M = 4.88f;  // Width of the paint/key
    constexpr float KEY_HEIGHT_M = 5.79f; // Height of the paint/key (from baseline to free throw line)
    constexpr float HOOP_FROM_BASELINE_M = 1.575f; // Distance from baseline to center of hoop
    constexpr float FREE_THROW_CIRCLE_RADIUS_M = 1.8f;
    constexpr float CENTER_CIRCLE_RADIUS_M = 1.8f;
    constexpr float TOP_OF_KEY_RADIUS_M = 1.22f;
    constexpr float HALFCOURT_CIRCLE_RADIUS_M = 1.33f;
    
    // 3-Point Line
    constexpr float ARC_RADIUS_M = 7.24f; // 3-point arc radius
    constexpr float CORNER_3_FROM_SIDELINE_M = 0.91f; // Distance from sideline to corner 3-point line
    constexpr float CORNER_3_LENGTH_FROM_BASELINE_M = 4.27f; // Length of corner 3-point line
    
    // Backboard and rim
    constexpr float BACKBOARD_WIDTH_M = 1.829f;
    constexpr float RIM_DIAMETER_M = 0.4572f;
    constexpr float BACKBOARD_OFFSET_FROM_HOOP_M = (HOOP_FROM_BASELINE_M - 1.22f); // Distance from hoop center to backboard
}
