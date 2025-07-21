#include "game.hpp"
#include "types.hpp"
#include "constants.hpp"
#include "gen.hpp"
#include "helper.hpp"

using namespace madrona;
using namespace madrona::math;



namespace madBasketball {
inline void assignInbounder(Engine &ctx, Entity ball_entity, Position ball_pos, uint32_t new_team_idx, Quat new_orientation, bool is_oob)
{
    GameState &gameState = ctx.singleton<GameState>();
    bool inbounder_assigned = false;

    // Find the first available player on the new team.
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Team &agent_team = ctx.get<Team>(agent);
        Inbounding &inbounding = ctx.get<Inbounding>(agent);
        Position &agent_pos = ctx.get<Position>(agent);
        InPossession &in_possession = ctx.get<InPossession>(agent);
        Orientation &agent_orient = ctx.get<Orientation>(agent);
        if ((uint32_t)agent_team.teamIndex == new_team_idx && !inbounder_assigned) {
            inbounder_assigned = true;
            inbounding.imInbounding = true;
            agent_pos = ball_pos; // Move player to the ball

            // Give them possession of the ball
            ctx.get<Grabbed>(ball_entity) = {true, (uint32_t) agent.id};
            in_possession.hasBall = true;
            in_possession.ballEntityID = ball_entity.id;

            // Set the agent's orientation to face the court
            agent_orient.orientation = new_orientation;
        }
    };

    // If we successfully found a player, update the game state.
    if(inbounder_assigned) {
        gameState.teamInPossession = (float)new_team_idx;
        gameState.inboundingInProgress = 1.0f;
        gameState.inboundClock = 5.f; // Reset the 5-second clock

        // Only increment the out-of-bounds count if it wasn't a 5-second turnover
        if (is_oob) {
            gameState.outOfBoundsCount++;
        }
    }
}


//=================================================== Ball Systems ===================================================
inline void moveBallRandomly(Engine &ctx,
                             Position &ball_pos,
                             RandomMovement &random_movement)
{
    random_movement.moveTimer++;
    if (random_movement.moveTimer >= random_movement.moveInterval)
    {
        random_movement.moveTimer = 0.f;
        const GridState *grid = ctx.data().grid;

        // Random movement in continuous space (0.1m steps)
        float dx = (float) (ctx.data().rng.sampleI32(0, 3) - 1) * 0.1f; // -0.1, 0, or 0.1 meters
        float dy = (float) (ctx.data().rng.sampleI32(0, 3) - 1) * 0.1f; // -0.1, 0, or 0.1 meters

        float new_x = ball_pos.position.x + dx;
        float new_y = ball_pos.position.y + dy;

        new_x = clamp(new_x, 0.f, grid->width);
        new_y = clamp(new_y, 0.f, grid->height);

        ball_pos.position.x = new_x;
        ball_pos.position.y = new_y;
    }
}

inline void moveBallSystem(Engine &ctx,
                           Position &ball_pos,
                           BallPhysics &ball_physics,
                           Grabbed &grabbed)
{
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Position &agent_pos = ctx.get<Position>(agent);
        InPossession &in_possession = ctx.get<InPossession>(agent);
        // Make the ball move with the agent if it's held
        bool agent_is_holding_this_ball = (in_possession.hasBall == true &&
                                           grabbed.isGrabbed &&
                                           grabbed.holderEntityID == (uint32_t)agent.id);
        if (agent_is_holding_this_ball) 
        {
            ball_pos = agent_pos;  // Move basketball to agent's new position
        }
    }

    if (ball_physics.velocity.length() == 0 || grabbed.isGrabbed) {return;}

    const GridState* grid = ctx.data().grid; // To clamp later
    float new_x = ball_pos.position.x + ball_physics.velocity[0];
    float new_y = ball_pos.position.y + ball_physics.velocity[1];
    float new_z = ball_pos.position.z + ball_physics.velocity[2];

    new_x = clamp(new_x, 0.f, grid->width);
    new_y = clamp(new_y, 0.f, grid->height);
    // new_z = clamp(new_z, 0.f, grid->depth);

    // Convert to discrete grid for wall collision checking
    int32_t discrete_x = (int32_t)(new_x * grid->cellsPerMeter);
    int32_t discrete_y = (int32_t)(new_y * grid->cellsPerMeter);
    discrete_x = clamp(discrete_x, 0, grid->discreteWidth - 1);
    discrete_y = clamp(discrete_y, 0, grid->discreteHeight - 1);

    const Cell &new_cell = grid->cells[discrete_y * grid->discreteWidth + discrete_x];

    if (!(new_cell.flags & CellFlag::Wall)) {
        ball_pos.position.x = new_x;
        ball_pos.position.y = new_y;
        ball_pos.position.z = new_z;
    }
}



inline void updatePointsWorthSystem(Engine &ctx,
                                    Position &agent_pos,
                                    InPossession &in_possession,
                                    Team &team)
{
    // Get all hoop positions
    Vector3 hoop_score_zones[NUM_HOOPS];
    uint32_t hoop_ids[NUM_HOOPS];
    for (CountT i = 0; i < NUM_HOOPS; i++) {
        Entity hoop = ctx.data().hoops[i];
        hoop_score_zones[i] = ctx.get<ScoringZone>(hoop).center;
        hoop_ids[i] = hoop.id;
    };

    // Find the hoop this agent should be shooting at (opposing team's hoop)
    Vector3 target_hoop_score_zone{};
    bool found_target_hoop = false;
    for (int i = 0; i < NUM_HOOPS; i++) {
        if (hoop_ids[i] != team.defendingHoopID) {
            target_hoop_score_zone = hoop_score_zones[i];
            found_target_hoop = true;
            break;
        }
    }

    // Calculate points worth for this agent's current position
    if (found_target_hoop) {
        in_possession.pointsWorth = getShotPointValue(agent_pos, target_hoop_score_zone);
    }
    else {
        in_possession.pointsWorth = 2; // Default to 2 points if we can't find the target hoop
    }
}

//=================================================== Agent Systems ===================================================
inline void grabSystem(Engine &ctx,
                       Entity agent_entity,
                       Action &action,
                       ActionMask &action_mask,
                       Position &agent_pos,
                       InPossession &in_possession,
                       Team &team,
                       GrabCooldown &grab_cooldown)
{
    GameState &gameState = ctx.singleton<GameState>();
    if (action_mask.can_grab == 0 || action.grab == 0) { return; }
    grab_cooldown.cooldown = 10.f;
    action.grab = 0;

    for (CountT i = 0; i < NUM_BASKETBALLS; i++) {
        Entity ball = ctx.data().balls[i];
        BallPhysics &ball_physics = ctx.get<BallPhysics>(ball);
        Grabbed &grabbed = ctx.get<Grabbed>(ball);
        if (ball_physics.inFlight) { continue; }

        bool agent_is_holding_this_ball = (in_possession.hasBall == true &&
                                           grabbed.isGrabbed &&
                                           grabbed.holderEntityID == (uint32_t)agent_entity.id);

        // If agent already has a ball, drop it
        if (agent_is_holding_this_ball) {
            in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
            in_possession.hasBall = false;
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER;
            grabbed.isGrabbed = false;
            continue;
        }

        // Check if ball is within grab range (0.5 meters)
        Position basketball_pos = ctx.get<Position>(ball);
        float distance_between_ball_and_player = (basketball_pos.position - agent_pos.position).length();

        if (distance_between_ball_and_player <= 0.3f) 
        {
            if (gameState.isOneOnOne == 1.f && team.teamIndex != gameState.teamInPossession) 
            {
                ctx.singleton<WorldClock>().resetNow = true;
                continue;
            }
            // Check if we're stealing from another agent
            for (CountT j = 0; j < NUM_AGENTS; j++) 
            {
                Entity agent = ctx.data().agents[j];
                InPossession &other_in_possession = ctx.get<InPossession>(agent);
                GrabCooldown &robbed_agent_grab_cooldown = ctx.get<GrabCooldown>(agent);
                if (other_in_possession.ballEntityID == (uint32_t)ball.id)
                {
                    other_in_possession.hasBall = false;
                    other_in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
                    robbed_agent_grab_cooldown.cooldown = SIMULATION_HZ; // Just makes it so they can't grab for one second
                }
            }

            in_possession.hasBall = true;
            in_possession.ballEntityID = ball.id;
            grabbed.holderEntityID = (uint32_t)agent_entity.id;
            grabbed.isGrabbed = true;
            ball_physics.inFlight = false; // Make it so the ball isn't "in flight" anymore
            ball_physics.velocity = Vector3::zero(); // And change its velocity to be zero

            // Clear shot information since this is a new possession
            ball_physics.shotByAgentID = ENTITY_ID_PLACEHOLDER;
            ball_physics.shotByTeamID = ENTITY_ID_PLACEHOLDER;
            ball_physics.shotPointValue = 2; // Default to 2 points

            gameState.teamInPossession = (float)team.teamIndex; // Update the team in possession
            gameState.liveBall = 1.f;
        }
    }
}



inline void passSystem(Engine &ctx,
                    Entity agent_entity,
                    Action &action,
                    ActionMask &action_mask,
                    Orientation &agent_orientation,
                    InPossession &in_possession,
                    Inbounding &inbounding)
{

    if (action_mask.can_pass == 0 || action.pass == 0) {return;}
    GameState &gameState = ctx.singleton<GameState>();

    for (CountT i = 0; i < NUM_BASKETBALLS; i++) {
        Entity ball = ctx.data().balls[i];
        BallPhysics &ball_physics = ctx.get<BallPhysics>(ball);
        Grabbed &grabbed = ctx.get<Grabbed>(ball);
        if (grabbed.holderEntityID == (uint32_t)agent_entity.id) {
            grabbed.isGrabbed = false;  // Ball is no longer grabbed
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER; // Ball is no longer held by anyone
            in_possession.hasBall = false; // Since agents can only hold 1 ball at a time, if they pass it they can't be holding one anymore
            in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER; // Whoever passed the ball is no longer in possession of it
            inbounding.imInbounding = false;
            ball_physics.velocity = agent_orientation.orientation.rotateVec(Vector3{0.f, 0.1f, 0.f}); // Setting the ball's velocity to have the same direction as the agent's orientation
                                                                                                      // Note: we use 0, 0.1, 0 because that's forward in our simulation specifically
            gameState.inboundingInProgress = 0.0f;
        }
    }
}


inline void shootSystem(Engine &ctx,
                        Entity agent_entity,
                        Action &action,
                        ActionMask &action_mask,
                        Position agent_pos,
                        Orientation &agent_orientation,
                        Inbounding &inbounding,
                        InPossession &in_possession,
                        Team &team)
{
    if (action_mask.can_shoot == 0 || action.shoot == 0) {return;}

    // Find the attacking hoop (not defendingHoopID)
    Vector3 attacking_hoop_score_zone = {0.f, 0.f, 0.f};
    float scoring_radius = 0.f;
    for (CountT i = 0; i < NUM_HOOPS; i++) {
        Entity hoop = ctx.data().hoops[i];
        if ((uint32_t)hoop.id != team.defendingHoopID) {
            attacking_hoop_score_zone = ctx.get<ScoringZone>(hoop).center;
            scoring_radius = ctx.get<ScoringZone>(hoop).radius;
        }
    }

    // Calculate vector to attacking hoop
    Vector3 ideal_shot_vector = attacking_hoop_score_zone - agent_pos.position;

    // Calculate intended angle towards hoop
    float intended_direction = std::atan2(ideal_shot_vector.x, ideal_shot_vector.y);



    // 1. Mess up angle based on distance
    float distance_to_hoop = ideal_shot_vector.length();
    float dist_stddev = DIST_DEVIATION_PER_METER/100 * distance_to_hoop;
    float deviation_from_distance = sampleUniform(ctx, -dist_stddev, dist_stddev);


    // 2. Mess up angle based on contest level (how close nearest defender is)
    float deviation_from_defender = 0.0f;
    float distance_to_nearest_defender = std::numeric_limits<float>::infinity();
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Team &defender_team = ctx.get<Team>(agent);
        Position &defender_pos = ctx.get<Position>(agent);
        if (defender_team.teamIndex != team.teamIndex)
        {
            Vector3 diff = agent_pos.position - defender_pos.position;
            float dist_to_def = diff.length();
            if (dist_to_def < distance_to_nearest_defender)
            {
                distance_to_nearest_defender = dist_to_def;
            }
        }
    };

    if (distance_to_nearest_defender < 2.0f) { // Only apply pressure if defender is  close
        float def_stddev = (DEF_DEVIATION_PER_METER/100) / (distance_to_nearest_defender + 0.1f);
        deviation_from_defender = sampleUniform(ctx, -def_stddev, def_stddev);
    }


    // 3. Mess up angle based on agent velocity
    float deviation_from_velocity = 0.0f;
    if (action.moveSpeed > 0) {
        float vel_stddev = VEL_DEVIATION_FACTOR/100 * action.moveSpeed;
        deviation_from_velocity = sampleUniform(ctx, -vel_stddev, vel_stddev);
    }

    // Combine all deviations and apply to the final shot direction
    float total_deviation = deviation_from_distance + deviation_from_defender + deviation_from_velocity;
    float shot_direction = intended_direction + total_deviation;

    // This is the final, correct trajectory vector for the ball
    Vector3 final_shot_vec = {sinf(shot_direction), cosf(shot_direction), 0.f};
    // final_shot_vec = (distance_to_hoop <= 5.f) ? ideal_shot_vector : Vector3{0, 1, 0}; // DEBUG
    bool shot_is_going_in = false;
    float how_far_to_go_along_shot_to_be_closest_to_hoop = ideal_shot_vector.dot(final_shot_vec);
    if (how_far_to_go_along_shot_to_be_closest_to_hoop < 0) {shot_is_going_in = false;}
    else
    {
        float closest_distance_to_hoop_sq = ideal_shot_vector.length2() - how_far_to_go_along_shot_to_be_closest_to_hoop * how_far_to_go_along_shot_to_be_closest_to_hoop;
        shot_is_going_in = closest_distance_to_hoop_sq <= scoring_radius * scoring_radius;
    }

    

    


    // Find the rotation that aligns the agent's orientation with the final shot direction vector.
    const Vector3 base_forward = {0.0f, 1.0f, 0.0f};
    agent_orientation.orientation = findRotationBetweenVectors(base_forward, final_shot_vec);


    GameState &gameState = ctx.singleton<GameState>();

    // Shoot the damn ball
    for (CountT i = 0; i < NUM_BASKETBALLS; i++) {
        Entity ball = ctx.data().balls[i];
        BallPhysics &ball_physics = ctx.get<BallPhysics>(ball);
        Grabbed &grabbed = ctx.get<Grabbed>(ball);
        if (grabbed.holderEntityID == agent_entity.id)
        {
            // Calculate the point value of this shot from the agent's current position
            int32_t shot_point_value = getShotPointValue(agent_pos, attacking_hoop_score_zone);
            if(shot_is_going_in == true) 
            {
                ball_physics.shotIsGoingIn = true;
                gameState.scoredBaskets++;
            }
            grabbed.isGrabbed = false;
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER;
            in_possession.hasBall = false;
            in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
            inbounding.imInbounding = false;

            ball_physics.velocity = final_shot_vec * .1f;



            ball_physics.inFlight = true;

            // Set who shot the ball for scoring system (these don't change after touching)
            ball_physics.shotByAgentID = (uint32_t)agent_entity.id;
            ball_physics.shotByTeamID = (uint32_t)team.teamIndex;
            ball_physics.shotPointValue = shot_point_value;

            // Also set last touched (these can change if ball is touched after shooting)
            ball_physics.lastTouchedByAgentID = (uint32_t)agent_entity.id;
            ball_physics.lastTouchedByTeamID = (uint32_t)team.teamIndex;
        }
    }
}


inline void moveAgentSystem(Engine &ctx,
                        Action &action,
                        ActionMask &action_mask,
                        Position &agent_pos, // Note: This should now store floats
                        InPossession &in_possession,
                        Inbounding &inbounding,
                        Orientation &agent_orientation,
                        Attributes &attributes)
{
    const GridState *grid = ctx.data().grid;
    if (action.rotate != 0)
    {
        float turn_angle = (pi/180.f) * action.rotate * 6;
        Quat turn = Quat::angleAxis(turn_angle, Vector3{0, 0, 1});
        agent_orientation.orientation = turn * agent_orientation.orientation;
    }

    if (action_mask.can_move == 0 || action.moveSpeed == 0) {return;}

    if (action.moveSpeed > 0)
    {
        // Treat moveSpeed as a velocity in meters/second, not a distance.
        // Let's say a moveSpeed of 1 corresponds to 1 m/s.
        float agent_velocity_magnitude = action.moveSpeed * attributes.speed * 4;
        if (in_possession.hasBall == 1) {agent_velocity_magnitude *= BALL_AGENT_SLOWDOWN;}

        constexpr float angle_between_directions = ANGLE_BETWEEN_DIRECTIONS;
        float move_angle = action.moveAngle * angle_between_directions;

        // Calculate velocity vector components
        float vel_x = std::sin(move_angle);
        if (inbounding.imInbounding == 1.f) {vel_x = 0.f;}
        float vel_y = -std::cos(move_angle); // Your forward is -Y
        
        Vector3 agent_orientation_as_vec = agent_orientation.orientation.rotateVec(AGENT_BASE_FORWARD);
        float dot_between_orientation_and_velocity = Vector3{vel_x, vel_y, 0}.normalize().dot(agent_orientation_as_vec);
        

        if (dot_between_orientation_and_velocity < -0.1f) {agent_velocity_magnitude *= 0.35f;} // moving backwards
        else if (dot_between_orientation_and_velocity < 0.1f) {agent_velocity_magnitude *= 0.5f;} // moving sideways
        

        // Calculate distance to move this frame
        float dx = vel_x * agent_velocity_magnitude * TIMESTEPS_TO_SECONDS_FACTOR;
        float dy = vel_y * agent_velocity_magnitude * TIMESTEPS_TO_SECONDS_FACTOR;



        // Update position (now using floats)
        float new_x = agent_pos.position.x + dx;
        float new_y = agent_pos.position.y + dy;

        // Boundary checking in continuous space
        new_x = clamp(new_x, 0.f, grid->width);
        new_y = clamp(new_y, 0.f, grid->height);

        // Convert to discrete grid for wall collision checking
        int32_t discrete_x = (int32_t)(new_x * grid->cellsPerMeter);
        int32_t discrete_y = (int32_t)(new_y * grid->cellsPerMeter);
        discrete_x = clamp(discrete_x, 0, grid->discreteWidth - 1);
        discrete_y = clamp(discrete_y, 0, grid->discreteHeight - 1);

        const Cell &new_cell = grid->cells[discrete_y * grid->discreteWidth + discrete_x];

        if (!(new_cell.flags & CellFlag::Wall)) {
            agent_pos.position.x = new_x;
            agent_pos.position.y = new_y;
        }
    }
}


inline void actionMaskSystem(Engine &ctx,
                             ActionMask &action_mask,
                             GrabCooldown &grab_cooldown,
                             InPossession &in_possession,
                             Inbounding &inbounding)
{
    GameState &gameState = ctx.singleton<GameState>();

    action_mask.can_move = 1;
    action_mask.can_grab = 1;
    action_mask.can_pass = 0;
    action_mask.can_shoot = 0;

    // Offensive actions
    if (in_possession.hasBall)
    {
        action_mask.can_pass = 1;
        action_mask.can_shoot = 1;
    }
    
    if (gameState.inboundingInProgress == 1.f)
    {
        action_mask.can_shoot = 0;
        action_mask.can_grab = 0;
        if (inbounding.imInbounding && gameState.liveBall == 0.f)
        {
            action_mask.can_move = 0;
        }
    }
    
    if (grab_cooldown.cooldown > 0.f)
    {
        action_mask.can_grab = 0;
    }
}



inline void agentCollisionSystem(Engine &ctx,
    Entity entity_a,
    Position &pos_a,
    Orientation &orient_a)
{
    // Query for all agents to check for collisions.
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity entity_b = ctx.data().agents[i];
        
        // Don't check an agent against itself and avoid duplicate checks.
        if (entity_a.id >= entity_b.id) {
            continue;
        }

        Position& pos_b = ctx.get<Position>(entity_b);
        Orientation& orient_b = ctx.get<Orientation>(entity_b);

        // --- 1. Get the properties of Rectangle A ---
        Vector3 center_a = pos_a.position;
        Vector3 fwd_a = orient_a.orientation.rotateVec({0, 1, 0}); // Forward vector
        Vector3 right_a = {fwd_a.y, -fwd_a.x, 0}; // Perpendicular vector

        Vector3 half_width_a = right_a * (AGENT_SHOULDER_WIDTH / 2.0f);
        Vector3 half_depth_a = fwd_a * (AGENT_DEPTH / 2.0f);

        Vector3 vertices_a[4] = {
            center_a - half_depth_a + half_width_a,
            center_a - half_depth_a - half_width_a,
            center_a + half_depth_a - half_width_a,
            center_a + half_depth_a + half_width_a
        };

        // --- 2. Get the properties of Rectangle B ---
        Vector3 center_b = pos_b.position;
        Vector3 fwd_b = orient_b.orientation.rotateVec({0, 1, 0});
        Vector3 right_b = {fwd_b.y, -fwd_b.x, 0};

        Vector3 half_width_b = right_b * (AGENT_SHOULDER_WIDTH / 2.0f);
        Vector3 half_depth_b = fwd_b * (AGENT_DEPTH / 2.0f);

        Vector3 vertices_b[4] = {
            center_b - half_depth_b + half_width_b,
            center_b - half_depth_b - half_width_b,
            center_b + half_depth_b - half_width_b,
            center_b + half_depth_b + half_width_b
        };
        
        // --- 3. Define the axes to test for separation (SAT) ---
        // For two rectangles, there are 4 potential separating axes.
        Vector3 axes[4] = {
            right_a.normalize(),
            fwd_a.normalize(),
            right_b.normalize(),
            fwd_b.normalize()
        };

        bool is_colliding = true;
        float min_overlap = std::numeric_limits<float>::max();
        Vector3 mtv_axis = {0,0,0}; // Minimum Translation Vector axis

        // --- 4. Test each axis for overlap ---
        for (int j = 0; j < 4; j++) {
            const Vector3& axis = axes[j];
            Projection p_a = projectRectangle(vertices_a, axis);
            Projection p_b = projectRectangle(vertices_b, axis);

            if (!projectionsOverlap(p_a, p_b)) {
                // Found a separating axis, so there is NO collision.
                is_colliding = false;
                break;
            } else {
                // Overlap found, calculate how much.
                float overlap = fminf(p_a.max, p_b.max) - fmaxf(p_a.min, p_b.min);
                if (overlap < min_overlap) {
                    min_overlap = overlap;
                    mtv_axis = axis;
                }
            }
        }
        
        // --- 5. If all axes had overlaps, then we have a collision ---
        if (is_colliding)
        {
            // --- Collision Response ---
            float penetration_depth = min_overlap;
            Vector3 correction_vec = mtv_axis;

            // Ensure the correction vector points from A to B
            Vector3 center_to_center = center_b - center_a;
            if (dot(center_to_center, correction_vec) < 0) {
                correction_vec = -correction_vec;
            }

            // Move each agent away from the other by half of the overlap.
            pos_a.position -= correction_vec * penetration_depth * 0.5f;
            pos_b.position += correction_vec * penetration_depth * 0.5f;
        }
    }
}


inline void hardCodeDefenseSystem(Engine &ctx,
    Team &defender_team,
    Position &defender_pos,
    Action &defender_action,
    Attributes &defender_attributes)
    {
        GameState &gameState = ctx.singleton<GameState>();
        
        if (gameState.teamInPossession == defender_team.teamIndex)
        {
            defender_action.moveSpeed = 0;
            return;
        }
        
        defender_action.grab = 1.f;
        Vector3 guarding_pos; // The place we want our defensive agent to go to defend
        bool found_offender = false;
        for (CountT i = 0; i < NUM_AGENTS; i++) {
            Entity agent = ctx.data().agents[i];
            InPossession &offender_in_possession = ctx.get<InPossession>(agent);
            Position &offender_pos = ctx.get<Position>(agent);
            if (offender_in_possession.hasBall && !found_offender) {
                for (CountT j = 0; j < NUM_HOOPS; j++) {
                    Entity hoop = ctx.data().hoops[j];
                    if (defender_team.defendingHoopID == hoop.id) {
                        Position &hoop_pos = ctx.get<Position>(hoop);
                    guarding_pos = offender_pos.position + GUARDING_DISTANCE * (hoop_pos.position - offender_pos.position).normalize();
                    found_offender = true;
                }
            }
        }
    }
    
    if (!found_offender)
    {
        defender_action.moveSpeed = 0;
        return;
    }
    
    Vector3 current_target = defender_attributes.currentTargetPosition;
    Vector3 ideal_target = guarding_pos;
    float interpolation_factor = defender_attributes.reactionSpeed * TIMESTEPS_TO_SECONDS_FACTOR;
    
    defender_attributes.currentTargetPosition = current_target + (ideal_target - current_target) * interpolation_factor;
    
    
    
    Vector3 move_vector = defender_attributes.currentTargetPosition - defender_pos.position;
    if (move_vector.length2() < 0.01f)
    {
        defender_action.moveSpeed = 0;
        return;
    }
    
    
    
    const Vector3 move_directions[] = {
        {0.f, -1.f, 0.f},  // 0: Up
        {1.f, -1.f, 0.f},  // 1: Up-Right
        {1.f, 0.f, 0.f},   // 2: Right
        {1.f, 1.f, 0.f},   // 3: Down-Right
        {0.f, 1.f, 0.f},   // 4: Down
        {-1.f, 1.f, 0.f},  // 5: Down-Left
        {-1.f, 0.f, 0.f},  // 6: Left
        {-1.f, -1.f, 0.f}  // 7: Up-Left
    };

    Vector3 desired_dir = move_vector.normalize();
    float max_dot = -2.f; // Initialize with a value lower than any possible dot product
    int32_t best_move_angle = 0;
    
    // Find which of the 8 directions is most aligned with our desired direction
    for (int32_t i = 0; i < 8; ++i)
    {
        float current_dot = dot(desired_dir, move_directions[i].normalize());
        if (current_dot > max_dot)
        {
            max_dot = current_dot;
            best_move_angle = i;
        }
    }
    
    // Set the action to the best-matching direction
    defender_action.moveSpeed = 1;
    defender_action.moveAngle = best_move_angle;
    defender_action.rotate = 0;
}




inline void rewardSystem(Engine &ctx,
                         Entity agent_entity,
                         Reward &reward,
                         Position &agent_pos,
                         Team &team,
                         InPossession &in_possession)
{
    // Find attacking hoop
    Position target_hoop_pos;
    for (CountT i = 0; i < NUM_HOOPS; i++)
    {
        Entity hoop = ctx.data().hoops[i];
        if ((uint32_t)hoop.id != team.defendingHoopID)
        {
            target_hoop_pos = ctx.get<Position>(hoop);
        }
    }

    // Find agent who shot the ball and reward them if the shot is going in
    for (CountT j = 0; j < NUM_BASKETBALLS; j++)
    {
        Entity ball = ctx.data().balls[j];
        BallPhysics &ball_physics = ctx.get<BallPhysics>(ball);
        if (ball_physics.shotByAgentID == (uint32_t)agent_entity.id && ball_physics.shotIsGoingIn == true)
        {
            reward.r += 5*ball_physics.shotPointValue;
        }
    }

    float distance_from_hoop = (agent_pos.position - target_hoop_pos.position).length();
    float proximity_reward = exp(-.1f * distance_from_hoop);
    reward.r += proximity_reward * in_possession.hasBall;
}

//=================================================== Hoop Systems ===================================================
inline void scoreSystem(Engine &ctx,
                        Entity hoop_entity,
                        Position &hoop_pos,
                        ScoringZone &scoring_zone)
{
    GameState &gameState = ctx.singleton<GameState>();

    for (CountT i = 0; i < NUM_BASKETBALLS; i++) {
        Entity ball = ctx.data().balls[i];
        Position &ball_pos = ctx.get<Position>(ball);
        BallPhysics &ball_physics = ctx.get<BallPhysics>(ball);
        float distance_to_hoop = std::sqrt((ball_pos.position.x - hoop_pos.position.x) * (ball_pos.position.x - hoop_pos.position.x) +
                                        (ball_pos.position.y - hoop_pos.position.y) * (ball_pos.position.y - hoop_pos.position.y));

        if (distance_to_hoop <= scoring_zone.radius && ball_physics.inFlight) {
            // Use the point value that was calculated when the shot was taken
            int32_t points_scored = ball_physics.shotPointValue;

            // Find which team is defending this hoop (has defendingHoopID == hoop_entity.id)
            uint32_t inbounding_team_idx = 0; // Default fallback
            for (CountT j = 0; j < NUM_AGENTS; j++) {
                Entity agent = ctx.data().agents[j];
                Team &team = ctx.get<Team>(agent);
                Stats &agent_stats = ctx.get<Stats>(agent);
                if (team.defendingHoopID == (uint32_t)hoop_entity.id)
                {
                    inbounding_team_idx = (uint32_t)team.teamIndex;
                }

                if (agent.id == ball_physics.shotByAgentID)
                {
                    agent_stats.points += (team.defendingHoopID == (uint32_t)hoop_entity.id) ? -ball_physics.shotPointValue : ball_physics.shotPointValue;
                }
            }

            Position inbound_spot;
            Quat inbound_orientation;

            if ((uint32_t)hoop_entity.id == (uint32_t)gameState.team0Hoop)
            {
                // Someone scored on Team 0's hoop, so Team 1 gets the points
                gameState.team1Score += points_scored;

                // Inbound spot is on the baseline behind the hoop that was scored on.
                inbound_spot = Position{{COURT_MIN_X, hoop_pos.position.y+(PIXELS_PER_METER/60), 0.f}};
            }
            else
            {
                // Someone scored on Team 1's hoop, so Team 0 gets the points
                gameState.team0Score += points_scored;

                inbound_spot = Position{{COURT_MAX_X, hoop_pos.position.y+(PIXELS_PER_METER/60), 0.f}};
            }

            gameState.scoredBaskets++;

            // Set the ball's state for the inbound
            ball_physics.inFlight = false;
            ball_physics.velocity = Vector3::zero();

            // Clear shot information since the shot scored
            ball_physics.shotByAgentID = ENTITY_ID_PLACEHOLDER;
            ball_physics.shotByTeamID = ENTITY_ID_PLACEHOLDER;
            ball_physics.shotPointValue = 2; // Reset to default
            ball_physics.shotIsGoingIn = false;

            // Set up the inbound for the defending team.
            if (gameState.isOneOnOne == 0.f)
            {
                ball_pos = inbound_spot;
                inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
                assignInbounder(ctx, ball, inbound_spot, inbounding_team_idx, inbound_orientation, false);
            }
            else 
            {
                ctx.singleton<WorldClock>().resetNow = true;
            }
        }
    }
}


//=================================================== General Systems ===================================================
inline void resetSystem(Engine &ctx, WorldClock &world_clock) {
    // This system only executes its logic if the reset flag has been set.
    if (world_clock.resetNow == 0) {
        return;
    }

    resetWorld(ctx);

    // Finally, clear the world's master reset flag.
    ctx.singleton<WorldClock>().resetNow = false;
}

inline void tick(Engine &ctx,
                 Reset &reset,
                 Done &done,
                 CurStep &episode_step,
                 GrabCooldown &grab_cooldown,
                 Reward &reward)
{
    reward.r = 0.f;
    // If a reset has been triggered, mark the agent as done for the learning side.
    if (reset.resetNow == 1) {
        done.episodeDone = 1.f;
        episode_step.step = 0;
    } else {
        done.episodeDone = 0.f;
        episode_step.step++;
    }

    // Per-step logic like cooldowns can stay here.
    grab_cooldown.cooldown = fmaxf(0.f, grab_cooldown.cooldown - 1.f);
}



inline void clockSystem(Engine &ctx, WorldClock &world_clock)
{
    GameState &gameState = ctx.singleton<GameState>();

    // Decrement game and shot clocks if the ball is live
    if (gameState.liveBall > 0.5f && gameState.gameClock > 0.f)
    {
        gameState.gameClock -= TIMESTEPS_TO_SECONDS_FACTOR;
        gameState.shotClock -= TIMESTEPS_TO_SECONDS_FACTOR;
    }

    // Decrement the inbound clock if an inbound is in progress
    if (gameState.inboundingInProgress > 0.5f)
    {
        gameState.inboundClock -= TIMESTEPS_TO_SECONDS_FACTOR;
    }

    if (gameState.gameClock <= 0.f && gameState.liveBall > 0.5f)
    {
        world_clock.resetNow = 1;
    }

    if (gameState.shotClock < 0.f)
    {
        gameState.shotClock = 0.f;
    }
    Position hoop0_pos = ctx.get<Position>(ctx.data().hoops[0]);
    Position hoop1_pos = ctx.get<Position>(ctx.data().hoops[1]);
    // printf("WorldID: %d: End of clockSystem. Hoop0 position is: (%f, %f, %f) and hoop1 position is: (%f, %f, %f)\n", ctx.worldID().idx, hoop0_pos.position.x, hoop0_pos.position.y, hoop0_pos.position.z, hoop1_pos.position.x, hoop1_pos.position.y, hoop1_pos.position.z);
    float agent0_team = ctx.get<Team>(ctx.data().agents[0]).teamIndex;
    float agent1_team = ctx.get<Team>(ctx.data().agents[1]).teamIndex;
    // printf("WorldID: %d: End of clockSystem. Agent0 team is: (%f) and Agent1 position is: (%f)\n", ctx.worldID().idx, agent0_team, agent1_team);
}



inline void updateLastTouchSystem(Engine &ctx,
                                  Position &ball_pos,
                                  BallPhysics &ball_physics)
{
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Team &team = ctx.get<Team>(agent);
        Position &agent_pos = ctx.get<Position>(agent);
        // Check if agent is within touch distance (0.2 meters)
        float distance = (ball_pos.position - agent_pos.position).length();

        if (distance <= AGENT_SIZE_M)
        {
            ball_physics.lastTouchedByAgentID = (uint32_t)agent.id;
            ball_physics.lastTouchedByTeamID = (uint32_t)team.teamIndex;
        }
    }
}



inline void outOfBoundsSystem(Engine &ctx,
                            Entity ball_entity,
                            Position &ball_pos,
                            BallPhysics &ball_physics)
{
    GameState &gameState = ctx.singleton<GameState>();

    // Check if the ball's center has crossed the court boundaries and we are not currently inbounding

    if ((ball_pos.position.x < COURT_MIN_X || ball_pos.position.x > COURT_MAX_X ||
        ball_pos.position.y < COURT_MIN_Y || ball_pos.position.y > COURT_MAX_Y) &&
        gameState.inboundingInProgress == 0.f)
    {
        if (gameState.isOneOnOne == 1.f) {
            ctx.singleton<WorldClock>().resetNow = true;
        }
        else
        {
            ball_physics.inFlight = false;
            ball_physics.velocity = Vector3::zero();
            gameState.liveBall = 0.f;

            // The team that did NOT last touch the ball gets possession.
            uint32_t new_team_idx = 1 - ball_physics.lastTouchedByTeamID;

            // Find the player who had the ball and reset their position
            for (CountT i = 0; i < NUM_AGENTS; i++) {
                Entity agent = ctx.data().agents[i];
                InPossession &in_possession = ctx.get<InPossession>(agent);
                Position &agent_pos = ctx.get<Position>(agent);
                // If this agent was the one who went out of bounds with the ball...
                if (in_possession.hasBall && in_possession.ballEntityID == ball_entity.id)
                {
                    agent_pos.position += findVectorToCenter(ctx, agent_pos);

                    // Take the ball away
                    in_possession.hasBall = false;
                    in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
                }
            }

            // Call the helper to give the ball to the other team.
            Quat inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
            assignInbounder(ctx, ball_entity, ball_pos, new_team_idx, inbound_orientation, true);
        }
    }
}


inline void inboundViolationSystem(Engine &ctx, WorldClock &world_clock)
{
    GameState &gameState = ctx.singleton<GameState>();

    // This is the conditional check. If this isn't true, the system does nothing.
    if (!(gameState.inboundingInProgress > 0.5f && gameState.inboundClock <= 0.f)) {return;}

    uint32_t current_team_idx = (uint32_t)gameState.teamInPossession;
    uint32_t new_team_idx = 1 - current_team_idx;
    uint32_t ball_to_turnover_id = ENTITY_ID_PLACEHOLDER;

    gameState.liveBall = 0.f;

    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Inbounding &inb = ctx.get<Inbounding>(agent);
        Position &agent_pos = ctx.get<Position>(agent);
        InPossession &poss = ctx.get<InPossession>(agent);
        if (inb.imInbounding) {
            ball_to_turnover_id = poss.ballEntityID;

            inb.imInbounding = false;
            poss.hasBall = false;
            poss.ballEntityID = ENTITY_ID_PLACEHOLDER;

            agent_pos.position += findVectorToCenter(ctx, agent_pos);
        }
    }

    if (ball_to_turnover_id != ENTITY_ID_PLACEHOLDER) {
        for (CountT i = 0; i < NUM_BASKETBALLS; i++) {
            Entity ball = ctx.data().balls[i];
            Position &ball_pos = ctx.get<Position>(ball);
            Grabbed &grabbed = ctx.get<Grabbed>(ball);
            if (ball.id == (int32_t)ball_to_turnover_id) {
                grabbed = {false, ENTITY_ID_PLACEHOLDER};
                Quat inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
                assignInbounder(ctx, ball, ball_pos, new_team_idx, inbound_orientation, true);
            }
        }
    }
}


// This is a temporary helper struct used ONLY by the fillObservationsSystem
// to gather all agent data before sorting it into the observation vector.
struct AgentObservationData {
    int32_t id;
    int32_t teamID;
    Position pos;
    Orientation orient;
    InPossession in_pos;
    Inbounding inb;
    GrabCooldown cooldown;
};


inline void fillObservationsSystem(Engine &ctx,
                                   Entity agent_entity,
                                   Observations &observations,
                                   Position &agent_pos,
                                   Orientation &agent_orientation,
                                   InPossession &in_possession,
                                   Inbounding &inbounding,
                                   Team &agent_team,
                                   GrabCooldown &grab_cooldown)
{
    auto &obs = observations.observationsArray;
    const GameState &gameState = ctx.singleton<GameState>();
    int32_t idx = 0;

    // Helper lambda to fill a block of the array with a Vector3
    auto fill_vec3 = [&](const Vector3 &vec) {
        obs[idx++] = vec.x;
        obs[idx++] = vec.y;
        obs[idx++] = vec.z;
    };

    // Helper lambda to fill a block of the array with a Quaternion
    auto fill_quat = [&](const Quat &q) {
        obs[idx++] = q.w;
        obs[idx++] = q.x;
        obs[idx++] = q.y;
        obs[idx++] = q.z;
    };

    // ===================================================
    // Part 1: Gather All Data from the World
    // ===================================================

    // --- Ball State ---
    Position ball_pos;
    BallPhysics ball_phys;
    Grabbed ball_grabbed;

    // we assume 1 ball
    Entity ball = ctx.data().balls[0];
    ball_pos = ctx.get<Position>(ball);
    ball_phys = ctx.get<BallPhysics>(ball);
    ball_grabbed = ctx.get<Grabbed>(ball);

    // --- Hoop Positions ---
    Position hoop_positions[NUM_HOOPS];
    uint32_t hoop_ids[NUM_HOOPS];
    for (CountT i = 0; i < NUM_HOOPS; i++) {
        Entity hoop = ctx.data().hoops[i];
        hoop_positions[i] = ctx.get<Position>(hoop);
        hoop_ids[i] = hoop.id;
    };

    // --- All Agent Data ---
    AgentObservationData all_agents[NUM_AGENTS];
    int agent_idx = 0;
    int32_t inbounder_id = -1; // Store the ID of the current inbounder
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Team &t = ctx.get<Team>(agent);
        Position &p = ctx.get<Position>(agent);
        Orientation &o = ctx.get<Orientation>(agent);
        InPossession &ip = ctx.get<InPossession>(agent);
        Inbounding &ib = ctx.get<Inbounding>(agent);
        GrabCooldown &gc = ctx.get<GrabCooldown>(agent);
        all_agents[agent_idx++] = { agent.id, t.teamIndex, p, o, ip, ib, gc };
        if (ib.imInbounding) {
            inbounder_id = agent.id;
        }
    };

    // ===================================================
    // Part 2: Fill The Observation Array in Order
    // ===================================================

    // Game Context
    obs[idx++] = gameState.gameClock;
    obs[idx++] = gameState.shotClock;
    obs[idx++] = gameState.period;
    obs[idx++] = gameState.inboundingInProgress;
    obs[idx++] = gameState.inboundClock;

    // Egocentric Score
    if (agent_team.teamIndex == 0)
    {
        obs[idx++] = gameState.team0Score;
        obs[idx++] = gameState.team1Score;
    }
    else
    {
        obs[idx++] = gameState.team1Score;
        obs[idx++] = gameState.team0Score;
    }

    // Now fill the ball info using the variables we populated earlier.
    fill_vec3(ball_pos.position);
    fill_vec3(ball_phys.velocity);
    obs[idx++] = (float)ball_grabbed.isGrabbed;
    obs[idx++] = (float)ball_phys.inFlight;
    obs[idx++] = (float)ball_phys.shotPointValue;
    obs[idx++] = (float)ball_phys.lastTouchedByTeamID;

    // Hoop Info
    Position attacking_hoop_pos = (hoop_ids[0] != agent_team.defendingHoopID) ? hoop_positions[0] : hoop_positions[1];
    Position defending_hoop_pos = (hoop_ids[0] == agent_team.defendingHoopID) ? hoop_positions[0] : hoop_positions[1];
    fill_vec3(attacking_hoop_pos.position);
    fill_vec3(defending_hoop_pos.position);

    // Self Data
    fill_vec3(agent_pos.position);
    fill_quat(agent_orientation.orientation);
    obs[idx++] = (float)in_possession.hasBall;
    obs[idx++] = (float)in_possession.pointsWorth;
    obs[idx++] = (float)inbounding.imInbounding;
    obs[idx++] = grab_cooldown.cooldown;

    // Teammate & Opponent Data
    int teammate_count = 0;
    int opponent_count = 0;
    const int max_teammates = (NUM_AGENTS / 2) - 1;
    const int max_opponents = NUM_AGENTS / 2;

    for (int i = 0; i < agent_idx; i++) {
        if (all_agents[i].id == agent_entity.id) continue;

        if (all_agents[i].teamID == agent_team.teamIndex)
        {
            if (teammate_count < max_teammates)
            {
                fill_vec3(all_agents[i].pos.position);
                fill_quat(all_agents[i].orient.orientation);
                obs[idx++] = (float)all_agents[i].in_pos.hasBall;
                teammate_count++;
            }
        }
        else
        {
            if (opponent_count < max_opponents)
            {
                fill_vec3(all_agents[i].pos.position);
                fill_quat(all_agents[i].orient.orientation);
                obs[idx++] = (float)all_agents[i].in_pos.hasBall;
                opponent_count++;
            }
        }
    }

    // Padding for agent data
    int agent_feature_size = 3 + 4 + 1; // Pos, Orient, HasBall
    for (int i = teammate_count; i < max_teammates; i++)
    {
        for (int j = 0; j < agent_feature_size; j++) obs[idx++] = 0.f;
    }
    for (int i = opponent_count; i < max_opponents; i++)
    {
        for (int j = 0; j < agent_feature_size; j++) obs[idx++] = 0.f;
    }

    // One-hot encoded vector for who has the ball
    for (int i = 0; i < agent_idx; i++)
    {
        obs[idx++] = (all_agents[i].id == (int32_t)ball_grabbed.holderEntityID) ? 1.f : 0.f;
    }
    for (int i = agent_idx; i < NUM_AGENTS; i++) { obs[idx++] = 0.f; }

    // One-hot encoded vector for who is inbounding
    for (int i = 0; i < agent_idx; i++)
    {
        obs[idx++] = (all_agents[i].id == inbounder_id) ? 1.f : 0.f;
    }
    for (int i = agent_idx; i < NUM_AGENTS; i++) { obs[idx++] = 0.f; }


    assert(idx < (int32_t)obs.size()); // Ensure we didn't overflow the observation array
    // Zero out any remaining space for safety
    for (; idx < (int32_t)obs.size(); idx++)
    {
        obs[idx] = 0.f;
    }
}

TaskGraphNodeID setupGameStepTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
    {
    auto tickNode = builder.addToGraph<ParallelForNode<Engine, tick,
        Reset, Done, CurStep, GrabCooldown, Reward>>(deps);

    auto actionMaskingNode = builder.addToGraph<ParallelForNode<Engine, actionMaskSystem,
        ActionMask, GrabCooldown, InPossession, Inbounding>>({tickNode});

    auto moveAgentSystemNode = builder.addToGraph<ParallelForNode<Engine, moveAgentSystem,
        Action, ActionMask, Position, InPossession, Inbounding, Orientation, Attributes>>({actionMaskingNode});

    auto grabSystemNode = builder.addToGraph<ParallelForNode<Engine, grabSystem,
        Entity, Action, ActionMask, Position, InPossession, Team, GrabCooldown>>({moveAgentSystemNode});

    auto passSystemNode = builder.addToGraph<ParallelForNode<Engine, passSystem,
        Entity, Action, ActionMask, Orientation, InPossession, Inbounding>>({grabSystemNode});

    auto shootSystemNode = builder.addToGraph<ParallelForNode<Engine, shootSystem,
        Entity, Action, ActionMask, Position, Orientation, Inbounding, InPossession, Team>>({passSystemNode});

    auto moveBallSystemNode = builder.addToGraph<ParallelForNode<Engine, moveBallSystem,
        Position, BallPhysics, Grabbed>>({shootSystemNode});

    auto scoreSystemNode = builder.addToGraph<ParallelForNode<Engine, scoreSystem,
        Entity, Position, ScoringZone>>({moveBallSystemNode});

    auto outOfBoundsSystemNode = builder.addToGraph<ParallelForNode<Engine, outOfBoundsSystem,
        Entity, Position, BallPhysics>>({scoreSystemNode});

    auto updateLastTouchSystemNode = builder.addToGraph<ParallelForNode<Engine, updateLastTouchSystem,
        Position, BallPhysics>>({outOfBoundsSystemNode});

    auto clockSystemNode = builder.addToGraph<ParallelForNode<Engine, clockSystem,
        WorldClock>>({updateLastTouchSystemNode});

    auto inboundViolationSystemNode = builder.addToGraph<ParallelForNode<Engine, inboundViolationSystem,
        WorldClock>>({clockSystemNode});

    auto resetSystemNode = builder.addToGraph<ParallelForNode<Engine, resetSystem,
        WorldClock>>({inboundViolationSystemNode});

    auto updatePointsWorthNode = builder.addToGraph<ParallelForNode<Engine, updatePointsWorthSystem,
        Position, InPossession, Team>>({resetSystemNode});

    auto agentCollisionNode = builder.addToGraph<ParallelForNode<Engine, agentCollisionSystem,
        Entity, Position, Orientation>>({updatePointsWorthNode});

    auto hardCodeDefenseSystemNode = builder.addToGraph<ParallelForNode<Engine, hardCodeDefenseSystem,
        Team, Position, Action, Attributes>>({agentCollisionNode});

    auto fillObservationsNode = builder.addToGraph<ParallelForNode<Engine, fillObservationsSystem,
        Entity, Observations, Position, Orientation, InPossession,
        Inbounding, Team, GrabCooldown>>({hardCodeDefenseSystemNode});

    auto rewardSystemNode = builder.addToGraph<ParallelForNode<Engine, rewardSystem,
        Entity, Reward, Position, Team, InPossession>>({fillObservationsNode});

    return rewardSystemNode;
}

}
