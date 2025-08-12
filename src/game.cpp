#include "game.hpp"
#include "types.hpp"
#include "constants.hpp"
#include "gen.hpp"
#include "helper.hpp"
#include <cmath>

using namespace madrona;
using namespace madrona::math;



namespace madBasketball {
inline void assignInbounder(Engine &ctx, Entity ball_entity, Position ball_pos, int32_t new_team_idx, Quat new_orientation, bool is_oob)
{
    GameState &gameState = ctx.singleton<GameState>();
    float inbounder_assigned = 0.0f;

    // Find the first available player on the new team.
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Team &agent_team = ctx.get<Team>(agent);
        Inbounding &inbounding = ctx.get<Inbounding>(agent);
        Position &agent_pos = ctx.get<Position>(agent);
        InPossession &in_possession = ctx.get<InPossession>(agent);
        Orientation &agent_orient = ctx.get<Orientation>(agent);
        if (agent_team.teamIndex == new_team_idx && inbounder_assigned == 0) {
            inbounder_assigned = 1;
            inbounding.imInbounding = 1;
            agent_pos = ball_pos; // Move player to the ball

            // Give them possession of the ball
            ctx.get<Grabbed>(ball_entity) = {1, agent.id};
            in_possession.hasBall = 1;
            in_possession.ballEntityID = ball_entity.id;

            // Set the agent's orientation to face the court
            agent_orient.orientation = new_orientation;
        }
    };

    // If we successfully found a player, update the game state.
    if(inbounder_assigned > 0) {
        gameState.teamInPossession = (float)new_team_idx;
        gameState.inboundingInProgress = 1;
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
                           Grabbed &grabbed,
                           Velocity &ball_velocity)
{
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Position &agent_pos = ctx.get<Position>(agent);
        InPossession &in_possession = ctx.get<InPossession>(agent);
        // Make the ball move with the agent if it's held
        int32_t agent_is_holding_this_ball = (in_possession.hasBall == 1 &&
                                           grabbed.isGrabbed == 1 &&
                                           grabbed.holderEntityID == agent.id) ? 1 : 0;
        if (agent_is_holding_this_ball == 1) 
        {
            ball_pos = agent_pos;  // Move basketball to agent's new position
        }
    }

    if (ball_velocity.velocity.length() == 0 || grabbed.isGrabbed == 1) {return;}

    const GridState* grid = ctx.data().grid; // To clamp later
    float new_x = ball_pos.position.x + ball_velocity.velocity[0];
    float new_y = ball_pos.position.y + ball_velocity.velocity[1];
    float new_z = ball_pos.position.z + ball_velocity.velocity[2];

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
    int32_t hoop_ids[NUM_HOOPS];
    for (CountT i = 0; i < NUM_HOOPS; i++) {
        Entity hoop = ctx.data().hoops[i];
        hoop_score_zones[i] = ctx.get<ScoringZone>(hoop).center;
        hoop_ids[i] = hoop.id;
    };

    // Find the hoop this agent should be shooting at (opposing team's hoop)
    Vector3 target_hoop_score_zone{};
    float found_target_hoop = 0.0f;
    for (int i = 0; i < NUM_HOOPS; i++) {
        if (hoop_ids[i] != team.defendingHoopID) {
            target_hoop_score_zone = hoop_score_zones[i];
            found_target_hoop = 1.0f;
            break;
        }
    }

    // Calculate points worth for this agent's current position
    if (found_target_hoop == 1) {
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
        Velocity &ball_velocity = ctx.get<Velocity>(ball);
        if (ball_physics.inFlight == 1) { continue; }

        int32_t agent_is_holding_this_ball = (in_possession.hasBall == 1 &&
                                           grabbed.isGrabbed == 1 &&
                                           grabbed.holderEntityID == agent_entity.id) ? 1 : 0;

        // If agent already has a ball, drop it
        if (agent_is_holding_this_ball == 1) {
            in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
            in_possession.hasBall = 0;
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER;
            grabbed.isGrabbed = 0;
            continue;
        }

        // Check if ball is within grab range (0.5 meters)
        Position basketball_pos = ctx.get<Position>(ball);
        float distance_between_ball_and_player = (basketball_pos.position - agent_pos.position).length();

        if (distance_between_ball_and_player <= 0.3f) 
        {
            if (gameState.isOneOnOne == 1.f && team.teamIndex != gameState.teamInPossession) 
            {
                ctx.singleton<WorldClock>().resetNow = 1.0f;
                continue;
            }
            // Check if we're stealing from another agent
            for (CountT j = 0; j < NUM_AGENTS; j++) 
            {
                Entity agent = ctx.data().agents[j];
                InPossession &other_in_possession = ctx.get<InPossession>(agent);
                GrabCooldown &robbed_agent_grab_cooldown = ctx.get<GrabCooldown>(agent);
                if (other_in_possession.ballEntityID == ball.id)
                {
                    other_in_possession.hasBall = 0;
                    other_in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
                    robbed_agent_grab_cooldown.cooldown = SIMULATION_HZ; // Just makes it so they can't grab for one second
                }
            }

            in_possession.hasBall = 1;
            in_possession.ballEntityID = ball.id;
            grabbed.holderEntityID = agent_entity.id;
            grabbed.isGrabbed = 1;
            ball_physics.inFlight = 0; // Make it so the ball isn't "in flight" anymore
            ball_velocity.velocity = Vector3::zero(); // And change its velocity to be zero

            // Clear shot information since this is a new possession
            ball_physics.shotByAgentID = ENTITY_ID_PLACEHOLDER;
            ball_physics.shotByTeamID = ENTITY_ID_PLACEHOLDER;
            ball_physics.shotPointValue = 2; // Default to 2 points

            gameState.teamInPossession = (float)team.teamIndex; // Update the team in possession
            gameState.liveBall = 1;
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
        Grabbed &grabbed = ctx.get<Grabbed>(ball);
        Velocity &ball_velocity = ctx.get<Velocity>(ball);
        if (grabbed.holderEntityID == agent_entity.id) {
            grabbed.isGrabbed = 0;  // Ball is no longer grabbed
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER; // Ball is no longer held by anyone
            in_possession.hasBall = 0; // Since agents can only hold 1 ball at a time, if they pass it they can't be holding one anymore
            in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER; // Whoever passed the ball is no longer in possession of it
            inbounding.imInbounding = 0;
            ball_velocity.velocity = agent_orientation.orientation.rotateVec(Vector3{0.f, 0.1f, 0.f}); // Setting the ball's velocity to have the same direction as the agent's orientation
                                                                                                      // Note: we use 0, 0.1, 0 because that's forward in our simulation specifically
            gameState.inboundingInProgress = 0;
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
                        Team &team,
                        Reward &reward,
                        Velocity &vel)
{
    if (action_mask.can_shoot == 0 || action.shoot == 0) {return;}

    // Find the attacking hoop (not defendingHoopID)
    Vector3 attacking_hoop_score_zone = {0.f, 0.f, 0.f};
    float scoring_radius = 0.f;
    for (CountT i = 0; i < NUM_HOOPS; i++) {
        Entity hoop = ctx.data().hoops[i];
        if (hoop.id != team.defendingHoopID) {
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
    float dist_stddev = DIST_DEVIATION_PER_METER * distance_to_hoop;
    float deviation_from_distance = sampleUniform(ctx, -dist_stddev, dist_stddev);


    // 2. Mess up angle based on contest level (how close nearest defender is)
    float deviation_from_defender = 0.0f;
    float distance_to_nearest_defender = std::numeric_limits<float>::infinity();
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity defender = ctx.data().agents[i];
        Team &defender_team = ctx.get<Team>(defender);
        Position &defender_pos = ctx.get<Position>(defender);
        if (defender_team.teamIndex != team.teamIndex)
        {
            float dist_to_def = (agent_pos.position - defender_pos.position).length();
            if (dist_to_def < distance_to_nearest_defender)
            {
                distance_to_nearest_defender = dist_to_def;
            }
        }
    };

    if (distance_to_nearest_defender < 2.0f) { // Only apply pressure if defender is  close
        float def_stddev = (DEF_DEVIATION_PER_METER) / (distance_to_nearest_defender + 0.1f);
        deviation_from_defender = sampleUniform(ctx, -def_stddev, def_stddev);
    }


    // 3. Mess up angle based on agent velocity
    float deviation_from_velocity = 0.0f;
    if (action.move > 0) {
        float vel_stddev = VEL_DEVIATION_FACTOR * vel.velocity.length();
        deviation_from_velocity = sampleUniform(ctx, -vel_stddev, vel_stddev);
    }

    // Combine all deviations and apply to the final shot direction
    float total_deviation = deviation_from_distance + deviation_from_defender + deviation_from_velocity;
    float shot_direction = intended_direction + total_deviation;
    Vector3 final_shot_vec = {sinf(shot_direction), cosf(shot_direction), 0.f};


    float shot_is_going_in = 0.0f;
    float how_far_to_go_along_shot_to_be_closest_to_hoop = ideal_shot_vector.dot(final_shot_vec);
    if (how_far_to_go_along_shot_to_be_closest_to_hoop < 0) {shot_is_going_in = 0.0f;}
    else
    {
        float closest_distance_to_hoop_sq = ideal_shot_vector.length2() - how_far_to_go_along_shot_to_be_closest_to_hoop * how_far_to_go_along_shot_to_be_closest_to_hoop;
        shot_is_going_in = (closest_distance_to_hoop_sq <= scoring_radius * scoring_radius) ? 1.0f : 0.0f;
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
        Velocity &ball_velocity = ctx.get<Velocity>(ball);
        if (grabbed.holderEntityID == agent_entity.id)
        {
            // Calculate the point value of this shot from the agent's current position
            int32_t shot_point_value = getShotPointValue(agent_pos, attacking_hoop_score_zone);
            if(shot_is_going_in == 1) 
            {
                ball_physics.shotIsGoingIn = 1;
                gameState.scoredBaskets++;
            }
            else {reward.r -= 1.f;}
            grabbed.isGrabbed = 0;
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER;
            in_possession.hasBall = 0;
            in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
            inbounding.imInbounding = 0.0f;

            ball_velocity.velocity = final_shot_vec * .1f;



            ball_physics.inFlight = 1;

            // Set who shot the ball for scoring system (these don't change after touching)
            ball_physics.shotByAgentID = agent_entity.id;
            ball_physics.shotByTeamID = team.teamIndex;
            ball_physics.shotPointValue = shot_point_value;

            // Also set last touched (these can change if ball is touched after shooting)
            ball_physics.lastTouchedByAgentID = agent_entity.id;
            ball_physics.lastTouchedByTeamID = team.teamIndex;
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
                        Attributes &attributes,
                        Velocity &agent_vel)
{
    const GridState *grid = ctx.data().grid;
    if (action.rotate != 0)
    {
        float turn_angle = (action.rotate == 1) ? (pi/180.f) * 6 : (pi/180.f) * -6; // rotate can only be 1 or 2 (hopefully)
        Quat turn = Quat::angleAxis(turn_angle, Vector3{0, 0, 1});
        agent_orientation.orientation = turn * agent_orientation.orientation;
    }

    if (action_mask.can_move == 0) {return;}


    
    float move_angle = action.moveAngle * ANGLE_BETWEEN_DIRECTIONS;
    
    // Calculate velocity vector components
    Vector3 delta_vel = Vector3{std::sin(move_angle), -std::cos(move_angle), 0} * attributes.quickness * action.move; // Forward is -Y
    
    
    float maximum_speed = attributes.maxSpeed;
    Vector3 agent_orientation_as_vec = agent_orientation.orientation.rotateVec(AGENT_BASE_FORWARD);
    float dot_between_orientation_and_velocity = 0.0f;
    if (agent_vel.velocity.length2() > 1e-6f) {dot_between_orientation_and_velocity = agent_vel.velocity.normalize().dot(agent_orientation_as_vec);}

    if (dot_between_orientation_and_velocity < -0.1f) // moving backwards
    {
        maximum_speed *= .1f;
        delta_vel *= .1f;
    } 
    else if (dot_between_orientation_and_velocity <= 0.8f) // moving sideways
    {
        maximum_speed *= .7f;
        delta_vel *= .1f;
    }
    agent_vel.velocity += delta_vel;
    if (inbounding.imInbounding == 1) {delta_vel.x = 0.f;}
    if (in_possession.hasBall == 1) {maximum_speed *= BALL_AGENT_SLOWDOWN;}
    if (agent_vel.velocity.length() > maximum_speed) {agent_vel.velocity *= maximum_speed / agent_vel.velocity.length();}
    

    // Calculate distance to move this frame
    float dx = agent_vel.velocity.x * TIMESTEPS_TO_SECONDS_FACTOR;
    float dy = agent_vel.velocity.y * TIMESTEPS_TO_SECONDS_FACTOR;


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

    if (!(new_cell.flags & CellFlag::Wall)) 
    {
        agent_pos.position.x = new_x;
        agent_pos.position.y = new_y;
    }
    agent_vel.velocity *= .95f;
}


inline void actionMaskSystem(Engine &ctx,
                             ActionMask &action_mask,
                             GrabCooldown &grab_cooldown,
                             InPossession &in_possession,
                             Inbounding &inbounding,
                             Team &team)
{
    GameState &gameState = ctx.singleton<GameState>();

    action_mask.can_move = 1;
    action_mask.can_grab = 1;
    action_mask.can_pass = 0;
    action_mask.can_shoot = 0;

    // Offensive actions
    if (in_possession.hasBall == 1)
    {
        action_mask.can_pass = 1;
        action_mask.can_shoot = 1;
    }
    
    if (gameState.inboundingInProgress == 1)
    {
        action_mask.can_shoot = 0;
        action_mask.can_grab = 0;
        if (inbounding.imInbounding == 1 && gameState.liveBall == 0)
        {
            action_mask.can_move = 0;
        }
    }
    
    if (grab_cooldown.cooldown > 0.f)
    {
        action_mask.can_grab = 0;
    }

    // ======================== FOR TAG ==========================
    action_mask.can_pass = 0;
    // action_mask.can_shoot = 0;
    action_mask.can_grab = 0;
    // if (gameState.teamInPossession == team.teamIndex) 
    // {
    //     action_mask.can_move = 0;
    // }
}



inline void agentCollisionSystem(Engine &ctx,
                                 Entity entity_a,
                                 Position &pos_a,
                                 Orientation &orient_a,
                                 Reward &reward,
                                 Team &team)
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
            // ======================== FOR TAG ==========================
            GameState gameState = ctx.singleton<GameState>();

            Reward &entity_b_reward = ctx.get<Reward>(entity_b);
            if (gameState.teamInPossession == team.teamIndex)
            {
                reward.r -= 10;
                entity_b_reward.r += 10;
                ctx.singleton<WorldClock>().resetNow = 1.0f;
            }

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
    Attributes &defender_attributes,
    Orientation &defender_orientation)
    {
        GameState &gameState = ctx.singleton<GameState>();
        
        if (gameState.teamInPossession == defender_team.teamIndex)
        {
            defender_action.move = 0;
            return;
        }
        
        defender_action.grab = 1.f;
        Vector3 guarding_pos; // The place we want our defensive agent to go to defend
        int32_t found_offender = 0;
        for (CountT i = 0; i < NUM_AGENTS; i++) {
            Entity agent = ctx.data().agents[i];
            InPossession &offender_in_possession = ctx.get<InPossession>(agent);
            Position &offender_pos = ctx.get<Position>(agent);
            if (offender_in_possession.hasBall == 1 && found_offender == 0) {
                for (CountT j = 0; j < NUM_HOOPS; j++) {
                    Entity hoop = ctx.data().hoops[j];
                    if (defender_team.defendingHoopID == hoop.id) {
                    Position &hoop_pos = ctx.get<Position>(hoop);
                    Vector3 hoop_direction = hoop_pos.position - offender_pos.position;
                    if (hoop_direction.length2() > 1e-6f) {
                        guarding_pos = offender_pos.position + GUARDING_DISTANCE * hoop_direction.normalize();
                    } else {
                        guarding_pos = offender_pos.position;
                    }
                    found_offender = 1;
                }
            }
        }
    }
    
    if (found_offender == 0)
    {
        defender_action.move = 0;
        return;
    }
    
    Vector3 current_target = defender_attributes.currentTargetPosition;
    Vector3 ideal_target = guarding_pos;
    float interpolation_factor = defender_attributes.reactionSpeed * TIMESTEPS_TO_SECONDS_FACTOR;
    
    defender_attributes.currentTargetPosition = current_target + (ideal_target - current_target) * interpolation_factor;
    
    
    
    Vector3 move_vector = defender_attributes.currentTargetPosition - defender_pos.position;
    if (move_vector.length2() < 0.01f)
    {
        defender_action.move = 0;
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
    defender_action.move = 1;
    defender_action.moveAngle = best_move_angle;


    // make defender face basketball
    Vector3 defender_orientation_as_vec = defender_orientation.orientation.rotateVec(AGENT_BASE_FORWARD);
    float angle_between_vectors = acos(clamp(defender_orientation_as_vec.dot(move_vector.normalize()), -1.f, 1.f)); // cos(angle_between_vecs) = vec1 \cdot vec2 / (||vec1|| ||vec2||) adn we're isolating angle_between_vec
    if (angle_between_vectors > pi/8.f)
    {
        float direction_cross = defender_orientation_as_vec.x * move_vector.y - defender_orientation_as_vec.y * move_vector.x;
        if (direction_cross < 0) {defender_action.rotate = -1.f;}
        else if (direction_cross > 0) {defender_action.rotate = 1.f;}
        else {defender_action.rotate = 0.f;}
    }
    else {defender_action.rotate = 0.f;}
}


inline void updateCurrentShotPercentage(Engine &ctx,
                                        Attributes &attributes,
                                        Position &agent_pos,
                                        Velocity &agent_vel,
                                        InPossession &in_possession,
                                        Team &agent_team)
{
    if (in_possession.hasBall == 0) 
    {
        attributes.currentShotPercentage = 0.f;
        return;
    }

    Position hoop_positions[NUM_HOOPS];
    int32_t hoop_ids[NUM_HOOPS];
    for (CountT i = 0; i < NUM_HOOPS; i++) 
    {
        Entity hoop = ctx.data().hoops[i];
        hoop_positions[i] = ctx.get<Position>(hoop);
        hoop_ids[i] = hoop.id;
    };

    Position attacking_hoop_pos = (hoop_ids[0] != agent_team.defendingHoopID) ? hoop_positions[0] : hoop_positions[1];

    float dist_to_hoop = (attacking_hoop_pos.position - agent_pos.position).length();
    float distance_to_nearest_defender = std::numeric_limits<float>::infinity();
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity defender = ctx.data().agents[i];
        Team &defender_team = ctx.get<Team>(defender);
        Position &defender_pos = ctx.get<Position>(defender);
        if (defender_team.teamIndex != agent_team.teamIndex)
        {
            float dist_to_def = (agent_pos.position - defender_pos.position).length();
            if (dist_to_def < distance_to_nearest_defender)
            {
                distance_to_nearest_defender = dist_to_def;
            }
        }
    };

    float dist_stddev = DIST_DEVIATION_PER_METER * dist_to_hoop;
    float def_stddev = DEF_DEVIATION_PER_METER / distance_to_nearest_defender + .0001f;
    float vel_stddev = VEL_DEVIATION_FACTOR * agent_vel.velocity.length();
    // In order to get the normal distribution that will give us the probability of the shot going in,
    // we are getting the standard deviation (sqrt of variance) by square rooting the sum of the variances of 
    // each uniform distribution which is the stddev^2/3
    float final_stddev = std::sqrt((dist_stddev*dist_stddev/3.f) + (def_stddev*def_stddev/3) + (vel_stddev*vel_stddev/3));

    float max_make_angle = std::atan(HOOP_SCORE_ZONE_SIZE/dist_to_hoop); // The greatest angle from ideal vec that could occur with a make
    float z_score = max_make_angle / final_stddev;
    attributes.currentShotPercentage = erf(z_score / std::sqrt(2.f));
}

inline void rewardSystem(Engine &ctx,
                         Entity agent_entity,
                         Reward &reward,
                         Position &agent_pos,
                         Team &team,
                         InPossession &in_possession,
                         Attributes &attributes)
{
    GameState &gameState = ctx.singleton<GameState>();
    Entity other_agent;
    for (CountT i = 0; i < NUM_AGENTS; i++)
    {
        if (ctx.data().agents[i].id != agent_entity.id) {other_agent = ctx.data().agents[i];}
    }
    float dist_to_other_agent = (ctx.get<Position>(other_agent).position - agent_pos.position).length();
    if (team.teamIndex == gameState.teamInPossession)
    {
        // ================== FOR NORMAL BASKETBALL ========================
        // Find attacking hoop
        if (gameState.gameClock > 5.f)
        {
            Position target_hoop_pos;
            for (CountT i = 0; i < NUM_HOOPS; i++)
            {
                Entity hoop = ctx.data().hoops[i];
                if (hoop.id != team.defendingHoopID)
                {
                    target_hoop_pos = ctx.get<Position>(hoop);
                }
            }

            // Find agent who shot the ball and reward them if the shot is going in
            for (CountT j = 0; j < NUM_BASKETBALLS; j++)
            {
                Entity ball = ctx.data().balls[j];
                BallPhysics &ball_physics = ctx.get<BallPhysics>(ball);
                if (ball_physics.shotByAgentID == agent_entity.id && ball_physics.shotIsGoingIn == 1)
                {
                    reward.r += ball_physics.shotPointValue;
                }
                else if (ball_physics.shotByAgentID == agent_entity.id && ball_physics.shotIsGoingIn == 0 && ball_physics.inFlight == 1)
                {
                    reward.r -= 1;
                }
            }
            
            reward.r += attributes.currentShotPercentage;
        }


        // ======================== FOR TAG ==========================
        // reward.r += in_possession.hasBall;
        // reward.r -= (exp(-0.4f * dist_to_other_agent));
    }
    else
    {
        reward.r -= 1.f;
        reward.r += exp(-0.4f * dist_to_other_agent);
    }
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
        Velocity &ball_velocity = ctx.get<Velocity>(ball);
        float distance_to_hoop = std::sqrt((ball_pos.position.x - hoop_pos.position.x) * (ball_pos.position.x - hoop_pos.position.x) +
                                        (ball_pos.position.y - hoop_pos.position.y) * (ball_pos.position.y - hoop_pos.position.y));

        if (distance_to_hoop <= scoring_zone.radius && ball_physics.inFlight == 1.f) {
            // Use the point value that was calculated when the shot was taken
            int32_t points_scored = ball_physics.shotPointValue;

            // Find which team is defending this hoop (has defendingHoopID == hoop_entity.id)
            int32_t inbounding_team_idx = 0; // Default fallback
            for (CountT j = 0; j < NUM_AGENTS; j++) {
                Entity agent = ctx.data().agents[j];
                Team &team = ctx.get<Team>(agent);
                Stats &agent_stats = ctx.get<Stats>(agent);
                if (team.defendingHoopID == hoop_entity.id)
                {
                    inbounding_team_idx = team.teamIndex;
                }

                if (agent.id == ball_physics.shotByAgentID)
                {
                    agent_stats.points += (team.defendingHoopID == hoop_entity.id) ? -ball_physics.shotPointValue : ball_physics.shotPointValue;
                }
            }

            Position inbound_spot;
            Quat inbound_orientation;

            if (hoop_entity.id == gameState.team0Hoop)
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
            ball_physics.inFlight = 0.0f;
            ball_velocity.velocity = Vector3::zero();

            // Clear shot information since the shot scored
            ball_physics.shotByAgentID = ENTITY_ID_PLACEHOLDER;
            ball_physics.shotByTeamID = ENTITY_ID_PLACEHOLDER;
            ball_physics.shotPointValue = 2; // Reset to default
            ball_physics.shotIsGoingIn = 0.0f;

            // Set up the inbound for the defending team.
            if (gameState.isOneOnOne == 0.f)
            {
                ball_pos = inbound_spot;
                inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
                assignInbounder(ctx, ball, inbound_spot, inbounding_team_idx, inbound_orientation, false);
            }
            else 
            {
                ctx.singleton<WorldClock>().resetNow = 1.0f;
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
    ctx.singleton<WorldClock>().resetNow = 0.0f;
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
        // ======================== FOR TAG ==========================
        Entity off_agent = ctx.data().agents[0];
        Entity cur_agent;
        for (CountT i = 1; i < NUM_AGENTS; i++)
        {
            cur_agent = ctx.data().agents[i];
            if (ctx.get<Team>(cur_agent).teamIndex == gameState.teamInPossession) {off_agent = cur_agent;}
        }
        ctx.get<Reward>(off_agent).r += 10.f;
        world_clock.resetNow = 1.0f;
    }

    


    if (gameState.shotClock < 0.f)
    {
        gameState.shotClock = 0.f;
    }
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
            ball_physics.lastTouchedByAgentID = agent.id;
            ball_physics.lastTouchedByTeamID = team.teamIndex;
        }
    }
}



inline void outOfBoundsSystem(Engine &ctx,
                            Entity ball_entity,
                            Position &ball_pos,
                            BallPhysics &ball_physics,
                            Velocity &ball_velocity)
{
    GameState &gameState = ctx.singleton<GameState>();

    // Check if the ball's center has crossed the court boundaries and we are not currently inbounding

    if ((ball_pos.position.x < COURT_MIN_X || ball_pos.position.x > COURT_MAX_X ||
        ball_pos.position.y < COURT_MIN_Y || ball_pos.position.y > COURT_MAX_Y) &&
        gameState.inboundingInProgress == 0.f)
    {
        if (gameState.isOneOnOne == 1.f) 
        {
            // ======================== FOR TAG ==========================
            Entity off_agent = ctx.data().agents[0];
            Entity cur_agent;
            for (CountT i = 1; i < NUM_AGENTS; i++)
            {
                cur_agent = ctx.data().agents[i];
                if (ctx.get<Team>(cur_agent).teamIndex == gameState.teamInPossession) {off_agent = cur_agent;}
            }
            ctx.get<Reward>(off_agent).r -= 100.f;
            
            ctx.singleton<WorldClock>().resetNow = 1.0f;
        }
        else
        {
            ball_physics.inFlight = 0.0f;
            ball_velocity.velocity = Vector3::zero();
            gameState.liveBall = 0;

            // The team that did NOT last touch the ball gets possession.
            int32_t new_team_idx = 1 - ball_physics.lastTouchedByTeamID;

            // Find the player who had the ball and reset their position
            for (CountT i = 0; i < NUM_AGENTS; i++) {
                Entity agent = ctx.data().agents[i];
                InPossession &in_possession = ctx.get<InPossession>(agent);
                Position &agent_pos = ctx.get<Position>(agent);
                // If this agent was the one who went out of bounds with the ball...
                if (in_possession.hasBall == 1 && in_possession.ballEntityID == ball_entity.id)
                {
                    agent_pos.position += findVectorToCenter(ctx, agent_pos);

                    // Take the ball away
                    in_possession.hasBall = 0;
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

    int32_t current_team_idx = gameState.teamInPossession;
    int32_t new_team_idx = 1 - current_team_idx;
    int32_t ball_to_turnover_id = ENTITY_ID_PLACEHOLDER;

    gameState.liveBall = 0;

    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Inbounding &inb = ctx.get<Inbounding>(agent);
        Position &agent_pos = ctx.get<Position>(agent);
        InPossession &poss = ctx.get<InPossession>(agent);
        if (inb.imInbounding > 0.5f) {
            ball_to_turnover_id = poss.ballEntityID;

            inb.imInbounding = 0.0f;
            poss.hasBall = 0.0f;
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
                grabbed = {0, ENTITY_ID_PLACEHOLDER};
                Quat inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
                assignInbounder(ctx, ball, ball_pos, new_team_idx, inbound_orientation, true);
            }
        }
    }
}


// This is a helper struct used ONLY by the fillObservationsSystem
// to gather all agent data before sorting it into the observation vector.
struct AgentObservationData {
    int32_t id;
    int32_t teamID;
    Position pos;
    Orientation orient;
    Velocity velocity;
    InPossession in_pos;
    Inbounding inb;
    GrabCooldown cooldown;
    Attributes attributes;
};


inline void fillObservationsSystem(Engine &ctx,
                                   Entity agent_entity,
                                   Observations &observations,
                                   Position &agent_pos,
                                   Orientation &agent_orientation,
                                   InPossession &in_possession,
                                   Inbounding &inbounding,
                                   Team &agent_team,
                                   GrabCooldown &grab_cooldown,
                                   Velocity &agent_vel,
                                   Attributes &agent_attributes)
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
    Velocity ball_velocity;

    // we assume 1 ball
    Entity ball = ctx.data().balls[0];
    ball_pos = ctx.get<Position>(ball);
    ball_phys = ctx.get<BallPhysics>(ball);
    ball_grabbed = ctx.get<Grabbed>(ball);
    ball_velocity = ctx.get<Velocity>(ball);

    // --- Hoop Positions ---
    Position hoop_positions[NUM_HOOPS];
    int32_t hoop_ids[NUM_HOOPS];
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
        Velocity &v = ctx.get<Velocity>(agent);
        InPossession &ip = ctx.get<InPossession>(agent);
        Inbounding &ib = ctx.get<Inbounding>(agent);
        GrabCooldown &gc = ctx.get<GrabCooldown>(agent);
        Attributes &attr = ctx.get<Attributes>(agent);
        all_agents[agent_idx++] = { agent.id, t.teamIndex, p, o, v, ip, ib, gc, attr };
        if (ib.imInbounding > 0.5f) {
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
    fill_vec3(ball_velocity.velocity);
    obs[idx++] = ball_grabbed.isGrabbed;
    obs[idx++] = ball_phys.inFlight;
    obs[idx++] = (float)ball_phys.shotPointValue;
    obs[idx++] = (float)ball_phys.lastTouchedByTeamID;

    // Hoop Info
    Position attacking_hoop_pos = (hoop_ids[0] != agent_team.defendingHoopID) ? hoop_positions[0] : hoop_positions[1];
    Position defending_hoop_pos = (hoop_ids[0] == agent_team.defendingHoopID) ? hoop_positions[0] : hoop_positions[1];
    fill_vec3(attacking_hoop_pos.position);
    fill_vec3(defending_hoop_pos.position);

    // Self Data
    fill_vec3(agent_pos.position);                                                                                                              // 3
    fill_vec3(Vector3::zero());                                                                                                                 // 3
    obs[idx++] = 0.f;                                                                                                                           // 1
    fill_quat(agent_orientation.orientation);                                                                                                   // 4
    Vector3 agent_orient_as_vec = (agent_orientation.orientation).rotateVec(AGENT_BASE_FORWARD);                                    
    fill_vec3(agent_orient_as_vec);                                                                                                             // 3
    if (agent_vel.velocity.length2() > 1e-6f) {fill_vec3(agent_vel.velocity.normalize());}                                                      // 3
    else {fill_vec3(Vector3::zero());}
    obs[idx++] = agent_vel.velocity.length();                                                                                                   // 1
    float dot_between_orient_and_vel = 0.f;
    if (agent_vel.velocity.length2() > 1e-6f) {dot_between_orient_and_vel = agent_vel.velocity.normalize().dot(agent_orient_as_vec);}
    obs[idx++] = dot_between_orient_and_vel; // Dot product between orientation vec and velocity that determines acceleration                   // 1
    obs[idx++] = (dot_between_orient_and_vel <= 0.8f) ? 0.1f : 1.f; // Acceleration based on the dot product ^                                  // 1
    Vector3 dir_to_hoop = (attacking_hoop_pos.position - agent_pos.position);
    float dist_to_hoop = dir_to_hoop.length();
    if (dist_to_hoop > 1e-6f) {fill_vec3(dir_to_hoop.normalize());}                                                                             // 3
    else {fill_vec3(Vector3{0.f, 0.f, 0.f});}
    obs[idx++] = dist_to_hoop; // distance to hoop                                                                                              // 1    
    Vector3 dir_to_ball = (ball_pos.position - agent_pos.position);
    float dist_to_ball = dir_to_ball.length();
    if (dist_to_ball > 1e-6f) {fill_vec3(dir_to_ball.normalize());}                                                                             // 3
    else {fill_vec3(Vector3{0.f, 0.f, 0.f});}
    obs[idx++] = dist_to_ball; // distance to ball                                                                                              // 11
    obs[idx++] = inbounding.imInbounding;
    obs[idx++] = grab_cooldown.cooldown;
    obs[idx++] = agent_attributes.maxSpeed;
    obs[idx++] = agent_attributes.quickness;
    obs[idx++] = agent_attributes.shooting;
    obs[idx++] = agent_attributes.freeThrowPercentage;
    obs[idx++] = agent_attributes.reactionSpeed;
    obs[idx++] = agent_attributes.currentShotPercentage;
    obs[idx++] = (float)in_possession.pointsWorth;
    obs[idx++] = in_possession.hasBall;
    

    // Teammate & Opponent Data
    int teammate_count = 0;
    int opponent_count = 0;
    const int max_teammates = (NUM_AGENTS / 2) - 1;
    const int max_opponents = NUM_AGENTS / 2;

    for (int i = 0; i < agent_idx; i++) {
        if (all_agents[i].id == agent_entity.id) continue;

        Vector3 vec_to_agent = all_agents[i].pos.position - agent_pos.position;
        if (all_agents[i].teamID == agent_team.teamIndex)
        {
            if (teammate_count < max_teammates)
            {
                fill_vec3(all_agents[i].pos.position);
                if (vec_to_agent.length2() > 1e-6f) {fill_vec3(vec_to_agent.normalize());} 
                else {fill_vec3(Vector3{0.f, 0.f, 0.f});}
                obs[idx++] = vec_to_agent.length();
                fill_quat(all_agents[i].orient.orientation);
                Vector3 teammate_orient_as_vec = all_agents[i].orient.orientation.rotateVec(AGENT_BASE_FORWARD);
                fill_vec3(teammate_orient_as_vec);
                if (all_agents[i].velocity.velocity.length2() > 1e-6f) {fill_vec3(all_agents[i].velocity.velocity.normalize());}
                else {fill_vec3(Vector3::zero());}
                obs[idx++] = all_agents[i].velocity.velocity.length(); // Speed
                float teammate_dot_between_orient_and_vel = 0.f;
                if (all_agents[i].velocity.velocity.length2() > 1e-6f) {teammate_dot_between_orient_and_vel = all_agents[i].velocity.velocity.normalize().dot(teammate_orient_as_vec);}
                obs[idx++] = teammate_dot_between_orient_and_vel; // Dot product between orientation vec and velocity
                obs[idx++] = (teammate_dot_between_orient_and_vel <= 0.8f) ? 0.1f : 1.f; // Acceleration based on dot ^
                Vector3 teammate_dir_to_hoop = (attacking_hoop_pos.position - all_agents[i].pos.position);
                float teammate_dist_to_hoop = teammate_dir_to_hoop.length();
                if (teammate_dist_to_hoop > 1e-6f) {fill_vec3(teammate_dir_to_hoop.normalize());} 
                else {fill_vec3(Vector3{0.f, 0.f, 0.f});}
                obs[idx++] = teammate_dist_to_hoop;
                Vector3 teammate_dir_to_ball = (ball_pos.position - all_agents[i].pos.position);
                float teammate_dist_to_ball = teammate_dir_to_ball.length();
                if (teammate_dist_to_ball > 1e-6f) {fill_vec3(teammate_dir_to_ball.normalize());} 
                else {fill_vec3(Vector3{0.f, 0.f, 0.f});}
                obs[idx++] = teammate_dist_to_ball;
                obs[idx++] = all_agents[i].inb.imInbounding;
                obs[idx++] = all_agents[i].cooldown.cooldown;
                obs[idx++] = all_agents[i].attributes.maxSpeed;
                obs[idx++] = all_agents[i].attributes.quickness;
                obs[idx++] = all_agents[i].attributes.shooting;
                obs[idx++] = all_agents[i].attributes.freeThrowPercentage;
                obs[idx++] = all_agents[i].attributes.reactionSpeed;
                obs[idx++] = all_agents[i].attributes.currentShotPercentage;
                obs[idx++] = (float)all_agents[i].in_pos.pointsWorth;
                obs[idx++] = all_agents[i].in_pos.hasBall;
                teammate_count++;
            }
        }   
        else
        {
            if (opponent_count < max_opponents)
            {                
                fill_vec3(all_agents[i].pos.position);
                if (vec_to_agent.length2() > 1e-6f) {fill_vec3(vec_to_agent.normalize());} 
                else {fill_vec3(Vector3{0.f, 0.f, 0.f});}
                obs[idx++] = vec_to_agent.length();
                fill_quat(all_agents[i].orient.orientation);
                Vector3 opponent_orient_as_vec = all_agents[i].orient.orientation.rotateVec(AGENT_BASE_FORWARD);
                fill_vec3(opponent_orient_as_vec);
                if (all_agents[i].velocity.velocity.length2() > 1e-6f) {fill_vec3(all_agents[i].velocity.velocity.normalize());}
                else {fill_vec3(Vector3::zero());}
                obs[idx++] = all_agents[i].velocity.velocity.length(); // Speed
                float opponent_dot_between_orient_and_vel = 0.f;
                if (all_agents[i].velocity.velocity.length2() > 1e-6f) {opponent_dot_between_orient_and_vel = all_agents[i].velocity.velocity.normalize().dot(opponent_orient_as_vec);}
                obs[idx++] = opponent_dot_between_orient_and_vel; // Dot product between orientation vec and velocity
                obs[idx++] = (opponent_dot_between_orient_and_vel <= 0.8f) ? 0.1f : 1.f; // Acceleration based on dot ^
                // Safe direction to hoop for opponent (handle zero-length case)
                Vector3 opponent_dir_to_hoop = (defending_hoop_pos.position - all_agents[i].pos.position);
                float opponent_dist_to_hoop = opponent_dir_to_hoop.length();
                if (opponent_dist_to_hoop > 1e-6f) {
                    fill_vec3(opponent_dir_to_hoop.normalize());
                } else {
                    fill_vec3(Vector3{0.f, 0.f, 0.f}); // Safe default direction
                }
                obs[idx++] = opponent_dist_to_hoop;
                // Safe direction to ball for opponent (handle zero-length case)
                Vector3 opponent_dir_to_ball = (ball_pos.position - all_agents[i].pos.position);
                float opponent_dist_to_ball = opponent_dir_to_ball.length();
                if (opponent_dist_to_ball > 1e-6f) {
                    fill_vec3(opponent_dir_to_ball.normalize());
                } else {
                    fill_vec3(Vector3{0.f, 0.f, 0.f}); // Safe default direction
                }
                obs[idx++] = opponent_dist_to_ball;
                obs[idx++] = all_agents[i].inb.imInbounding;
                obs[idx++] = all_agents[i].cooldown.cooldown;
                obs[idx++] = all_agents[i].attributes.maxSpeed;
                obs[idx++] = all_agents[i].attributes.quickness;
                obs[idx++] = all_agents[i].attributes.shooting;
                obs[idx++] = all_agents[i].attributes.freeThrowPercentage;
                obs[idx++] = all_agents[i].attributes.reactionSpeed;
                obs[idx++] = all_agents[i].attributes.currentShotPercentage;
                obs[idx++] = (float)all_agents[i].in_pos.pointsWorth;
                obs[idx++] = all_agents[i].in_pos.hasBall;
                opponent_count++;
            }
        }
    }


    // Padding for agent data
    constexpr int agent_feature_size = 3 + 3 + 1 + 4 + 3 + 3 + 1 + 1 + 1 + 3 + 1 + 3 + 11; // 37 total: Pos, VecTo, DistanceTo, Orient, orientAsVec, Velocity, Speed, VelOrientDot, acceleration, DirToHoop, DistToHoop, DirToBall, DistToBall, inbounding, cooldown, attributes, pointsWorth, HasBall
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
        ActionMask, GrabCooldown, InPossession, Inbounding, Team>>({tickNode});

    auto moveAgentSystemNode = builder.addToGraph<ParallelForNode<Engine, moveAgentSystem,
        Action, ActionMask, Position, InPossession, Inbounding, Orientation, Attributes, Velocity>>({actionMaskingNode});

    auto grabSystemNode = builder.addToGraph<ParallelForNode<Engine, grabSystem,
        Entity, Action, ActionMask, Position, InPossession, Team, GrabCooldown>>({moveAgentSystemNode});

    auto passSystemNode = builder.addToGraph<ParallelForNode<Engine, passSystem,
        Entity, Action, ActionMask, Orientation, InPossession, Inbounding>>({grabSystemNode});

    auto shootSystemNode = builder.addToGraph<ParallelForNode<Engine, shootSystem,
        Entity, Action, ActionMask, Position, Orientation, Inbounding, InPossession, Team, Reward, Velocity>>({passSystemNode});

    auto moveBallSystemNode = builder.addToGraph<ParallelForNode<Engine, moveBallSystem,
        Position, Grabbed, Velocity>>({shootSystemNode});

    auto updateCurrentShotPercentageNode = builder.addToGraph<ParallelForNode<Engine, updateCurrentShotPercentage,
        Attributes, Position, Velocity, InPossession, Team>>({moveBallSystemNode});

    auto scoreSystemNode = builder.addToGraph<ParallelForNode<Engine, scoreSystem,
        Entity, Position, ScoringZone>>({updateCurrentShotPercentageNode});

    auto outOfBoundsSystemNode = builder.addToGraph<ParallelForNode<Engine, outOfBoundsSystem,
        Entity, Position, BallPhysics, Velocity>>({scoreSystemNode});

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
        Entity, Position, Orientation, Reward, Team>>({updatePointsWorthNode});

    auto hardCodeDefenseSystemNode = builder.addToGraph<ParallelForNode<Engine, hardCodeDefenseSystem,
        Team, Position, Action, Attributes, Orientation>>({agentCollisionNode});

    auto fillObservationsNode = builder.addToGraph<ParallelForNode<Engine, fillObservationsSystem,
        Entity, Observations, Position, Orientation, InPossession,
        Inbounding, Team, GrabCooldown, Velocity, Attributes>>({hardCodeDefenseSystemNode});

    auto rewardSystemNode = builder.addToGraph<ParallelForNode<Engine, rewardSystem,
        Entity, Reward, Position, Team, InPossession, Attributes>>({fillObservationsNode});

    return rewardSystemNode;
}

}
