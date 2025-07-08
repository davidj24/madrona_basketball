#include "game.hpp"
#include "types.hpp"
#include "constants.hpp"
#include "helper.hpp"

using namespace madrona;
using namespace madrona::math;



namespace madBasketball {
inline void assignInbounder(Engine &ctx, Entity ball_entity, Position ball_pos, uint32_t new_team_idx, Quat new_orientation, bool is_oob)
{
    GameState &gameState = ctx.singleton<GameState>();
    bool inbounder_assigned = false;

    // Find the first available player on the new team.
    ctx.iterateQuery(ctx.data().agentQuery,
        [&](Entity agent_entity, Team &agent_team, InPossession &in_possession,
            Position &agent_pos, Orientation &agent_orient, Inbounding &inbounding, GrabCooldown &cooldown,
            Reset &, Action &, ActionMask &, Reward &, Done &, CurStep &, Stats &, Attributes &)
    {
        if ((uint32_t)agent_team.teamIndex == new_team_idx && !inbounder_assigned)
        {
            inbounder_assigned = true;
            inbounding.imInbounding = true;
            agent_pos = ball_pos; // Move player to the ball

            // Give them possession of the ball
            ctx.get<Grabbed>(ball_entity) = {true, (uint32_t)agent_entity.id};
            in_possession.hasBall = true;
            in_possession.ballEntityID = ball_entity.id;

            // Set the agent's orientation to face the court
            agent_orient.orientation = new_orientation;
        }
    });

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
    ctx.iterateQuery(ctx.data().agentQuery,
        [&](Entity &agent_entity, Team &, InPossession &in_possession,
            Position &agent_pos, Orientation &, Inbounding &, GrabCooldown &,
            Reset &, Action &, ActionMask &, Reward &, Done &, CurStep &, Stats &, Attributes &)
    {
        // Make the ball move with the agent if it's held
        bool agent_is_holding_this_ball = (in_possession.hasBall == true &&
                                            grabbed.isGrabbed &&
                                            grabbed.holderEntityID == (uint32_t)agent_entity.id);
        if (agent_is_holding_this_ball) {
            ball_pos = agent_pos;  // Move basketball to agent's new position
        }
    });

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
    Position hoop_positions[NUM_HOOPS];
    uint32_t hoop_ids[NUM_HOOPS];
    int hoop_idx = 0;
    ctx.iterateQuery(ctx.data().hoopQuery,
        [&](Entity hoop_entity, Position &hoop_pos, ImAHoop &,
            Reset &, Done &, CurStep &, ScoringZone &) {
            if (hoop_idx < NUM_HOOPS) {
                hoop_positions[hoop_idx] = hoop_pos;
                hoop_ids[hoop_idx] = hoop_entity.id;
                hoop_idx++;
            }
    });

    // Find the hoop this agent should be shooting at (opposing team's hoop)
    Position target_hoop_pos{};
    bool found_target_hoop = false;
    for (int i = 0; i < hoop_idx; i++) {
        if (hoop_ids[i] != team.defendingHoopID) {
            target_hoop_pos = hoop_positions[i];
            found_target_hoop = true;
            break;
        }
    }

    // Calculate points worth for this agent's current position
    if (found_target_hoop) {
        in_possession.pointsWorth = getShotPointValue(agent_pos, target_hoop_pos);
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
    if (action_mask.can_grab == 0.f || action.grab == 0) {return;}
    grab_cooldown.cooldown = 10.f;
    action.grab = 0.f;        ctx.iterateQuery(ctx.data().ballQuery,
            [&](Entity ball_entity,
                Position &basketball_pos,
                Grabbed &grabbed,
                BallPhysics &ball_physics,
                Reset &, Done &, CurStep &)
        {
        if (ball_physics.inFlight) {return;}
        bool agent_is_holding_this_ball = (in_possession.hasBall == true &&
                                            grabbed.isGrabbed &&
                                            grabbed.holderEntityID == (uint32_t)agent_entity.id);

        // If agent already has a ball, drop it
        if (agent_is_holding_this_ball)
        {
            in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
            in_possession.hasBall = false;
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER;
            grabbed.isGrabbed = false;
            return;
        }

        // Check if ball is within grab range (0.5 meters)
        float distance_between_ball_and_player = sqrt((basketball_pos.position.x - agent_pos.position.x) * (basketball_pos.position.x - agent_pos.position.x) +
                            (basketball_pos.position.y - agent_pos.position.y) * (basketball_pos.position.y - agent_pos.position.y));

        if (distance_between_ball_and_player <= 0.3f)
        {                                              
            ctx.iterateQuery(ctx.data().agentQuery, [&] (Entity &, Team &, InPossession &other_in_possession, Position &, Orientation &, Inbounding &, GrabCooldown &robbed_agent_grab_cooldown, Reset &, Action &, ActionMask &, Reward &, Done &, CurStep &, Stats &, Attributes &)
            {
                if (other_in_possession.ballEntityID == (uint32_t)ball_entity.id) // if we're stealing from another agent
                {
                    other_in_possession.hasBall = false;
                    other_in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
                    robbed_agent_grab_cooldown.cooldown = SIMULATION_HZ;
                    if (gameState.isOneOnOne == 1.f)
                    {
                        ctx.iterateQuery(ctx.data().worldClockQuery, [&](Reset &world_reset, IsWorldClock &lalala) {
                            world_reset.resetNow = 1;
                        });
                    }
                }
            });

            in_possession.hasBall = true;
            in_possession.ballEntityID = ball_entity.id;
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
    });

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




    ctx.iterateQuery(ctx.data().ballQuery, [&] (Entity, Position &, Grabbed &grabbed, BallPhysics &ball_physics, Reset &, Done &, CurStep &)
    {
        if (grabbed.holderEntityID == agent_entity.id)
        {
            grabbed.isGrabbed = false;  // Ball is no longer grabbed
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER; // Ball is no longer held by anyone
            in_possession.hasBall = false; // Since agents can only hold 1 ball at a time, if they pass it they can't be holding one anymore
            in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER; // Whoever passed the ball is no longer in possession of it
            inbounding.imInbounding = false;
            ball_physics.velocity = agent_orientation.orientation.rotateVec(Vector3{0.f, 0.1f, 0.f}); // Setting the ball's velocity to have the same direction as the agent's orientation
                                                                                                      // Note: we use 0, 0.1, 0 because that's forward in our simulation specifically
            gameState.inboundingInProgress = 0.0f;
        }
    });
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
    Position attacking_hoop_pos = {0.f, 0.f, 0.f};
    ctx.iterateQuery(ctx.data().hoopQuery, [&](Entity hoop_entity, Position &hoop_pos, ImAHoop &, Reset &, Done &, CurStep &, ScoringZone &)
    {
        if ((uint32_t)hoop_entity.id != team.defendingHoopID)
        {
            attacking_hoop_pos = hoop_pos;
            return;
        }
    });

    // Calculate vector to attacking hoop
    Vector3 shot_vector = Vector3{
        attacking_hoop_pos.position.x - agent_pos.position.x,
        attacking_hoop_pos.position.y - agent_pos.position.y,
        0.f
    };


    // Calculate intended angle towards hoop
    float intended_direction = std::atan2(shot_vector.x, shot_vector.y);

    // ======================== DEVIATION TUNERS ==============================
    float dist_deviation_per_meter = .2f;
    float def_deviation_per_meter = .1f;
    float vel_deviation_factor = 2.f;


    // 1. Mess up angle based on distance
    float distance_to_hoop = shot_vector.length();
    float dist_stddev = dist_deviation_per_meter/100 * distance_to_hoop;
    float deviation_from_distance = sampleUniform(ctx, -dist_stddev, dist_stddev);


    // 2. Mess up angle based on contest level (how close nearest defender is)
    float deviation_from_defender = 0.0f;
    float distance_to_nearest_defender = std::numeric_limits<float>::infinity();
    ctx.iterateQuery(ctx.data().agentQuery, [&](Entity, Team &defender_team, InPossession, Position &defender_pos, Orientation, Inbounding, GrabCooldown, Reset &, Action &, ActionMask &, Reward &, Done &, CurStep &, Stats &, Attributes &)
    {
        if (defender_team.teamIndex != team.teamIndex)
        {
            Vector3 diff = agent_pos.position - defender_pos.position;
            float dist_to_def = diff.length();
            if (dist_to_def < distance_to_nearest_defender)
            {
                distance_to_nearest_defender = dist_to_def;
            }
        }
    });

    if (distance_to_nearest_defender < 2.0f) { // Only apply pressure if defender is  close
        float def_stddev = (def_deviation_per_meter/100) / (distance_to_nearest_defender + 0.1f);
        deviation_from_defender = sampleUniform(ctx, -def_stddev, def_stddev);
    }


    // 3. Mess up angle based on agent velocity
    float deviation_from_velocity = 0.0f;
    if (action.moveSpeed > 0) {
        float vel_stddev = vel_deviation_factor/100 * action.moveSpeed;
        deviation_from_velocity = sampleUniform(ctx, -vel_stddev, vel_stddev);
    }

    // Combine all deviations and apply to the final shot direction
    float total_deviation = deviation_from_distance + deviation_from_defender + deviation_from_velocity;
    float shot_direction = intended_direction + total_deviation;

    // This is the final, correct trajectory vector for the ball - Preserved from your code
    Vector3 final_shot_vec = {sinf(shot_direction), cosf(shot_direction), 0.f};


    const Vector3 base_forward = {0.0f, 1.0f, 0.0f};


    // Find the rotation that aligns the agent's orientation with the final shot direction vector.
    agent_orientation.orientation = findRotationBetweenVectors(base_forward, final_shot_vec);


    // Shoot the damn ball
    ctx.iterateQuery(ctx.data().ballQuery, [&] (Entity, Position &, Grabbed &grabbed, BallPhysics &ball_physics, Reset &, Done &, CurStep &)
    {
        if (grabbed.holderEntityID == agent_entity.id)
        {
            // Calculate the point value of this shot from the agent's current position
            int32_t shot_point_value = getShotPointValue(agent_pos, attacking_hoop_pos);

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
    });
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
                                 Position &entity_a_pos,
                                 InPossession &in_possession_a)
{
    // Query for all agents to get their positions.
    // We need Entity to compare IDs and Position to read/write locations.
    ctx.iterateQuery(ctx.data().agentQuery, [&](Entity entity_b, Team, InPossession, Position &entity_b_pos, Orientation, Inbounding, GrabCooldown, Reset &, Action &, ActionMask &, Reward &, Done &, CurStep &, Stats &, Attributes &)
    {

        // Don't check an agent against itself.
        // Only check pairs where A's ID is less than B's to avoid checking each pair twice.
        if (entity_a.id >= entity_b.id) {return;}

        // Calculate the vector and distance between the two agents
        Vector3 vec_between_agents = (entity_b_pos.position - entity_a_pos.position);
        float dist_between_agents = vec_between_agents.length();

        if (dist_between_agents < AGENT_SIZE_M)
        {
            // Calculate how much they are overlapping. If dist is 0, provide a small default push.
            float penetration_depth = AGENT_SIZE_M - dist_between_agents;
            Vector3 correction_vec = (dist_between_agents > 1e-5f) ? (vec_between_agents / dist_between_agents) : Vector3{1.f, 0.f, 0.f};

            // Move each agent away from the other by half of the overlap.
            // This "resolves" the collision by pushing them apart.
            entity_a_pos.position.x -= correction_vec.x * penetration_depth * 0.5f;
            entity_a_pos.position.y -= correction_vec.y * penetration_depth * 0.5f;

            entity_b_pos.position.x += correction_vec.x * penetration_depth * 0.5f;
            entity_b_pos.position.y += correction_vec.y * penetration_depth * 0.5f;
        }
    });
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
    Vector3 guarding_pos; // The place we want our defensive agent to go to to defend
    bool found_offender = false;
    ctx.iterateQuery(ctx.data().agentQuery, [&](Entity, Team, InPossession &offender_in_possession, Position &offender_pos, Orientation, Inbounding, GrabCooldown, Reset &, Action &, ActionMask &, Reward &, Done &, CurStep &, Stats &, Attributes &) {
        if (offender_in_possession.hasBall == 1.f && !found_offender)
        {
            ctx.iterateQuery(ctx.data().hoopQuery, [&] (Entity hoop_entity, Position &hoop_pos, ImAHoop &, Reset &, Done &, CurStep &, ScoringZone &)
            {
                if (defender_team.defendingHoopID == hoop_entity.id)
                {
                    guarding_pos = offender_pos.position + GUARDING_DISTANCE * (hoop_pos.position - offender_pos.position).normalize();
                    found_offender = true;
                }
            });
        }
    });

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

inline void resetRewardsSystem(Engine &ctx, Reward &rew)
{
    rew.r = 0.f;
}

//=================================================== Hoop Systems ===================================================
inline void scoreSystem(Engine &ctx,
                        Entity hoop_entity,
                        Position &hoop_pos,
                        ScoringZone &scoring_zone)
{
    GameState &gameState = ctx.singleton<GameState>();

    ctx.iterateQuery(ctx.data().ballQuery, [&] (Entity ball_entity, Position &ball_pos, Grabbed, BallPhysics &ball_physics, Reset &, Done &, CurStep &)
    {
        float distance_to_hoop = std::sqrt((ball_pos.position.x - hoop_pos.position.x) * (ball_pos.position.x - hoop_pos.position.x) +
                                        (ball_pos.position.y - hoop_pos.position.y) * (ball_pos.position.y - hoop_pos.position.y));

        if (distance_to_hoop <= scoring_zone.radius && ball_physics.inFlight)
        {
            // Use the point value that was calculated when the shot was taken
            int32_t points_scored = ball_physics.shotPointValue;

            // Find which team is defending this hoop (has defendingHoopID == hoop_entity.id)
            uint32_t inbounding_team_idx = 0; // Default fallback
            ctx.iterateQuery(ctx.data().agentQuery,
                [&](Entity agent_entity, Team &team, InPossession, Position,
                    Orientation, Inbounding, GrabCooldown, Reset &, Action &,
                    ActionMask &, Reward &rew, Done &, CurStep &, Stats &, Attributes &)
            {
                if (team.defendingHoopID == (uint32_t)hoop_entity.id)
                {
                    inbounding_team_idx = (uint32_t)team.teamIndex;
                    return; // Found the defending team
                }

                if (agent_entity.id == ball_physics.shotByAgentID)
                {
                    rew.r = (float) points_scored;
                    // We need to get Stats component for this agent, but it's not in the query
                    // For now, we'll skip updating individual agent stats
                    // TODO: Either add Stats to agentQuery or create a separate query
                }
            });

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

            // Set up the inbound for the defending team.
            if (gameState.isOneOnOne == 0.f)
            {
                ball_pos = inbound_spot;
                inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
                assignInbounder(ctx, ball_entity, inbound_spot, inbounding_team_idx, inbound_orientation, false);
            }
            else
            {
                ctx.iterateQuery(ctx.data().worldClockQuery, [&](Reset &world_reset, IsWorldClock &lalala) {
                    world_reset.resetNow = 1;
                });
            }
        }
    });
}


//=================================================== General Systems ===================================================
inline void resetSystem(Engine &ctx, Reset &world_reset, IsWorldClock &)
{
    // This system only executes its logic if the reset flag has been set.
    if (world_reset.resetNow == 0)
    {
        return;
    }

    GameState &gameState = ctx.singleton<GameState>();
    const GridState *grid = ctx.data().grid;

    // --- Part 1: Reset GameState Singleton ---
    if (gameState.gameClock <= 0.f && gameState.isOneOnOne == 0.f)
    {
        // End-of-quarter logic (only for full games)
        if (gameState.period < 4 || gameState.team0Score == gameState.team1Score)
        {
            gameState.period++;
            gameState.gameClock = TIME_PER_PERIOD;
            gameState.shotClock = 24.0f;
            gameState.liveBall = 1.0f;
            gameState.inboundingInProgress = 0.f;
        }
        else
        {
            gameState.liveBall = 0.f; // Game over
        }
    }
    else
    {
        // This was a manual reset or a reset after a score in 1v1 mode.
        gameState = GameState {
            .inboundingInProgress = 0.0f,
            .liveBall = 1.0f,
            .period = 1.0f,
            .teamInPossession = 0.0f,
            .team0Hoop = gameState.team0Hoop, // Preserve existing hoop IDs
            .team0Score = 0.0f,
            .team1Hoop = gameState.team1Hoop, // Preserve existing hoop IDs
            .team1Score = 0.0f,
            .gameClock = TIME_PER_PERIOD,
            .shotClock = 24.0f,
            .scoredBaskets = 0.f,
            .outOfBoundsCount = 0.f,
            .inboundClock = 0.0f,
            .isOneOnOne = gameState.isOneOnOne // Preserve the 1v1 mode setting
        };
    }

    // --- Part 2: Reset All Agents ---
    Vector3 team_colors[2] = {Vector3{0, 100, 255}, Vector3{255, 0, 100}};
    Entity basketball_entity;
    ctx.iterateQuery(ctx.data().ballQuery, [&](Entity e, Position &pos, Grabbed &grabbed, BallPhysics &phys, Reset &reset, Done &done, CurStep &step)
    {
        basketball_entity = e;
    });
    uint32_t offensive_agent_id = ENTITY_ID_PLACEHOLDER;

    int agent_i = 0;
    ctx.iterateQuery(ctx.data().agentQuery,
        [&](Entity agent, Team &team, InPossession &in_pos, Position &pos, Orientation &orient, Inbounding &inb, GrabCooldown &cooldown,
            Reset &reset, Action &action, ActionMask &mask, Reward &reward, Done &done, CurStep &step, Stats &stats, Attributes &attrs)
    {
        // Reset all components to their default state
        action = Action{0, 0, 0, 0, 0, 0};
        mask = ActionMask{0, 0, 0, 0};
        reset.resetNow = 0;
        inb = Inbounding{false, true};
        // reward.r = 0.f;
        done.episodeDone = 1.f;
        step.step = 0;
        in_pos = {false, ENTITY_ID_PLACEHOLDER, 2};
        orient = Orientation {Quat::id()};
        cooldown = GrabCooldown{0.f};
        stats = {0.f, 0.f};

        if (gameState.isOneOnOne == 1.f)
        {
            Vector3 base_pos = { grid->startX + (agent_i * 2.f), grid->startY, 0.f };
            float x_dev = sampleUniform(ctx, -START_POS_STDDEV, START_POS_STDDEV);
            float y_dev = sampleUniform(ctx, -START_POS_STDDEV, START_POS_STDDEV);
            pos.position = base_pos + Vector3{x_dev, y_dev, 0.f};
            pos.position.x = clamp(pos.position.x, 0.f, grid->width);
            pos.position.y = clamp(pos.position.y, 0.f, grid->height);

            if (agent_i == 0) 
            {
                offensive_agent_id = agent.id;
                in_pos = {true, (uint32_t)basketball_entity.id, 2};
            } 
            else 
            {
                pos = Position { Vector3{ grid->startX - 1 - (-2*(agent_i % 2)), grid->startY - 2 + agent_i/2, 0.f } };
                in_pos = {false, ENTITY_ID_PLACEHOLDER, 2};
            }
        }
        else
        {
            pos = Position { Vector3{ grid->startX - 1 - (-2*(agent_i % 2)), grid->startY - 2 + agent_i/2, 0.f } };
        }

        attrs = {1.f - agent_i*DEFENDER_SLOWDOWN, 0.f, 0.f, 6.5f, pos.position};

        uint32_t defending_hoop_id = (agent_i % 2 == 0) ? gameState.team0Hoop : gameState.team1Hoop;
        team = Team{agent_i % 2, team_colors[agent_i % 2], defending_hoop_id};

        agent_i++;
    });

    // --- Part 3: Reset All Basketballs ---
    ctx.iterateQuery(ctx.data().ballQuery, [&](Entity, Position &pos, Grabbed &grabbed, BallPhysics &phys, Reset &reset, Done &done, CurStep &step)
    {
        pos = Position { Vector3{grid->startX, grid->startY, 0.f} };
        reset.resetNow = 0;
        done.episodeDone = 1.f;
        step.step = 0;
        phys = BallPhysics {false, Vector3::zero(), ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, 2};
        if (gameState.isOneOnOne == 1.f) 
        {
            grabbed = Grabbed{true, offensive_agent_id};
        } 
        else 
        {
            grabbed = Grabbed{false, ENTITY_ID_PLACEHOLDER};
        }
    });

    // --- Part 4: Reset Hoops ---
    ctx.iterateQuery(ctx.data().hoopQuery, [&](Entity, Position &, ImAHoop &, Reset &reset, Done &done, CurStep &step, ScoringZone &)
    {
        reset.resetNow = 0;
        done.episodeDone = 1.f;
        step.step = 0;
    });

    // Finally, clear the world's master reset flag.
    world_reset.resetNow = 0;
}




inline void tick(Engine &ctx,
                 Reset &reset,
                 Done &done,
                 CurStep &episode_step,
                 GrabCooldown &grab_cooldown)
{
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



inline void clockSystem(Engine &ctx, Reset &reset, IsWorldClock &)
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
        reset.resetNow = 1;
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
    ctx.iterateQuery(ctx.data().agentQuery, [&] (Entity agent_entity, Team &team, InPossession, Position &agent_pos, Orientation, Inbounding, GrabCooldown, Reset &, Action &, ActionMask &, Reward &, Done &, CurStep &, Stats &, Attributes &)
    {
        // Check if agent is within touch distance (0.2 meters)
        float distance = (ball_pos.position - agent_pos.position).length();

        if (distance <= AGENT_SIZE_M)
        {
            ball_physics.lastTouchedByAgentID = (uint32_t)agent_entity.id;
            ball_physics.lastTouchedByTeamID = (uint32_t)team.teamIndex;
        }
    });
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
        if (gameState.isOneOnOne == 1.f)
        {
            ctx.iterateQuery(ctx.data().worldClockQuery, [&](Reset &world_reset, IsWorldClock &lalala) {
                world_reset.resetNow = 1;
            });
        }
        else
        {
            ball_physics.inFlight = false;
            ball_physics.velocity = Vector3::zero();
            gameState.liveBall = 0.f;

            // The team that did NOT last touch the ball gets possession.
            uint32_t new_team_idx = 1 - ball_physics.lastTouchedByTeamID;

            // Find the player who had the ball and reset their position
            ctx.iterateQuery(ctx.data().agentQuery, [&](Entity, Team, InPossession &in_possession, Position &agent_pos, Orientation, Inbounding, GrabCooldown, Reset &, Action &, ActionMask &, Reward &, Done &, CurStep &, Stats &, Attributes &)
            {
                // If this agent was the one who went out of bounds with the ball...
                if (in_possession.hasBall && in_possession.ballEntityID == ball_entity.id)
                {
                    agent_pos.position += findVectorToCenter(ctx, agent_pos);

                    // Take the ball away
                    in_possession.hasBall = false;
                    in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
                }
            });

            // Call the helper to give the ball to the other team.
            Quat inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
            assignInbounder(ctx, ball_entity, ball_pos, new_team_idx, inbound_orientation, true);
        }
    }
}


inline void inboundViolationSystem(Engine &ctx, IsWorldClock &)
{
    GameState &gameState = ctx.singleton<GameState>();

    // This is the conditional check. If this isn't true, the system does nothing.
    if (!(gameState.inboundingInProgress > 0.5f && gameState.inboundClock <= 0.f)) {return;}

    uint32_t current_team_idx = (uint32_t)gameState.teamInPossession;
    uint32_t new_team_idx = 1 - current_team_idx;
    uint32_t ball_to_turnover_id = ENTITY_ID_PLACEHOLDER;

    gameState.liveBall = 0.f;

    ctx.iterateQuery(ctx.data().agentQuery, [&](Entity, Team, InPossession &poss, Position &agent_pos, Orientation, Inbounding &inb, GrabCooldown, Reset &, Action &, ActionMask &, Reward &, Done &, CurStep &, Stats &, Attributes &)
    {
        if (inb.imInbounding) {
            ball_to_turnover_id = poss.ballEntityID;

            inb.imInbounding = false;
            poss.hasBall = false;
            poss.ballEntityID = ENTITY_ID_PLACEHOLDER;

            agent_pos.position += findVectorToCenter(ctx, agent_pos);
        }
    });

    if (ball_to_turnover_id != ENTITY_ID_PLACEHOLDER)
    {
        ctx.iterateQuery(ctx.data().ballQuery, [&](Entity ball_entity, Position &ball_pos, Grabbed &grabbed, BallPhysics, Reset &, Done &, CurStep &)
        {
            if (ball_entity.id == (int32_t)ball_to_turnover_id)
            {
                grabbed = {false, ENTITY_ID_PLACEHOLDER};
                Quat inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
                assignInbounder(ctx, ball_entity, ball_pos, new_team_idx, inbound_orientation, true);
            }
        });
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

inline void rewardSystem(Engine &ctx,
                         Entity agent_entity,
                         Reward &reward)
{
}

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

    // Use iterateQuery to fetch the data and assign it to the variables above.
    // Since we assume 1 ball, this lambda will only run once.
    ctx.iterateQuery(ctx.data().ballQuery,
        [&](Entity e, Position &p, Grabbed &grab, BallPhysics &phys, Reset &, Done &, CurStep &)
    {
        ball_pos = p;
        ball_phys = phys;
        ball_grabbed = grab;
    });

    // --- Hoop Positions ---
    Position hoop_positions[NUM_HOOPS];
    uint32_t hoop_ids[NUM_HOOPS];
    int hoop_i = 0;
    ctx.iterateQuery(ctx.data().hoopQuery,
        [&](Entity e, Position &p, ImAHoop &, Reset &, Done &, CurStep &, ScoringZone &)
    {
        assert(hoop_i < NUM_HOOPS);
        hoop_positions[hoop_i] = p;
        hoop_ids[hoop_i] = e.id;
        hoop_i++;
    });

    // --- All Agent Data ---
    AgentObservationData all_agents[NUM_AGENTS];
    int agent_idx = 0;
    int32_t inbounder_id = -1; // Store the ID of the current inbounder
    ctx.iterateQuery(ctx.data().agentQuery,
        [&](Entity e, Team &t, InPossession &ip, Position &p, Orientation &o,
            Inbounding &ib, GrabCooldown &gc, Reset &, Action &, ActionMask &,
            Reward &, Done &, CurStep &, Stats &, Attributes &)
    {
        assert(agent_idx < NUM_AGENTS);
        all_agents[agent_idx++] = { e.id, t.teamIndex, p, o, ip, ib, gc };
        if (ib.imInbounding) {
            inbounder_id = e.id;
        }
    });

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
    // FIX: Access velocity components correctly from the BallPhysics struct
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
    auto actionMaskingNode = builder.addToGraph<ParallelForNode<Engine, actionMaskSystem,
        ActionMask, GrabCooldown, InPossession, Inbounding>>(deps);

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

    auto resetRewardSystemNode = builder.addToGraph<ParallelForNode<Engine, resetRewardsSystem,
        Reward>>({moveBallSystemNode});

    auto scoreSystemNode = builder.addToGraph<ParallelForNode<Engine, scoreSystem,
        Entity, Position, ScoringZone>>({resetRewardSystemNode});

    auto outOfBoundsSystemNode = builder.addToGraph<ParallelForNode<Engine, outOfBoundsSystem,
        Entity, Position, BallPhysics>>({scoreSystemNode});

    auto updateLastTouchSystemNode = builder.addToGraph<ParallelForNode<Engine, updateLastTouchSystem,
        Position, BallPhysics>>({outOfBoundsSystemNode});

    auto tickNode = builder.addToGraph<ParallelForNode<Engine, tick,
        Reset, Done, CurStep, GrabCooldown>>({updateLastTouchSystemNode});

    auto clockSystemNode = builder.addToGraph<ParallelForNode<Engine, clockSystem,
        Reset, IsWorldClock>>({tickNode});

    // Add the new inbound violation system to the graph
    auto inboundViolationSystemNode = builder.addToGraph<ParallelForNode<Engine, inboundViolationSystem,
        IsWorldClock>>({clockSystemNode});

    auto resetSystemNode = builder.addToGraph<ParallelForNode<Engine, resetSystem,
        Reset, IsWorldClock>>({inboundViolationSystemNode});

    auto updatePointsWorthNode = builder.addToGraph<ParallelForNode<Engine, updatePointsWorthSystem,
        Position, InPossession, Team>>({resetSystemNode});

    auto agentCollisionNode = builder.addToGraph<ParallelForNode<Engine, agentCollisionSystem,
        Entity, Position, InPossession>>({updatePointsWorthNode});

    auto hardCodeDefenseSystemNode = builder.addToGraph<ParallelForNode<Engine, hardCodeDefenseSystem,
        Team, Position, Action, Attributes>>({agentCollisionNode});

    auto fillObservationsNode = builder.addToGraph<ParallelForNode<Engine, fillObservationsSystem,
        Entity, Observations, Position, Orientation, InPossession,
        Inbounding, Team, GrabCooldown>>({hardCodeDefenseSystemNode});

    return fillObservationsNode;
}

}
