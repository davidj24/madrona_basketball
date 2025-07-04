#include "sim.hpp"
#include "types.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath> // For acosf


using namespace madrona;
using namespace madrona::math;

constexpr float SIMULATION_HZ = 62.0f; // How many timesteps are in one second
constexpr float TIMESTEPS_TO_SECONDS_FACTOR = 1.0f / SIMULATION_HZ;
constexpr float HOOP_SCORE_ZONE_SIZE = .1f;
constexpr float TIME_PER_PERIOD = 300.f;

constexpr float COURT_LENGTH_M = 28.65f;
constexpr float COURT_WIDTH_M = 15.24f;
constexpr float WORLD_WIDTH_M = 28.65f * 1.1f;
constexpr float WORLD_HEIGHT_M = 15.24f * 1.1f;
constexpr float COURT_MIN_X = (WORLD_WIDTH_M - COURT_LENGTH_M) / 2.0f;
constexpr float COURT_MAX_X = COURT_MIN_X + COURT_LENGTH_M;
constexpr float COURT_MIN_Y = (WORLD_HEIGHT_M - COURT_WIDTH_M) / 2.0f;
constexpr float COURT_MAX_Y = COURT_MIN_Y + COURT_WIDTH_M;

// This is the small buffer to ensure the player is placed *inside* the line
constexpr float IN_COURT_OFFSET = 0.1f; 

namespace madsimple {
    // =================================================== Helper Functions ===================================================
    
    // Computes the rotation needed to align the 'start' vector with the 'target' vector.
    inline Quat findRotationBetweenVectors(Vector3 start, Vector3 target) 
    {
        // Ensure the vectors are normalized (unit length)
        start = start.normalize();
        target = target.normalize();

        float dot_product = dot(start, target);

        // Case 1: If the vectors are already aligned, no rotation is needed.
        if (dot_product > 0.999999f) {
            return Quat::id();
        }

        // Case 2: If the vectors are in opposite directions, we need a 180-degree rotation.
        // For a 2D game, the most stable axis for a 180-degree turn is the Z-axis.
        if (dot_product < -0.999999f) {
            return Quat::angleAxis(pi, Vector3{0.f, 0.f, 1.f});
        }

        // Case 3: The general case.
        // The axis of rotation is the cross product of the two vectors.
        Vector3 rotation_axis = cross(start, target);
        rotation_axis = rotation_axis.normalize();

        // The angle is the arccosine of the dot product.
        float rotation_angle = acosf(dot_product);

        return Quat::angleAxis(rotation_angle, rotation_axis);
    }

    inline int32_t getShotPointValue(Position shot_pos, Position hoop_pos, float distance_to_hoop) 
    {
        const float COURT_LENGTH_M = 28.65f;
        const float COURT_WIDTH_M = 15.24f;
        const float WORLD_WIDTH_M = 31.515f;  // 28.65 * 1.1
        const float WORLD_HEIGHT_M = 16.764f; // 15.24 * 1.1

        const float ARC_RADIUS_M = 7.24f;
        const float CORNER_3_FROM_SIDELINE_M = 0.91f;
        const float CORNER_3_LENGTH_FROM_BASELINE_M = 4.27f;

        // --- Calculate Court's Position within the World (The crucial fix) ---
        const float court_min_x = (WORLD_WIDTH_M - COURT_LENGTH_M) / 2.0f;
        const float court_min_y = (WORLD_HEIGHT_M - COURT_WIDTH_M) / 2.0f;

        // --- Logic ---

        // 1. Check if the shot is in the corner lane, relative to the court's position.
        bool isInCornerLane = (shot_pos.position.y < court_min_y + CORNER_3_FROM_SIDELINE_M || 
                            shot_pos.position.y > court_min_y + COURT_WIDTH_M - CORNER_3_FROM_SIDELINE_M);

        if (isInCornerLane) {
            // 2. If so, check if the shot is within the corner's length, relative to the court's position.
            bool isShootingAtLeftHoop = hoop_pos.position.x < WORLD_WIDTH_M / 2.0f;
            
            if (isShootingAtLeftHoop) {
                if (shot_pos.position.x <= court_min_x + CORNER_3_LENGTH_FROM_BASELINE_M) {
                    return 3;
                }
            } else { // Shooting at the right hoop
                if (shot_pos.position.x >= court_min_x + COURT_LENGTH_M - CORNER_3_LENGTH_FROM_BASELINE_M) {
                    return 3;
                }
            }
        }

        // 3. If not a valid corner 3, check the distance against the arc.
        if (distance_to_hoop > ARC_RADIUS_M) {
            return 3;
        }

        // 4. If none of the 3-point conditions are met, it is a 2-point shot.
        return 2;
    }

    inline void assignInbounder(Engine &ctx, Entity ball_entity, uint32_t new_team_idx, bool is_turnover)
    {
        GameState &gameState = ctx.singleton<GameState>();
        bool inbounder_assigned = false;

        // Find the first available player on the new team.
        auto agent_query = ctx.query<Entity, Team, InPossession, Position, Inbounding>();
        Position ball_pos = ctx.get<Position>(ball_entity);

        ctx.iterateQuery(agent_query, [&](Entity agent_entity, Team &agent_team, InPossession &in_possession, Position &agent_pos, Inbounding &inbounding)
        {
            // FIX: Ensure you're comparing compatible types (uint32_t and int32_t)
            if ((uint32_t)agent_team.teamIndex == new_team_idx && !inbounder_assigned)
            {
                inbounder_assigned = true;
                inbounding.imInbounding = true;
                agent_pos = ball_pos; // Move player to the ball
                
                // Give them possession of the ball
                ctx.get<Grabbed>(ball_entity) = {true, (uint32_t)agent_entity.id};
                in_possession.hasBall = true;
                // FIX: Assign the ENTITY'S ID, not the entity object itself.
                in_possession.ballEntityID = ball_entity.id;
            }
        });

        // If we successfully found a player, update the game state.
        if(inbounder_assigned) {
            gameState.teamInPossession = (float)new_team_idx;
            gameState.liveBall = 0.f; // Ball is dead during an inbound
            gameState.inboundingInProgress = 1.0f;
            gameState.inboundClock = 5.f; // Reset the 5-second clock

            // Only increment the out-of-bounds count if it wasn't a 5-second turnover
            if (!is_turnover) {
                gameState.outOfBoundsCount++;
            }
        }
    }

    inline Vector3 clampToCourt(Vector3 pos)
    {
        float clamped_x = std::clamp(pos.x, COURT_MIN_X + IN_COURT_OFFSET, COURT_MAX_X - IN_COURT_OFFSET);
        float clamped_y = std::clamp(pos.y, COURT_MIN_Y + IN_COURT_OFFSET, COURT_MAX_Y - IN_COURT_OFFSET);
        
        // Return the new, valid position, keeping the original z-height.
        return Vector3{clamped_x, clamped_y, pos.z};
    }
    // =================================================== Registry ===================================================
    void Sim::registerTypes(ECSRegistry &registry, const Config &)
    {
        base::registerTypes(registry);

        // ================================================== Singletons ==================================================
        registry.registerSingleton<GameState>();




        // ================================================== General Components ==================================================
        registry.registerComponent<Reset>();
        registry.registerComponent<Position>();
        registry.registerComponent<Done>();
        registry.registerComponent<CurStep>();
        registry.registerComponent<RandomMovement>();
        registry.registerComponent<IsWorldClock>();


        // ================================================== Agent Components ==================================================
        registry.registerComponent<Action>();
        registry.registerComponent<Reward>();
        registry.registerComponent<Inbounding>();
        registry.registerComponent<InPossession>();
        registry.registerComponent<Orientation>();
        registry.registerComponent<Team>();
        registry.registerComponent<GrabCooldown>();


        // ================================================== Ball Components ==================================================
        registry.registerComponent<BallPhysics>();
        registry.registerComponent<Grabbed>();


        // ================================================== Hoop Components ==================================================
        registry.registerComponent<ImAHoop>();
        registry.registerComponent<ScoringZone>();


        // ================================================= Archetypes ================================================= 
        registry.registerArchetype<Agent>();
        registry.registerArchetype<Basketball>();
        registry.registerArchetype<Hoop>();
        registry.registerArchetype<WorldClock>();



        // ================================================= Tensor Exports For Viewer =================================================
        registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
        registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
        registry.exportColumn<Agent, Position>((uint32_t)ExportID::AgentPos);
        registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
        registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
        registry.exportColumn<Agent, InPossession>((uint32_t)ExportID::AgentPossession);
        registry.exportColumn<Agent, Team>((uint32_t)ExportID::TeamData);
        registry.exportColumn<Agent, Orientation>((uint32_t)ExportID::Orientation);

        registry.exportColumn<Basketball, Position>((uint32_t)ExportID::BasketballPos);
        registry.exportColumn<Basketball, BallPhysics>((uint32_t)ExportID::BallPhysicsData);
        registry.exportColumn<Basketball, Grabbed>((uint32_t)ExportID::BallGrabbed);

        registry.exportColumn<Hoop, Position>((uint32_t)ExportID::HoopPos);

        // Singleton exports
        registry.exportSingleton<GameState>((uint32_t)ExportID::GameState);
        
        // Export entity IDs for debugging
        registry.exportColumn<Agent, madrona::Entity>((uint32_t)ExportID::AgentEntityID);
        registry.exportColumn<Basketball, madrona::Entity>((uint32_t)ExportID::BallEntityID);
        
    }




    //=================================================== Ball Systems ===================================================
    inline void moveBallRandomly(Engine &ctx,
                        Position &ball_pos,
                        RandomMovement &random_movement)
    {
        random_movement.moveTimer ++;
        if (random_movement.moveTimer >= random_movement.moveInterval) 
        {
            random_movement.moveTimer = 0.f;
            const GridState *grid = ctx.data().grid;

            // Random movement in continuous space (0.1m steps)
            float dx = ((rand() % 3) - 1) * 0.1f; // -0.1, 0, or 0.1 meters
            float dy = ((rand() % 3) - 1) * 0.1f; // -0.1, 0, or 0.1 meters

            float new_x = ball_pos.position.x + dx;
            float new_y = ball_pos.position.y + dy;

            new_x = std::clamp(new_x, 0.f, grid->width);
            new_y = std::clamp(new_y, 0.f, grid->height);

            ball_pos.position.x = new_x;
            ball_pos.position.y = new_y;
        } 
    }



    inline void moveBallSystem(Engine &ctx,
                            Position &ball_pos,
                            BallPhysics &ball_physics,
                            Grabbed &grabbed)
    {
        auto holder_query = ctx.query<Entity, Position, InPossession>();
        ctx.iterateQuery(holder_query, [&](Entity &agent_entity, Position &agent_pos, InPossession &in_possession)
        {
            // Make the ball move with the agent if it's held
            bool agent_is_holding_this_ball = (in_possession.hasBall == true &&
                                                grabbed.isGrabbed &&
                                                grabbed.holderEntityID == (uint32_t)agent_entity.id);
            if (agent_is_holding_this_ball)
            {
                ball_pos = agent_pos;  // Move basketball to agent's new position
                return;
            }
        });    

        if (ball_physics.velocity.length() == 0 || grabbed.isGrabbed) {return;}

        const GridState* grid = ctx.data().grid; // To clamp later
        float new_x = ball_pos.position.x + ball_physics.velocity[0];
        float new_y = ball_pos.position.y + ball_physics.velocity[1];
        float new_z = ball_pos.position.z + ball_physics.velocity[2];

        new_x = std::clamp(new_x, 0.f, grid->width);
        new_y = std::clamp(new_y, 0.f, grid->height);
        // new_z = std::clamp(new_z, 0.f, grid->depth);
        
        // Convert to discrete grid for wall collision checking
        int32_t discrete_x = (int32_t)(new_x * grid->cellsPerMeter);
        int32_t discrete_y = (int32_t)(new_y * grid->cellsPerMeter);
        discrete_x = std::clamp(discrete_x, 0, grid->discreteWidth - 1);
        discrete_y = std::clamp(discrete_y, 0, grid->discreteHeight - 1);
        
        const Cell &new_cell = grid->cells[discrete_y * grid->discreteWidth + discrete_x];
        
        if (!(new_cell.flags & CellFlag::Wall)) {
            ball_pos.position.x = new_x;
            ball_pos.position.y = new_y;
            ball_pos.position.z = new_z;
        }
    }



    //=================================================== Agent Systems ===================================================
    inline void grabSystem(Engine &ctx,
                            Entity agent_entity,
                            Action &action,
                            Position &agent_pos,
                            InPossession &in_possession,
                            Team &team,
                            GrabCooldown &grab_cooldown)
    {
        GameState &gameState = ctx.singleton<GameState>();
        auto basketball_query = ctx.query<Entity, Position, Grabbed, BallPhysics>();
        if (action.grab == 0) {return;}
        if (grab_cooldown.cooldown > 0.f) {return;}
        grab_cooldown.cooldown = 10.f;

        ctx.iterateQuery(basketball_query, [&](Entity ball_entity, Position &basketball_pos, Grabbed &grabbed, BallPhysics &ball_physics) 
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
            
            if (distance_between_ball_and_player <= 0.5f)
            {
                auto agent_query = ctx.query<InPossession>();
                ctx.iterateQuery(agent_query, [&] (InPossession &other_in_possession)
                {
                    if (other_in_possession.ballEntityID == (uint32_t)ball_entity.id) // if we're stealing from another agent
                    {
                        other_in_possession.hasBall = false;
                        other_in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
                    }
                });

                in_possession.hasBall = true;
                in_possession.ballEntityID = ball_entity.id;
                grabbed.holderEntityID = (uint32_t)agent_entity.id;
                grabbed.isGrabbed = true;
                ball_physics.inFlight = false; // Make it so the ball isn't "in flight" anymore
                ball_physics.velocity = Vector3::zero(); // And change its velocity to be zero
                gameState.teamInPossession = (float)team.teamIndex; // Update the team in possession
                gameState.liveBall = 1.f;
            }
        });

    }



    inline void passSystem(Engine &ctx,
                        Entity agent_entity,
                        Action &action,
                        Orientation &agent_orientation,
                        InPossession &in_possession,
                        Inbounding &inbounding)
    {

        if (action.pass == 0 || !in_possession.hasBall) {return;}
        GameState &gameState = ctx.singleton<GameState>();




        auto held_ball_query = ctx.query<Grabbed, BallPhysics>();
        ctx.iterateQuery(held_ball_query, [&] (Grabbed &grabbed, BallPhysics &ball_physics)
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
                            Position agent_pos,
                            Orientation &agent_orientation,
                            Inbounding &inbounding,
                            InPossession &in_possession,
                            Team &team)
    {
        if (action.shoot == 0 || !in_possession.hasBall) {return;}

        // Find the attacking hoop (not defendingHoopID)
        auto hoop_query = ctx.query<Entity, Position, ScoringZone>();
        Position attacking_hoop_pos = {0.f, 0.f, 0.f};
        ctx.iterateQuery(hoop_query, [&](Entity hoop_entity, Position &hoop_pos, ScoringZone &scoring_zone) 
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
        // Create a single random number generator for all deviations
        static thread_local std::mt19937 rng(std::random_device{}());




        // ======================== DEVIATION TUNERS ==============================
        float dist_deviation_per_meter = 0.0f;
        float def_deviation_per_meter = .0f; 
        float vel_deviation_factor = 1.f;


        // 1. Mess up angle based on distance
        float distance_to_hoop = shot_vector.length();
        float dist_stddev = dist_deviation_per_meter/100 * distance_to_hoop;
        std::normal_distribution<float> dist_dist(0.0f, dist_stddev);
        float deviation_from_distance = dist_dist(rng);


        // 2. Mess up angle based on contest level (how close nearest defender is)
        float deviation_from_defender = 0.0f;
        float distance_to_nearest_defender = std::numeric_limits<float>::infinity();
        auto nearest_defender_query = ctx.query<Position, Team>();
        ctx.iterateQuery(nearest_defender_query, [&](Position &defender_pos, Team &defender_team) 
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
            std::normal_distribution<float> def_dist(0.0f, def_stddev);
            deviation_from_defender = def_dist(rng);
        }


        // 3. Mess up angle based on agent velocity
        float deviation_from_velocity = 0.0f;
        if (action.moveSpeed > 0) {
            float vel_stddev = vel_deviation_factor/10 * action.moveSpeed;
            std::normal_distribution<float> vel_dist(0.0f, vel_stddev);
            deviation_from_velocity = vel_dist(rng);
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
        auto held_ball_query = ctx.query<Grabbed, BallPhysics>();
        ctx.iterateQuery(held_ball_query, [&] (Grabbed &grabbed, BallPhysics &ball_physics)
        {
            if (grabbed.holderEntityID == agent_entity.id)
            {
                grabbed.isGrabbed = false;
                grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER;
                in_possession.hasBall = false;
                in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
                inbounding.imInbounding = false;
                ball_physics.pointsWorth = getShotPointValue(agent_pos, attacking_hoop_pos, distance_to_hoop);
                ball_physics.velocity = final_shot_vec * .1f;
                ball_physics.inFlight = true;
            }
        });
    }


    inline void moveAgentSystem(Engine &ctx,
                            Action &action,
                            Position &agent_pos, // Note: This should now store floats
                            InPossession &in_possession,
                            Orientation &agent_orientation)
    {
        const GridState *grid = ctx.data().grid;
        if (action.rotate != 0)
        {
            // Rotation logic is fine as it is
            float turn_angle = (pi/180.f) * action.rotate * 6;
            Quat turn = Quat::angleAxis(turn_angle, Vector3{0, 0, 1});
            agent_orientation.orientation = turn * agent_orientation.orientation;
        }

        if (action.moveSpeed > 0)
        {
            // Treat moveSpeed as a velocity in meters/second, not a distance.
            // Let's say a moveSpeed of 1 corresponds to 1 m/s.
            float agent_velocity_magnitude = action.moveSpeed * 4;
            if (in_possession.hasBall ==1) {agent_velocity_magnitude *= .8;}

            constexpr float angle_between_directions = pi / 4.f;
            float move_angle = action.moveAngle * angle_between_directions;

            // Calculate velocity vector components
            float vel_x = std::sin(move_angle);
            float vel_y = -std::cos(move_angle); // Your forward is -Y

            // Calculate distance to move this frame
            float dx = vel_x * agent_velocity_magnitude * TIMESTEPS_TO_SECONDS_FACTOR;
            float dy = vel_y * agent_velocity_magnitude * TIMESTEPS_TO_SECONDS_FACTOR;

            // Update position (now using floats)
            float new_x = agent_pos.position.x + dx;
            float new_y = agent_pos.position.y + dy;

            // Boundary checking in continuous space
            new_x = std::clamp(new_x, 0.f, grid->width);
            new_y = std::clamp(new_y, 0.f, grid->height);

            // Convert to discrete grid for wall collision checking
            int32_t discrete_x = (int32_t)(new_x * grid->cellsPerMeter);
            int32_t discrete_y = (int32_t)(new_y * grid->cellsPerMeter);
            discrete_x = std::clamp(discrete_x, 0, grid->discreteWidth - 1);
            discrete_y = std::clamp(discrete_y, 0, grid->discreteHeight - 1);
            
            const Cell &new_cell = grid->cells[discrete_y * grid->discreteWidth + discrete_x];
            
            if (!(new_cell.flags & CellFlag::Wall)) {
                agent_pos.position.x = new_x;
                agent_pos.position.y = new_y;
            }
        }
    }


    inline void actionMaskSystem(Engine &ctx,
                                 ActionMask &action_mask,
                                 InPossession &in_possession,
                                 Team &team,
                                 Inbounding &inbounding)
    {
        GameState &gameState = ctx.singleton<GameState>();
        if (some_condition)
        {

        }
        else
        {
            action_mask.can_move = 1.f;
            // Offensive actions
            if (gameState.teamInPossession == team.teamIndex)
            {
                
            }
            else // Defensive actions
            {

            }
        }
    }


    //=================================================== Hoop Systems ===================================================
    inline void scoreSystem(Engine &ctx,
                            Entity hoop_entity,
                            Position &hoop_pos,
                            ScoringZone &scoring_zone)
    {
        GameState &gameState = ctx.singleton<GameState>();
        
        auto ball_query = ctx.query<Position, BallPhysics>();
        ctx.iterateQuery(ball_query, [&] (Position &ball_pos, BallPhysics &ball_physics)
        {
            float distance_to_hoop = std::sqrt((ball_pos.position.x - hoop_pos.position.x) * (ball_pos.position.x - hoop_pos.position.x) + 
                                               (ball_pos.position.y - hoop_pos.position.y) * (ball_pos.position.y - hoop_pos.position.y));
            if (distance_to_hoop <= scoring_zone.radius && ball_physics.inFlight && gameState.liveBall == 1.f) 
            {
                // Ball is within scoring zone, score a point
                if ((float)hoop_entity.id == gameState.team0Hoop) {gameState.team1Score += ball_physics.pointsWorth;}
                else{gameState.team0Score += ball_physics.pointsWorth;}
                gameState.scoredBaskets++;

                // Reset the ball position and state
                ball_physics.inFlight = false;
                ball_pos = hoop_pos;
                ball_physics.velocity = Vector3::zero();
            }
        });
    }


    //=================================================== General Systems ===================================================
    inline void resetSystem(Engine &ctx, Reset &world_reset, IsWorldClock &)
    {
        // This system only runs if the world clock's reset is triggered.
        if (world_reset.resetNow == 0) {
            return;
        }

        GameState &gameState = ctx.singleton<GameState>();
        const GridState *grid = ctx.data().grid;
        
        // Check if the reset was triggered by the end of a period
        if (gameState.gameClock <= 0.f) {
            // Check if the game should continue
            if (gameState.period < 4 || gameState.team0Score == gameState.team1Score) {
                gameState.period++;
                gameState.gameClock = TIME_PER_PERIOD;
                gameState.shotClock = 24.0f;
                gameState.liveBall = 1.f; // Start the next period
            } else {
                // The game is over, freeze the clock and ball
                gameState.gameClock = 0.f;
                gameState.shotClock = 0.f;
                gameState.liveBall = 0.f;
            }
        } else { // This was a manual reset (e.g., from Python)
            // Fully reset the game state to the beginning
            gameState = GameState{
                .inboundingInProgress = 0.0f,
                .liveBall = 1.0f,
                .period = 1.0f,
                .teamInPossession = 0.0f,
                .team0Hoop = 0.0f,
                .team0Score = 0.0f,
                .team1Hoop = 1.0f,
                .team1Score = 0.0f,
                .gameClock = TIME_PER_PERIOD,
                .shotClock = 24.0f,
                .scoredBaskets = 0.f,
                .outOfBoundsCount = 0.f,
                .inboundClock = 0.f
            };
        }

        // Reset all agents
        auto agent_query = ctx.query<Action, Position, Reset, Inbounding, Done, CurStep, InPossession, Orientation, GrabCooldown>();
        float agent_start_x[4] = {grid->startX - 2.0f, grid->startX - 1.0f, grid->startX + 0.0f, grid->startX + 1.0f};
        int agent_i = 0;
        ctx.iterateQuery(agent_query, [&](Action &action, Position &pos, Reset &reset, Inbounding &inbounding, Done &done, CurStep &curstep, InPossession &inpos, Orientation &orient, GrabCooldown &cooldown) 
        {
            action = Action{0, 0, 0, 0, 0, 0, 0, 0};
            float x = (agent_i < 4) ? agent_start_x[agent_i] : grid->startX;
            pos = Position{Vector3{x, grid->startY, 0.f}};
            reset.resetNow = 0; // Clear the flag
            inbounding = Inbounding{false, true};
            done.episodeDone = 1.f; // Signal to python that a reset happened
            curstep.step = 0;
            inpos = {false, ENTITY_ID_PLACEHOLDER};
            orient = Orientation{Quat::id()};
            cooldown = GrabCooldown{0.f};
            agent_i++;
        });

        // Reset all basketballs
        auto basketball_query = ctx.query<Position, Reset, Done, CurStep, Grabbed, BallPhysics>();
        ctx.iterateQuery(basketball_query, [&](Position &pos, Reset &reset, Done &done, CurStep &curstep, Grabbed &grabbed, BallPhysics &ballphys) {
            pos = Position{Vector3{grid->startX, grid->startY, 0.f}};
            reset.resetNow = 0;
            done.episodeDone = 1.f;
            curstep.step = 0;
            grabbed = Grabbed{false, ENTITY_ID_PLACEHOLDER};
            ballphys = BallPhysics{false, Vector3::zero(), ENTITY_ID_PLACEHOLDER, 2};
        });

        // Reset all hoops 
        auto hoop_query = ctx.query<Position, Reset, Done, CurStep, ImAHoop, ScoringZone>();
        int hoop_i = 0;
        ctx.iterateQuery(hoop_query, [&](Position &pos, Reset &reset, Done &done, CurStep &curstep, ImAHoop &, ScoringZone &zone) {
            // This logic can be more sophisticated based on court dimensions
            if (hoop_i == 0)
                pos = Position{Vector3{3.0f, grid->height / 2.0f, 0.f}};
            else if (hoop_i == 1)
                pos = Position{Vector3{grid->width - 3.0f, grid->height / 2.0f, 0.f}};
            
            reset.resetNow = 0;
            done.episodeDone = 1.f;
            curstep.step = 0;
            zone = ScoringZone{HOOP_SCORE_ZONE_SIZE, 2.0f, Vector3{pos.position.x, pos.position.y, pos.position.z}};
            hoop_i++;
        });

        // Finally, clear the world's reset flag
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
        grab_cooldown.cooldown = std::max(0.f, grab_cooldown.cooldown - 1.f);
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
        auto touched_agent_query = ctx.query<Position, Team>();
        ctx.iterateQuery(touched_agent_query, [&] (Position &agent_pos, Team &team)
        {
            // Check if agent is within touch distance (0.5 meters)
            float distance = sqrt((ball_pos.position.x - agent_pos.position.x) * (ball_pos.position.x - agent_pos.position.x) +
                                (ball_pos.position.y - agent_pos.position.y) * (ball_pos.position.y - agent_pos.position.y) +
                                (ball_pos.position.z - agent_pos.position.z) * (ball_pos.position.z - agent_pos.position.z));
            
            if (distance <= 0.5f) 
            {
                ball_physics.lastTouchedByID = (uint32_t)team.teamIndex;
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
        bool is_out_of_bounds = ball_pos.position.x < COURT_MIN_X || ball_pos.position.x > COURT_MAX_X ||
                                ball_pos.position.y < COURT_MIN_Y || ball_pos.position.y > COURT_MAX_Y;

        if (is_out_of_bounds && gameState.inboundingInProgress == 0.f)
        {
            ball_physics.inFlight = false;
            ball_physics.velocity = Vector3::zero();

            // The team that did NOT last touch the ball gets possession.
            uint32_t new_team_idx = 1 - ball_physics.lastTouchedByID;

            // Find the player who had the ball and reset their position
            auto agent_query = ctx.query<InPossession, Position>();
            ctx.iterateQuery(agent_query, [&](InPossession &in_possession, Position &agent_pos)
            {
                // If this agent was the one who went out of bounds with the ball...
                if (in_possession.hasBall && in_possession.ballEntityID == ball_entity.id)
                {
                    // FIX: Instead of teleporting to center, clamp them to the nearest in-bounds spot.
                    agent_pos.position = clampToCourt(agent_pos.position);
                    
                    // Take the ball away
                    in_possession.hasBall = false;
                    in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
                }
            });
            
            // Call the helper to give the ball to the other team.
            assignInbounder(ctx, ball_entity, new_team_idx, false);
        }
    }


    inline void inboundViolationSystem(Engine &ctx, IsWorldClock &)
    {
        GameState &gameState = ctx.singleton<GameState>();

        // This is the conditional check. If this isn't true, the system does nothing.
        if (!(gameState.inboundingInProgress > 0.5f && gameState.inboundClock <= 0.f)) {
            return;
        }

        uint32_t current_team_idx = (uint32_t)gameState.teamInPossession;
        uint32_t new_team_idx = 1 - current_team_idx;

        uint32_t ball_to_turnover_id = ENTITY_ID_PLACEHOLDER;
        
        auto inbounder_query = ctx.query<Inbounding, InPossession, Position>();
        ctx.iterateQuery(inbounder_query, [&](Inbounding &inb, InPossession &poss, Position &agent_pos) {
            if (inb.imInbounding) {
                ball_to_turnover_id = poss.ballEntityID;
                
                inb.imInbounding = false;
                poss.hasBall = false;
                poss.ballEntityID = ENTITY_ID_PLACEHOLDER;

                agent_pos.position = clampToCourt(agent_pos.position);
            }
        });

        if (ball_to_turnover_id != ENTITY_ID_PLACEHOLDER) {
            auto ball_query = ctx.query<Entity, Grabbed>();
            ctx.iterateQuery(ball_query, [&](Entity ball_e, Grabbed &grabbed) {
                if (ball_e.id == (int32_t)ball_to_turnover_id) {
                    grabbed = {false, ENTITY_ID_PLACEHOLDER};
                    assignInbounder(ctx, ball_e, new_team_idx, true);
                }
            });
        }
    }

    // =================================================== Task Graph ===================================================
    void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                    const Config &)
    {
        TaskGraphBuilder &builder = taskgraph_mgr.init(0);
        
        auto moveAgentSystemNode = builder.addToGraph<ParallelForNode<Engine, moveAgentSystem,
            Action, Position, InPossession, Orientation>>({});

        auto grabSystemNode = builder.addToGraph<ParallelForNode<Engine, grabSystem,
            Entity, Action, Position, InPossession, Team, GrabCooldown>>({});

        auto passSystemNode = builder.addToGraph<ParallelForNode<Engine, passSystem,
            Entity, Action, Orientation, InPossession, Inbounding>>({});
        
        auto shootSystemNode = builder.addToGraph<ParallelForNode<Engine, shootSystem,
            Entity, Action, Position, Orientation, Inbounding, InPossession, Team>>({});

        auto moveBallSystemNode = builder.addToGraph<ParallelForNode<Engine, moveBallSystem,
            Position, BallPhysics, Grabbed>>({grabSystemNode, passSystemNode, shootSystemNode});

        auto scoreSystemNode = builder.addToGraph<ParallelForNode<Engine, scoreSystem,
            Entity, Position, ScoringZone>>({moveBallSystemNode});

        auto outOfBoundsSystemNode = builder.addToGraph<ParallelForNode<Engine, outOfBoundsSystem,
            Entity, Position, BallPhysics>>({moveBallSystemNode});
        
        auto updateLastTouchSystemNode = builder.addToGraph<ParallelForNode<Engine, updateLastTouchSystem,
            Position, BallPhysics>>({moveBallSystemNode});

        auto tickNode = builder.addToGraph<ParallelForNode<Engine, tick,
            Reset, Done, CurStep, GrabCooldown>>({});
        
        auto clockSystemNode = builder.addToGraph<ParallelForNode<Engine, clockSystem,
            Reset, IsWorldClock>>({});

        // Add the new inbound violation system to the graph
        auto inboundViolationSystemNode = builder.addToGraph<ParallelForNode<Engine, inboundViolationSystem,
            IsWorldClock>>({clockSystemNode});

        auto resetSystemNode = builder.addToGraph<ParallelForNode<Engine, resetSystem,
            Reset, IsWorldClock>>({clockSystemNode, tickNode});
    }

    // =================================================== Sim Creation ===================================================

    Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
    episodeMgr(init.episodeMgr),
    grid(init.grid),
    maxEpisodeLength(cfg.maxEpisodeLength)
    {
        ctx.singleton<GameState>() = GameState 
        {
            .inboundingInProgress = 0.0f,
            .liveBall = 1.0f,
            .period = 1.0f,
            .teamInPossession = 0.0f,
            .team0Hoop = 0.0f,
            .team0Score = 0.0f,
            .team1Hoop = 1.0f,
            .team1Score = 0.0f,
            .gameClock = TIME_PER_PERIOD,
            .shotClock = 24.0f,
            .scoredBaskets = 0.f,
            .outOfBoundsCount = 0.f
        };

        // Make sure to add the Reset component to the WorldClock entity
        Entity worldClock = ctx.makeEntity<WorldClock>();
        ctx.get<IsWorldClock>(worldClock) = {};
        ctx.get<Reset>(worldClock) = {0}; // Initialize resetNow to 0


        std::vector<Vector3> team_colors = {Vector3{0, 100, 255}, Vector3{255, 0, 100}};
        for (int i = 0; i < NUM_AGENTS; i++) 
        {
            Entity agent = ctx.makeEntity<Agent>();
            ctx.get<Action>(agent) = Action{0, 0, 0, 0, 0, 0, 0, 0};
            ctx.get<Position>(agent) = Position 
            {
                Vector3{
                    grid->startX + (i - 2) * 1.0f,
                    grid->startY,
                    0.f
                }
            };
            ctx.get<Reset>(agent) = Reset{0};
            ctx.get<Inbounding>(agent) = Inbounding{false, true};
            ctx.get<Reward>(agent).r = 0.f;
            ctx.get<Done>(agent).episodeDone = 0.f;
            ctx.get<CurStep>(agent).step = 0;
            ctx.get<InPossession>(agent) = {false, ENTITY_ID_PLACEHOLDER};
            ctx.get<Orientation>(agent) = Orientation {Quat::id()};
            ctx.get<GrabCooldown>(agent) = GrabCooldown{0.f};
            
            uint32_t defending_hoop_id = (i % 2 == 0) ? 1 : 0;
            ctx.get<Team>(agent) = Team{i % 2, team_colors[i % 2], defending_hoop_id};
        };

        for (int i = 0; i < NUM_BASKETBALLS; i++) 
        {
            Entity basketball = ctx.makeEntity<Basketball>();
            ctx.get<Position>(basketball) = Position { Vector3{grid->startX, grid->startY, 0.f} };
            ctx.get<Reset>(basketball) = Reset{0};
            ctx.get<Done>(basketball).episodeDone = 0.f;
            ctx.get<CurStep>(basketball).step = 0;
            ctx.get<Grabbed>(basketball) = Grabbed {false, ENTITY_ID_PLACEHOLDER};
            ctx.get<BallPhysics>(basketball) = BallPhysics {false, Vector3::zero(), ENTITY_ID_PLACEHOLDER, 2};
        }

        GameState &gameState = ctx.singleton<GameState>();
        for (int i = 0; i < NUM_HOOPS; i++) 
        {
            Entity hoop = ctx.makeEntity<Hoop>();
            Position hoop_pos;
            
            const float NBA_COURT_WIDTH = 28.65f;
            const float NBA_COURT_HEIGHT = 15.24f;
            const float HOOP_OFFSET_FROM_BASELINE = 1.575f;
            
            float court_start_x = (grid->width - NBA_COURT_WIDTH) / 2.0f;
            float court_center_y = grid->height / 2.0f;
            
            if (i == 0) 
            {
                gameState.team0Hoop = hoop.id;
                hoop_pos = Position { 
                    Vector3{
                        court_start_x + HOOP_OFFSET_FROM_BASELINE, 
                        court_center_y, 
                        0.f 
                    }
                };
            } 
            else if (i == 1) 
            {
                gameState.team1Hoop = hoop.id;
                hoop_pos = Position { 
                    Vector3{
                        court_start_x + NBA_COURT_WIDTH - HOOP_OFFSET_FROM_BASELINE, 
                        court_center_y, 
                        0.f 
                    }
                };
            } 
            else 
            {
                hoop_pos = Position 
                { 
                    Vector3{
                        grid->startX + 10.0f + i * 5.0f,   
                        grid->startY + 10.0f,  
                        0.f
                    }
                };
            }

            ctx.get<Position>(hoop) = hoop_pos;
            ctx.get<Reset>(hoop) = Reset{0};
            ctx.get<Done>(hoop).episodeDone = 0.f;
            ctx.get<CurStep>(hoop).step = 0;
            ctx.get<ImAHoop>(hoop) = ImAHoop{};
            ctx.get<ScoringZone>(hoop) = ScoringZone
            {
                HOOP_SCORE_ZONE_SIZE,
                .1f,
                Vector3{hoop_pos.position.x, hoop_pos.position.y, hoop_pos.position.z}
            };
        }
    }
}
