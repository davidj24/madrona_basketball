#include "sim.hpp"
#include "types.hpp"
#include "constants.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath> // For acosf


using namespace madrona;
using namespace madrona::math;

const Vector3 AGENT_BASE_FORWARD = {0, 1, 0}; 

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

    inline void assignInbounder(Engine &ctx, Entity ball_entity, Position ball_pos, uint32_t new_team_idx, Quat new_orientation, bool is_oob)
    {
        GameState &gameState = ctx.singleton<GameState>();
        bool inbounder_assigned = false;

        // Find the first available player on the new team.
        auto agent_query = ctx.query<Entity, Team, InPossession, Position, Orientation, Inbounding>();

        ctx.iterateQuery(agent_query, [&](Entity agent_entity, Team &agent_team, InPossession &in_possession, Position &agent_pos, Orientation &agent_orient, Inbounding &inbounding)
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
    
    inline Vector3 findVectorToCenter(Engine &ctx, Position entity_pos)
    {
        const GridState *grid = ctx.data().grid;
        return (Vector3{grid->startX, grid->startY, 0.f} - entity_pos.position).normalize();
    }
    
    inline int32_t getShotPointValue(Position shot_pos, Position hoop_pos) 
    {
        float distance_to_hoop = (shot_pos.position - hoop_pos.position).length();
        
        // 1. Check if the shot is in the corner lane, relative to the court's position.
        bool isInCornerLane = (shot_pos.position.y < COURT_MIN_Y + CORNER_3_FROM_SIDELINE_M || 
                            shot_pos.position.y > COURT_MIN_Y + COURT_WIDTH_M - CORNER_3_FROM_SIDELINE_M);

        if (isInCornerLane) {
            // 2. If so, check if the shot is within the corner's length, relative to the court's position.
            bool isShootingAtLeftHoop = hoop_pos.position.x < WORLD_WIDTH_M / 2.0f;
            
            if (isShootingAtLeftHoop) {
                if (shot_pos.position.x <= COURT_MIN_X + CORNER_3_LENGTH_FROM_BASELINE_M) {
                    return 3;
                }
            } else { // Shooting at the right hoop
                if (shot_pos.position.x >= COURT_MIN_X + COURT_LENGTH_M - CORNER_3_LENGTH_FROM_BASELINE_M) {
                    return 3;
                }
            }
        }

        // 3. If not a valid corner 3, check the distance against the arc.
        // If the shot is beyond the 3-point arc distance, it's a 3-pointer
        if (distance_to_hoop >= ARC_RADIUS_M) {
            return 3;
        }

        // 4. If none of the 3-point conditions are met, it is a 2-point shot.
        return 2;
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
        registry.registerComponent<ActionMask>();
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
        registry.exportColumn<Agent, ActionMask>((uint32_t)ExportID::ActionMask);
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

    
    
    inline void updatePointsWorthSystem(Engine &ctx,
                                        Position &agent_pos,
                                        InPossession &in_possession,
                                        Team &team)
    {
        // Get all hoop positions
        Position hoop_positions[NUM_HOOPS];
        uint32_t hoop_ids[NUM_HOOPS];
        int hoop_idx = 0;
        
        auto hoop_query = ctx.query<Entity, Position, ImAHoop>();
        ctx.iterateQuery(hoop_query, [&](Entity hoop_entity, Position &hoop_pos, ImAHoop &) {
            if (hoop_idx < NUM_HOOPS) {
                hoop_positions[hoop_idx] = hoop_pos;
                hoop_ids[hoop_idx] = hoop_entity.id;
                hoop_idx++;
            }
        });

        // Find the hoop this agent should be shooting at (opposing team's hoop)
        Position target_hoop_pos;
        bool found_target_hoop = false;
        
        for (int i = 0; i < hoop_idx; i++) 
        {
            if (hoop_ids[i] != team.defendingHoopID) 
            {
                target_hoop_pos = hoop_positions[i];
                found_target_hoop = true;
                break;
            }
        }
        
        // Calculate points worth for this agent's current position
        if (found_target_hoop) 
        {
            in_possession.pointsWorth = getShotPointValue(agent_pos, target_hoop_pos);
        } 
        else 
        {
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
        auto basketball_query = ctx.query<Entity, Position, Grabbed, BallPhysics>();
        if (action_mask.can_grab == 0.f || action.grab == 0.f) {return;}
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

        if (action_mask.can_pass == 0.f || action.pass == 0.f) {return;}
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
                            ActionMask &action_mask,
                            Position agent_pos,
                            Orientation &agent_orientation,
                            Inbounding &inbounding,
                            InPossession &in_possession,
                            Team &team)
    {
        if (action_mask.can_shoot == 0.f || action.shoot == 0.f) {return;}

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
                            Orientation &agent_orientation)
    {
        const GridState *grid = ctx.data().grid;
        if (action.rotate != 0)
        {
            float turn_angle = (pi/180.f) * action.rotate * 6;
            Quat turn = Quat::angleAxis(turn_angle, Vector3{0, 0, 1});
            agent_orientation.orientation = turn * agent_orientation.orientation;
        }

        if (action_mask.can_move == 0.f || action.moveSpeed == 0) {return;}

        if (action.moveSpeed > 0)
        {
            // Treat moveSpeed as a velocity in meters/second, not a distance.
            // Let's say a moveSpeed of 1 corresponds to 1 m/s.
            float agent_velocity_magnitude = action.moveSpeed * 4;
            if (in_possession.hasBall == 1) {agent_velocity_magnitude *= .8;}

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
                                 GrabCooldown &grab_cooldown,
                                 InPossession &in_possession,
                                 Inbounding &inbounding)
    {
        GameState &gameState = ctx.singleton<GameState>();

        action_mask.can_move = 1.f;
        action_mask.can_grab = 1.f;
        action_mask.can_pass = 0.f;
        action_mask.can_shoot = 0.f;

        // Offensive actions
        if (in_possession.hasBall)
        {
            action_mask.can_pass = 1.f;
            action_mask.can_shoot = 1.f;
        }

        if (gameState.inboundingInProgress == 1.f)
        {
            action_mask.can_shoot = 0.f;
            action_mask.can_grab = 0.f;
            if (inbounding.imInbounding == 1.f && gameState.liveBall == 0.f) 
            {
                action_mask.can_move = 0.f;
            }
        }

        if (grab_cooldown.cooldown > 0.f)
        {
            action_mask.can_grab = 0.f;
        }
    }



    inline void agentCollisionSystem(Engine &ctx, 
                                     Entity entity_a, 
                                     Position &entity_a_pos,
                                     InPossession &in_possession_a)
    {
        // Query for all agents to get their positions.
        // We need Entity to compare IDs and Position to read/write locations.
        auto agent_query = ctx.query<Entity, Position, InPossession>();
        ctx.iterateQuery(agent_query, [&](Entity entity_b, Position &entity_b_pos, InPossession &in_possession_b) 
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


    //=================================================== Hoop Systems ===================================================
    inline void scoreSystem(Engine &ctx,
                        Entity hoop_entity,
                        Position &hoop_pos,
                        ScoringZone &scoring_zone)
    {
        GameState &gameState = ctx.singleton<GameState>();
        
        auto ball_query = ctx.query<Entity, Position, BallPhysics>();
        ctx.iterateQuery(ball_query, [&] (Entity ball_entity, Position &ball_pos, BallPhysics &ball_physics)
        {
            float distance_to_hoop = std::sqrt((ball_pos.position.x - hoop_pos.position.x) * (ball_pos.position.x - hoop_pos.position.x) + 
                                            (ball_pos.position.y - hoop_pos.position.y) * (ball_pos.position.y - hoop_pos.position.y));

            if (distance_to_hoop <= scoring_zone.radius && ball_physics.inFlight && gameState.liveBall == 1.f) 
            {
                // Use the point value that was calculated when the shot was taken
                int32_t points_scored = ball_physics.shotPointValue;
                
                // Find which team is defending this hoop (has defendingHoopID == hoop_entity.id)
                uint32_t inbounding_team_idx = 0; // Default fallback
                auto defending_team_query = ctx.query<Team>();
                ctx.iterateQuery(defending_team_query, [&](Team &team) {
                    if (team.defendingHoopID == (uint32_t)hoop_entity.id) 
                    {
                        inbounding_team_idx = (uint32_t)team.teamIndex;
                        return; // Found the defending team
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
                ball_pos = inbound_spot;
                
                // Clear shot information since the shot scored
                ball_physics.shotByAgentID = ENTITY_ID_PLACEHOLDER;
                ball_physics.shotByTeamID = ENTITY_ID_PLACEHOLDER;
                ball_physics.shotPointValue = 2; // Reset to default

                // Set up the inbound for the defending team.
                inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
                assignInbounder(ctx, ball_entity, inbound_spot, inbounding_team_idx, inbound_orientation, false);
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
            action = Action{0, 0, 0, 0, 0, 0};
            float x = (agent_i < 4) ? agent_start_x[agent_i] : grid->startX;
            pos = Position{Vector3{x, grid->startY, 0.f}};
            reset.resetNow = 0; // Clear the flag
            inbounding = Inbounding{false, true};
            done.episodeDone = 1.f; // Signal to python that a reset happened
            curstep.step = 0;
            inpos = {false, ENTITY_ID_PLACEHOLDER, 2}; // Initialize with 2 points (default)
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
            ballphys = BallPhysics{false, Vector3::zero(), ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, 2};
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
        auto touched_agent_query = ctx.query<Entity, Position, Team>();
        ctx.iterateQuery(touched_agent_query, [&] (Entity agent_entity, Position &agent_pos, Team &team)
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
            ball_physics.inFlight = false;
            ball_physics.velocity = Vector3::zero();
            gameState.liveBall = 0.f;

            // The team that did NOT last touch the ball gets possession.
            uint32_t new_team_idx = 1 - ball_physics.lastTouchedByTeamID;

            // Find the player who had the ball and reset their position
            auto agent_query = ctx.query<InPossession, Position>();
            ctx.iterateQuery(agent_query, [&](InPossession &in_possession, Position &agent_pos)
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

        gameState.liveBall = 0.f;
        
        auto inbounder_query = ctx.query<Inbounding, InPossession, Position>();
        ctx.iterateQuery(inbounder_query, [&](Inbounding &inb, InPossession &poss, Position &agent_pos) {
            if (inb.imInbounding) {
                ball_to_turnover_id = poss.ballEntityID;
                
                inb.imInbounding = false;
                poss.hasBall = false;
                poss.ballEntityID = ENTITY_ID_PLACEHOLDER;

                agent_pos.position += findVectorToCenter(ctx, agent_pos);
            }
        });

        if (ball_to_turnover_id != ENTITY_ID_PLACEHOLDER) {
            auto ball_query = ctx.query<Position, Entity, Grabbed>();
            ctx.iterateQuery(ball_query, [&](Position &ball_pos, Entity ball_entity, Grabbed &grabbed) {
                if (ball_entity.id == (int32_t)ball_to_turnover_id) {
                    grabbed = {false, ENTITY_ID_PLACEHOLDER};
                    Quat inbound_orientation = findRotationBetweenVectors(AGENT_BASE_FORWARD, findVectorToCenter(ctx, ball_pos));
                    assignInbounder(ctx, ball_entity, ball_pos, new_team_idx, inbound_orientation, false);
                }
            });
        }
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
        auto &observations_array = observations.observationsArray;
        GameState &gameState = ctx.singleton<GameState>();
        uint32_t index = 0;


        // Self State for agent:
        observations_array[index++] = agent_pos.position.x;
        observations_array[index++] = agent_pos.position.y;
        observations_array[index++] = agent_pos.position.z;
        observations_array[index++] = agent_orientation.orientation.w;
        observations_array[index++] = agent_orientation.orientation.x;
        observations_array[index++] = agent_orientation.orientation.y;
        observations_array[index++] = agent_orientation.orientation.z;
        observations_array[index++] = in_possession.hasBall;
        observations_array[index++] = in_possession.pointsWorth; // How many points "I" would get if "I" scored from here
        observations_array[index++] = inbounding.imInbounding;
        observations_array[index++] = agent_team.teamIndex;
        observations_array[index++] = grab_cooldown.cooldown;
        
        // Game State
        observations_array[index++] = gameState.gameClock;
        observations_array[index++] = gameState.shotClock;
        observations_array[index++] = gameState.inboundClock;
        observations_array[index++] = gameState.period;
        observations_array[index++] = gameState.inboundingInProgress;

        float our_score = 0.f;
        float opponents_score = 0.f;
        if (agent_team.teamIndex == 0) 
        {
            our_score = gameState.team0Score;
            opponents_score = gameState.team1Score;
        }
        else 
        {
            our_score = gameState.team1Score;
            opponents_score = gameState.team0Score;
        }
        observations_array[index++] = our_score;
        observations_array[index++] = opponents_score;
        observations_array[index++] = gameState.teamInPossession;
        observations_array[index++] = gameState.liveBall;
        

        // Ball State
        auto ball_query = ctx.query<BallPhysics, Grabbed>();
        ctx.iterateQuery(ball_query, [&] (BallPhysics &ball_physics, Grabbed &grabbed)
        {
            observations_array[index++] = ball_physics.inFlight;
            observations_array[index++] = ball_physics.velocity.x;
            observations_array[index++] = ball_physics.velocity.y;
            observations_array[index++] = ball_physics.velocity.z;
            observations_array[index++] = ball_physics.lastTouchedByAgentID;
            observations_array[index++] = ball_physics.lastTouchedByTeamID;
            observations_array[index++] = ball_physics.shotByAgentID;
            observations_array[index++] = ball_physics.shotByTeamID;
            observations_array[index++] = ball_physics.shotPointValue;
        });


        // Other Agents State
        ctx.iterateQuery(ctx.query<Entity, Position, Orientation, InPossession, Inbounding, Team, GrabCooldown>(), [&] (Entity other_agent_entity, Position &other_agent_pos, 
                                                                                                              Orientation &other_agent_orientation,
                                                                                                              InPossession &other_agent_in_possession, 
                                                                                                              Inbounding &other_agent_inbounding, 
                                                                                                              Team &other_agent_team, 
                                                                                                              GrabCooldown &other_agent_grab_cooldown)
        {
            if (other_agent_entity.id == agent_entity.id) {return;}

            if (other_agent_team.teamIndex == agent_team.teamIndex) // This is a teammate
            {

            }
            observations_array[index++] = other_agent_pos.position.x;
            observations_array[index++] = other_agent_pos.position.y;
            observations_array[index++] = other_agent_pos.position.z;
            observations_array[index++] = other_agent_orientation.orientation.w;
            observations_array[index++] = other_agent_orientation.orientation.x;
            observations_array[index++] = other_agent_orientation.orientation.y;
            observations_array[index++] = other_agent_orientation.orientation.z;
            observations_array[index++] = other_agent_in_possession.hasBall;
            observations_array[index++] = other_agent_in_possession.pointsWorth; // How many points this agent would get if they scored
            observations_array[index++] = other_agent_inbounding.imInbounding;
            observations_array[index++] = other_agent_team.teamIndex;
            observations_array[index++] = other_agent_grab_cooldown.cooldown;
        });

    };

    // =================================================== Task Graph ===================================================
    void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                    const Config &)
    {
        TaskGraphBuilder &builder = taskgraph_mgr.init(0);

        auto actionMaskingNode = builder.addToGraph<ParallelForNode<Engine, actionMaskSystem,
            ActionMask, GrabCooldown, InPossession, Inbounding>>({});
        
        auto moveAgentSystemNode = builder.addToGraph<ParallelForNode<Engine, moveAgentSystem,
            Action, ActionMask, Position, InPossession, Inbounding, Orientation>>({actionMaskingNode});

        auto grabSystemNode = builder.addToGraph<ParallelForNode<Engine, grabSystem,
            Entity, Action, ActionMask, Position, InPossession, Team, GrabCooldown>>({actionMaskingNode});

        auto passSystemNode = builder.addToGraph<ParallelForNode<Engine, passSystem,
            Entity, Action, ActionMask, Orientation, InPossession, Inbounding>>({actionMaskingNode});
        
        auto shootSystemNode = builder.addToGraph<ParallelForNode<Engine, shootSystem,
            Entity, Action, ActionMask, Position, Orientation, Inbounding, InPossession, Team>>({actionMaskingNode});

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

        auto updatePointsWorthNode = builder.addToGraph<ParallelForNode<Engine, updatePointsWorthSystem,
            Position, InPossession, Team>>({});

        auto agentCollisionNode = builder.addToGraph<ParallelForNode<Engine, agentCollisionSystem,
            Entity, Position, InPossession>>({moveAgentSystemNode});
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
            .outOfBoundsCount = 0.f,
            .inboundClock = 0.0f
        };

        // Make sure to add the Reset component to the WorldClock entity
        Entity worldClock = ctx.makeEntity<WorldClock>();
        ctx.get<IsWorldClock>(worldClock) = {};
        ctx.get<Reset>(worldClock) = {0}; // Initialize resetNow to 0

        // Initialize GameState and create hoops first
        GameState &gameState = ctx.singleton<GameState>();
        for (int i = 0; i < NUM_HOOPS; i++) 
        {
            Entity hoop = ctx.makeEntity<Hoop>();
            Position hoop_pos;
            
            float court_start_x = (grid->width - COURT_LENGTH_M) / 2.0f;
            float court_center_y = grid->height / 2.0f;
            
            if (i == 0) 
            {
                gameState.team0Hoop = hoop.id;
                hoop_pos = Position { 
                    Vector3{
                        court_start_x + HOOP_FROM_BASELINE_M, 
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
                        court_start_x + COURT_LENGTH_M - HOOP_FROM_BASELINE_M, 
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

        // Now create agents with proper hoop references
        std::vector<Vector3> team_colors = {Vector3{0, 100, 255}, Vector3{255, 0, 100}};
        for (int i = 0; i < NUM_AGENTS; i++) 
        {
            Entity agent = ctx.makeEntity<Agent>();
            ctx.get<Action>(agent) = Action{0, 0, 0, 0, 0, 0};
            ctx.get<ActionMask>(agent) = ActionMask{0.f, 0.f, 0.f, 0.f};
            ctx.get<Position>(agent) = Position 
            {
                Vector3{
                    grid->startX - 1 - (-2*(i % 2)),
                    grid->startY - 2 + i/2,
                    0.f
                }
            };
            ctx.get<Reset>(agent) = Reset{0};
            ctx.get<Inbounding>(agent) = Inbounding{false, true};
            ctx.get<Reward>(agent).r = 0.f;
            ctx.get<Done>(agent).episodeDone = 0.f;
            ctx.get<CurStep>(agent).step = 0;
            ctx.get<InPossession>(agent) = {false, ENTITY_ID_PLACEHOLDER, 2};
            ctx.get<Orientation>(agent) = Orientation {Quat::id()};
            ctx.get<GrabCooldown>(agent) = GrabCooldown{0.f};
            
            // Use actual hoop entity IDs from gameState
            uint32_t defending_hoop_id = (i % 2 == 0) ? gameState.team0Hoop : gameState.team1Hoop;
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
            ctx.get<BallPhysics>(basketball) = BallPhysics {false, Vector3::zero(), ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, 2};
        }
    }
}
