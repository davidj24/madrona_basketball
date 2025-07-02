#include "sim.hpp"
#include "types.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath> // For acosf


using namespace madrona;
using namespace madrona::math;



// Computes the rotation needed to align the 'start' vector with the 'target' vector.
inline Quat findRotationBetweenVectors(Vector3 start, Vector3 target) 
{
    // Ensure the vectors are normalized (unit length)
    start.normalize();
    target.normalize();

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
    rotation_axis.normalize();

    // The angle is the arccosine of the dot product.
    float rotation_angle = acosf(dot_product);

    return Quat::angleAxis(rotation_angle, rotation_axis);
}

inline int32_t getShotPointValue(madsimple::Position shot_pos, madsimple::Position hoop_pos, float distance_to_hoop) 
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
    bool isInCornerLane = (shot_pos.y < court_min_y + CORNER_3_FROM_SIDELINE_M || 
                           shot_pos.y > court_min_y + COURT_WIDTH_M - CORNER_3_FROM_SIDELINE_M);

    if (isInCornerLane) {
        // 2. If so, check if the shot is within the corner's length, relative to the court's position.
        bool isShootingAtLeftHoop = hoop_pos.x < WORLD_WIDTH_M / 2.0f;
        
        if (isShootingAtLeftHoop) {
            if (shot_pos.x <= court_min_x + CORNER_3_LENGTH_FROM_BASELINE_M) {
                return 3;
            }
        } else { // Shooting at the right hoop
            if (shot_pos.x >= court_min_x + COURT_LENGTH_M - CORNER_3_LENGTH_FROM_BASELINE_M) {
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

namespace madsimple {

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



        // ================================================= Tensor Exports For Viewer =================================================
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

            float new_x = ball_pos.x + dx;
            float new_y = ball_pos.y + dy;

            new_x = std::clamp(new_x, 0.f, grid->width);
            new_y = std::clamp(new_y, 0.f, grid->height);

            ball_pos.x = new_x;
            ball_pos.y = new_y;
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
        float new_x = ball_pos.x + ball_physics.velocity[0];
        float new_y = ball_pos.y + ball_physics.velocity[1];
        float new_z = ball_pos.z + ball_physics.velocity[2];

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
            ball_pos.x = new_x;
            ball_pos.y = new_y;
            ball_pos.z = new_z;
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
            float distance_between_ball_and_player = sqrt((basketball_pos.x - agent_pos.x) * (basketball_pos.x - agent_pos.x) +
                                (basketball_pos.y - agent_pos.y) * (basketball_pos.y - agent_pos.y));
            
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
                ball_physics.inFlight = true;
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
            attacking_hoop_pos.x - agent_pos.x,
            attacking_hoop_pos.y - agent_pos.y,
            0.f
        };


        float distance_to_hoop = shot_vector.length();

        // Calculate intended angle towards hoop
        float intended_direction = std::atan2(shot_vector.x, shot_vector.y);

        // Mess up angle based on distance
        float direction_deviation_per_meter = 0.002f; // radians per meter distance
        float stddev = direction_deviation_per_meter * distance_to_hoop;
        static thread_local std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, stddev);
        float direction_deviation = dist(rng);
        float shot_direction = intended_direction + direction_deviation;


        // Mess up angle based on contest level (how close nearest defender is)


        // Mess up angle based on agent velocity




        // This is the final, correct trajectory vector for the ball
        Vector3 final_shot_vec = Vector3{std::sin(shot_direction), std::cos(shot_direction), 0.f};


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
                ball_physics.velocity = final_shot_vec * .15f;
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
        // Define the duration of a single simulation step.
        // For example, if your simulation runs at 30 steps per second.
        const float delta_time = 1.0f / 60.0f;

        const GridState *grid = ctx.data().grid;
        if (action.rotate != 0)
        {
            // Rotation logic is fine as it is
            float turn_angle = (pi/180.f) * action.rotate * 3;
            Quat turn = Quat::angleAxis(turn_angle, Vector3{0, 0, 1});
            agent_orientation.orientation = turn * agent_orientation.orientation;
        }

        if (action.moveSpeed > 0)
        {
            // Treat moveSpeed as a velocity in meters/second, not a distance.
            // Let's say a moveSpeed of 1 corresponds to 1 m/s.
            float agent_velocity_magnitude = action.moveSpeed * 5;
            if (in_possession.hasBall ==1) {agent_velocity_magnitude *= .8;}

            constexpr float angle_between_directions = pi / 4.f;
            float move_angle = action.moveAngle * angle_between_directions;

            // Calculate velocity vector components
            float vel_x = std::sin(move_angle);
            float vel_y = -std::cos(move_angle); // Your forward is -Y

            // Calculate distance to move this frame
            float dx = vel_x * agent_velocity_magnitude * delta_time;
            float dy = vel_y * agent_velocity_magnitude * delta_time;

            // Update position (now using floats)
            float new_x = agent_pos.x + dx;
            float new_y = agent_pos.y + dy;

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
                agent_pos.x = new_x;
                agent_pos.y = new_y;
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
            float distance_to_hoop = std::sqrt((ball_pos.x - hoop_pos.x) * (ball_pos.x - hoop_pos.x) + 
                                               (ball_pos.y - hoop_pos.y) * (ball_pos.y - hoop_pos.y));
            if (distance_to_hoop <= scoring_zone.radius && ball_physics.inFlight) 
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
    inline void tick(Engine &ctx,
                    Reset &reset,
                    Position &position,
                    Reward &reward, //add later
                    Done &done,
                    CurStep &episode_step,
                    GrabCooldown &grab_cooldown)
    {
        const GridState *grid = ctx.data().grid;

        Position new_pos = position;
        grab_cooldown.cooldown = std::max(0.f, grab_cooldown.cooldown - 1.f); // Decrease cooldown if it's greater than 0

        bool episode_done = false;
        if (reset.resetNow != 0) 
        {
            reset.resetNow = 0;
            episode_done = true;
        }

        uint32_t cur_step = episode_step.step;

        if (cur_step == ctx.data().maxEpisodeLength - 1) {episode_done = true;}

        if (episode_done) 
        {
            done.episodeDone = 1.f;

            // Reset singleton GameState
            GameState &gameState = ctx.singleton<GameState>();
            gameState = GameState{
                .inboundingInProgress = 0.0f,
                .liveBall = 1.0f,
                .period = 1.0f,
                .teamInPossession = 0.0f,
                .team0Hoop = 0.0f,
                .team0Score = 0.0f,
                .team1Hoop = 1.0f,
                .team1Score = 0.0f,
                .gameClock = 720.0f,
                .shotClock = 24.0f
            };

            // Reset all agents (no index math)
            auto agent_query = ctx.query<Action, Position, Reset, Inbounding, Reward, Done, CurStep, InPossession, Orientation, Team>();
            float agent_start_x[4] = {grid->startX - 2.0f, grid->startX - 1.0f, grid->startX + 0.0f, grid->startX + 1.0f};
            int agent_i = 0;
            ctx.iterateQuery(agent_query, [&](Action &action, Position &pos, Reset &reset, Inbounding &inbounding, Reward &reward, Done &done, CurStep &curstep, InPossession &inpos, Orientation &orient, Team &team) {
                action = Action{0, 0, 0, 0, 0, 0, 0, 0};
                float x = (agent_i < 4) ? agent_start_x[agent_i] : grid->startX;
                pos = Position{x, grid->startY, 0.f};
                reset = Reset{0};
                inbounding = Inbounding{false, true};
                reward.r = 0.f;
                done.episodeDone = 0.f;
                curstep.step = 0;
                inpos = {false, ENTITY_ID_PLACEHOLDER};
                orient = Orientation{Quat::id()};
                grab_cooldown = GrabCooldown{0.f};
                agent_i++;
            });

            // Reset all basketballs
            auto basketball_query = ctx.query<Position, Reset, Done, CurStep, Grabbed, BallPhysics>();
            ctx.iterateQuery(basketball_query, [&](Position &pos, Reset &reset, Done &done, CurStep &curstep, Grabbed &grabbed, BallPhysics &ballphys) {
                pos = Position{grid->startX, grid->startY, 0.f};
                reset = Reset{0};
                done.episodeDone = 0.f;
                curstep.step = 0;
                grabbed = Grabbed{false, ENTITY_ID_PLACEHOLDER};
                ballphys = BallPhysics{false, Vector3::zero(), ENTITY_ID_PLACEHOLDER};
            });

            // Reset all hoops (no index math)
            auto hoop_query = ctx.query<Position, Reset, Done, CurStep, ImAHoop, ScoringZone>();
            int hoop_i = 0;
            ctx.iterateQuery(hoop_query, [&](Position &pos, Reset &reset, Done &done, CurStep &curstep, ImAHoop &, ScoringZone &zone) {
                if (hoop_i == 0)
                    pos = Position{3.0f, grid->height / 2.0f, 0.f};
                else if (hoop_i == 1)
                    pos = Position{grid->width - 3.0f, grid->height / 2.0f, 0.f};
                else
                    pos = Position{grid->startX + 10.0f + hoop_i * 5.0f, grid->startY + 10.0f, 0.f};
                reset = Reset{0};
                done.episodeDone = 0.f;
                curstep.step = 0;
                zone = ScoringZone{1.0f, 2.0f, Vector3{pos.x, pos.y, pos.z}};
                hoop_i++;
            });

            // Reset this agent's position
            new_pos = Position{grid->startX, grid->startY, 0.f};
            episode_step.step = 0;
        }
        else 
        {
            done.episodeDone = 0.f;
            episode_step.step = cur_step + 1;
        }

        position = new_pos;

        // Calculate reward based on current position (convert to discrete for cell lookup)
        int32_t discrete_x = (int32_t)(position.x * grid->cellsPerMeter);
        int32_t discrete_y = (int32_t)(position.y * grid->cellsPerMeter);
        discrete_x = std::clamp(discrete_x, 0, grid->discreteWidth - 1);
        discrete_y = std::clamp(discrete_y, 0, grid->discreteHeight - 1);
        
        const Cell &cur_cell = grid->cells[discrete_y * grid->discreteWidth + discrete_x];
        reward.r = cur_cell.reward;
    }



    inline void updateLastTouchSystem(Engine &ctx,
                                    Position &ball_pos,
                                    BallPhysics &ball_physics)
    {
        auto touched_agent_query = ctx.query<Position, Team>();
        ctx.iterateQuery(touched_agent_query, [&] (Position &agent_pos, Team &team)
        {
            // Check if agent is within touch distance (0.5 meters)
            float distance = sqrt((ball_pos.x - agent_pos.x) * (ball_pos.x - agent_pos.x) +
                                (ball_pos.y - agent_pos.y) * (ball_pos.y - agent_pos.y) +
                                (ball_pos.z - agent_pos.z) * (ball_pos.z - agent_pos.z));
            
            if (distance <= 0.5f) 
            {
                ball_physics.lastTouchedByID = (uint32_t)team.teamIndex;
            }
        });
    }



    inline void outOfBoundsSystem(Engine &ctx,
                                Entity ball_entity,
                                Position &ball_pos,
                                Grabbed &grabbed,
                                BallPhysics &ball_physics)
{
    GameState &gameState = ctx.singleton<GameState>();

    // --- Define Court/World Dimensions (These MUST MATCH your Python viewer.py) ---
    const float COURT_LENGTH_M = 28.65f;
    const float COURT_WIDTH_M = 15.24f;
    const float WORLD_WIDTH_M = 28.65f * 1.1f;
    const float WORLD_HEIGHT_M = 15.24f * 1.1f;

    // --- Calculate the court's actual boundaries within the world ---
    const float court_min_x = (WORLD_WIDTH_M - COURT_LENGTH_M) / 2.0f;
    const float court_max_x = court_min_x + COURT_LENGTH_M;
    const float court_min_y = (WORLD_HEIGHT_M - COURT_WIDTH_M) / 2.0f;
    const float court_max_y = court_min_y + COURT_WIDTH_M;

    // Check if the ball's center has crossed the court boundaries
    if (ball_pos.x < court_min_x || ball_pos.x > court_max_x ||
        ball_pos.y < court_min_y || ball_pos.y > court_max_y)
    {
        // Reset the ball physics
        ball_physics.inFlight = false;
        ball_physics.velocity = Vector3::zero();
        gameState.outOfBoundsCount++;

        auto agent_query = ctx.query<Entity, Team, InPossession, Position, Inbounding>();
        ctx.iterateQuery(agent_query, [&] (Entity agent_entity, Team &agent_team, InPossession &in_possession, Position &agent_pos, Inbounding &inbounding)
        {
            // If an agent has the ball, we need to reset their position
            if (in_possession.ballEntityID == ball_entity.id && agent_team.teamIndex == ball_physics.lastTouchedByID && inbounding.imInbounding == false)
            {
                // Note: Using a generic centered start position instead of the old grid->startX
                agent_pos = Position {
                    WORLD_WIDTH_M / 2.0f,
                    WORLD_HEIGHT_M / 2.0f,
                    0.f
                };
                in_possession.hasBall = false;
                in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
            }

            if (agent_team.teamIndex != ball_physics.lastTouchedByID && gameState.inboundingInProgress < 0.5f)
            {
                inbounding.imInbounding = true;
                gameState.inboundingInProgress = 1.0f;
                agent_pos = ball_pos;
                grabbed.isGrabbed = true;
                grabbed.holderEntityID = agent_entity.id;
                in_possession.hasBall = true;
                in_possession.ballEntityID = ball_entity.id;
                gameState.teamInPossession = (float)agent_team.teamIndex;
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

        auto tickNode = builder.addToGraph<ParallelForNode<Engine, tick,
            Reset, Position, Reward, Done, CurStep, GrabCooldown>>({});
        
        // builder.addToGraph<ParallelForNode<Engine, moveBallRandomly,
        //     Position, RandomMovement>>({});

        auto grabSystemNode = builder.addToGraph<ParallelForNode<Engine, grabSystem,
            Entity, Action, Position, InPossession, Team, GrabCooldown>>({});

        auto passSystemNode = builder.addToGraph<ParallelForNode<Engine, passSystem,
            Entity, Action, Orientation, InPossession, Inbounding>>({});

        auto moveBallSystemNode = builder.addToGraph<ParallelForNode<Engine, moveBallSystem,
            Position, BallPhysics, Grabbed>>({grabSystemNode});

        auto outOfBoundsSystemNode = builder.addToGraph<ParallelForNode<Engine, outOfBoundsSystem,
            Entity, Position, Grabbed, BallPhysics>>({passSystemNode, moveBallSystemNode});

        auto updateLastTouchSystemNode = builder.addToGraph<ParallelForNode<Engine, updateLastTouchSystem,
            Position, BallPhysics>>({});

        auto shootSystemNode = builder.addToGraph<ParallelForNode<Engine, shootSystem,
            Entity, Action, Position, Orientation, Inbounding, InPossession, Team>>({});

        auto scoreSystemNode = builder.addToGraph<ParallelForNode<Engine, scoreSystem,
            Entity, Position, ScoringZone>>({shootSystemNode});
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
            .team0Hoop = 0.0f,  // Team 0 attacks hoop 0
            .team0Score = 0.0f,
            .team1Hoop = 1.0f,  // Team 1 attacks hoop 1  
            .team1Score = 0.0f,
            .gameClock = 720.0f,
            .shotClock = 24.0f,
            .scoredBaskets = 0.f,
            .outOfBoundsCount = 0.f
        };

        std::vector<Vector3> team_colors = {Vector3{0, 100, 255}, Vector3{255, 0, 100}};
        for (int i = 0; i < NUM_AGENTS; i++) 
        {
            Entity agent = ctx.makeEntity<Agent>();
            ctx.get<Action>(agent) = Action{0, 0, 0, 0, 0, 0, 0, 0}; // Initialize with no action - fixed field count
            ctx.get<Position>(agent) = Position 
            {
                grid->startX + (i - 2) * 1.0f, // Space agents 1 meter apart
                grid->startY,
                0.f
            };
            ctx.get<Reset>(agent) = Reset{0}; // Initialize reset component
            ctx.get<Inbounding>(agent) = Inbounding{false, true}; // Fixed field initialization
            ctx.get<Reward>(agent).r = 0.f;
            ctx.get<Done>(agent).episodeDone = 0.f;
            ctx.get<CurStep>(agent).step = 0;
            ctx.get<InPossession>(agent) = {false, ENTITY_ID_PLACEHOLDER};
            ctx.get<Orientation>(agent) = Orientation {Quat::id()};
            ctx.get<GrabCooldown>(agent) = GrabCooldown{0.f};
            
            // Set defending hoop based on team
            uint32_t defending_hoop_id = (i % 2 == 0) ? 1 : 0; // Team 0 defends hoop 1, Team 1 defends hoop 0
            
            ctx.get<Team>(agent) = Team{i % 2, team_colors[i % 2], defending_hoop_id}; // Fixed initialization
        };

        

        for (int i = 0; i < NUM_BASKETBALLS; i++) 
        {
            Entity basketball = ctx.makeEntity<Basketball>();
            ctx.get<Position>(basketball) = Position 
            {
                grid->startX,   
                grid->startY,  
                0.f
            };
            ctx.get<Reset>(basketball) = Reset{0}; // Initialize reset component
            ctx.get<Done>(basketball).episodeDone = 0.f;
            ctx.get<CurStep>(basketball).step = 0;
            ctx.get<Grabbed>(basketball) = Grabbed {false, ENTITY_ID_PLACEHOLDER};
            ctx.get<BallPhysics>(basketball) = BallPhysics {false, Vector3::zero(), ENTITY_ID_PLACEHOLDER};
            

            // Keep random movement commented out as requested
            // ctx.get<RandomMovement>(basketball) = RandomMovement {
            //     0.f,
            //     1.f + i * 2.f  // Different movement intervals: 1s, 3s, 5s...
            // };
        }

        GameState &gameState = ctx.singleton<GameState>();
        for (int i = 0; i < NUM_HOOPS; i++) 
        {
            Entity hoop = ctx.makeEntity<Hoop>();
            Position hoop_pos;
            
            // Define NBA court dimensions (same as in viewer)
            const float NBA_COURT_WIDTH = 28.65f;  // meters
            const float NBA_COURT_HEIGHT = 15.24f; // meters
            const float HOOP_OFFSET_FROM_BASELINE = 1.575f; // Distance from baseline to hoop center
            
            // Calculate court position within the world (centered)
            float court_start_x = (grid->width - NBA_COURT_WIDTH) / 2.0f;
            float court_start_y = (grid->height - NBA_COURT_HEIGHT) / 2.0f;
            float court_center_y = grid->height / 2.0f;
            
            if (i == 0) 
            {
                gameState.team0Hoop = hoop.id;
                // Left hoop - positioned at left baseline + offset, center court vertically
                hoop_pos = Position { 
                    court_start_x + HOOP_OFFSET_FROM_BASELINE, 
                    court_center_y, 
                    0.f 
                };
            } 
            else if (i == 1) 
            {
                gameState.team1Hoop = hoop.id;
                // Right hoop - positioned at right baseline - offset, center court vertically
                hoop_pos = Position { 
                    court_start_x + NBA_COURT_WIDTH - HOOP_OFFSET_FROM_BASELINE, 
                    court_center_y, 
                    0.f 
                };
            } 
            else 
            {
                // Additional hoops (if NUM_HOOPS > 2) - fallback positioning
                hoop_pos = Position 
                { 
                    grid->startX + 10.0f + i * 5.0f,   
                    grid->startY + 10.0f,  
                    0.f
                };
            }

            ctx.get<Position>(hoop) = hoop_pos;
            ctx.get<Reset>(hoop) = Reset{0};
            ctx.get<Done>(hoop).episodeDone = 0.f;
            ctx.get<CurStep>(hoop).step = 0;
            ctx.get<ImAHoop>(hoop) = ImAHoop{};
            ctx.get<ScoringZone>(hoop) = ScoringZone 
            {
                .3f, // Radius of scoring zone (1 meter)
                2.0f, // Height of scoring zone (2 meters)
                Vector3{hoop_pos.x, hoop_pos.y, hoop_pos.z} // Center of the scoring zone
            };
            

            // Keep random movement commented out
            // ctx.get<RandomMovement>(basketball) = RandomMovement {
            //     0.f,
            //     1.f + i * 2.f  // Different movement intervals: 1s, 3s, 5s...
            // };
        }

    }
}
