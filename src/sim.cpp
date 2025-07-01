#include "sim.hpp"
#include "types.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath> // For acosf


using namespace madrona;
using namespace madrona::math;



// This helper function computes the rotation needed to align the 'start' vector with the 'target' vector.
inline Quat findRotationBetweenVectors(Vector3 start, Vector3 target) {
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
inline void processGrab(Engine &ctx,
                        Entity agent_entity,
                        Action &action,
                        Position &agent_pos,
                        InPossession &in_possession,
                        Team &team)
{
    GameState &gameState = ctx.singleton<GameState>();
    auto basketball_query = ctx.query<Entity, Position, Grabbed, BallPhysics>();
    ctx.iterateQuery(basketball_query, [&](Entity ball_entity, Position &basketball_pos, Grabbed &grabbed, BallPhysics &ball_physics) 
    {
        if (action.grab == 0) {return;}

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

        // Check if ball is within grab distance (0.5 meters)
        float distance = sqrt((basketball_pos.x - agent_pos.x) * (basketball_pos.x - agent_pos.x) +
                             (basketball_pos.y - agent_pos.y) * (basketball_pos.y - agent_pos.y));
        
        if (distance <= 0.5f) // 0.5 meter grab radius
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
            gameState.teamInPossession = team.teamIndex; // Update the team in possession
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
            ball_physics.velocity = agent_orientation.orientation.rotateVec(Vector3{0, 1, 0}); // Setting the ball's velocity to have the same direction as the agent's orientation
                                                                                               // Note: we use 0, 2, 0 because that's forward in our simulation specifically
            ball_physics.inFlight = true;
            gameState.inboundingInProgress = false;
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
    ctx.iterateQuery(hoop_query, [&](Entity hoop_entity, Position &hoop_pos, ScoringZone &scoring_zone) {
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

    // Calculate intended angle
    float intended_direction = std::atan2(shot_vector.x, shot_vector.y); // This points in the direction of the hoop

    float direction_deviation_per_meter = 0.1f; // radians per meter distance
    float stddev = direction_deviation_per_meter * distance_to_hoop;
    static thread_local std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, stddev);
    float direction_deviation = dist(rng);
    float shot_direction = intended_direction + direction_deviation;

    // This is the final, correct trajectory vector for the ball
    Vector3 final_shot_vec = Vector3{std::sin(shot_direction), std::cos(shot_direction), 0.f};

    // --- Set Agent Orientation using the new method ---

    // 1. Define the agent's base "forward" direction to match your working passSystem.
    //    We use a unit vector for the direction.
    const Vector3 base_forward = {0.0f, 1.0f, 0.0f};

    // 2. Find the rotation that aligns the agent's "forward" direction
    //    with the final shot direction vector.
    agent_orientation.orientation = findRotationBetweenVectors(base_forward, final_shot_vec);

    // --- Release the ball (This part remains the same) ---
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
            ball_physics.velocity = final_shot_vec * 5.0f; // 5 m/s shot speed
            ball_physics.inFlight = true;
        }
    });
}


inline void moveAgentSystem(Engine &ctx,
                           Action &action,
                           Position &agent_pos, // Note: This should now store floats
                           Orientation &agent_orientation)
{
    // Define the duration of a single simulation step.
    // For example, if your simulation runs at 30 steps per second.
    const float delta_time = 1.0f / 60.0f;

    const GridState *grid = ctx.data().grid;
    if (action.rotate != 0)
    {
        // Rotation logic is fine as it is
        float turn_angle = (pi/180.f) * action.rotate * 4;
        Quat turn = Quat::angleAxis(turn_angle, Vector3{0, 0, 1});
        agent_orientation.orientation = turn * agent_orientation.orientation;
    }

    if (action.moveSpeed > 0)
    {
        // Treat moveSpeed as a velocity in meters/second, not a distance.
        // Let's say a moveSpeed of 1 corresponds to 1 m/s.
        float agent_velocity_magnitude = action.moveSpeed * 7;

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


//=================================================== General Systems ===================================================
inline void tick(Engine &ctx,
                 Entity agent_entity,
                 Reset &reset,
                 Position &position,
                 Reward &reward, //add later
                 Done &done,
                 CurStep &episode_step,
                 InPossession &in_possession,
                 Inbounding &inbounding)
{
    const GridState *grid = ctx.data().grid;

    Position new_pos = position;

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

        // Reset possession state
        in_possession.hasBall = false;
        inbounding.imInbounding = false;

        // Reset all basketballs when episode ends
        auto basketball_query = ctx.query<Position, Grabbed, BallPhysics>();
        ctx.iterateQuery(basketball_query, [&](Position &basketball_pos, Grabbed &grabbed, BallPhysics &ball_physics) 
        {
            grabbed.isGrabbed = false;
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER;
            ball_physics.inFlight = false;
            ball_physics.velocity = Vector3::zero();
            // Reset basketball to start position
            basketball_pos = Position 
            {
                grid->startX,
                grid->startY,
                0.f
            };
            ball_physics.lastTouchedByID = ENTITY_ID_PLACEHOLDER;
        });

        new_pos = Position {
            grid->startX,
            grid->startY,
            0.f
        };

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
    const GridState *grid = ctx.data().grid;
    GameState &gameState = ctx.singleton<GameState>();
    

    // Check if the ball is out of bounds (1 meter sideline)
    constexpr float COURT_SIDELINE_WIDTH = 1.0f; // 1 meter sideline
    if (ball_pos.x < COURT_SIDELINE_WIDTH || ball_pos.x >= grid->width - COURT_SIDELINE_WIDTH ||
        ball_pos.y < COURT_SIDELINE_WIDTH || ball_pos.y >= grid->height - COURT_SIDELINE_WIDTH)
        // ball_pos.z < 0 || ball_pos.z >= grid->depth) for when we're in 3D later
    {
        // Reset the ball physics
        ball_physics.inFlight = false;
        ball_physics.velocity = Vector3::zero();

        auto agent_query = ctx.query<Entity, Team, InPossession, Position, Inbounding>();
        ctx.iterateQuery(agent_query, [&] (Entity agent_entity, Team &agent_team, InPossession &in_possession, Position &agent_pos, Inbounding &inbounding)
        {
            // If an agent has the ball, we need to reset their position
            if (in_possession.ballEntityID == ball_entity.id && agent_team.teamIndex == ball_physics.lastTouchedByID && inbounding.imInbounding == false)
            {
                agent_pos = Position {
                    grid->startX,
                    grid->startY,
                    0.f
                };
                in_possession.hasBall = false;
                in_possession.ballEntityID = ENTITY_ID_PLACEHOLDER;
            }

            if (agent_team.teamIndex != ball_physics.lastTouchedByID && !gameState.inboundingInProgress)
            {
                inbounding.imInbounding = true;
                gameState.inboundingInProgress = true;
                agent_pos = ball_pos;
                grabbed.isGrabbed = true;
                grabbed.holderEntityID = agent_entity.id;
                in_possession.hasBall = true;
                in_possession.ballEntityID = ball_entity.id;
                gameState.teamInPossession = agent_team.teamIndex;
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
        Action, Position, Orientation>>({});

    auto tickNode = builder.addToGraph<ParallelForNode<Engine, tick,
        Entity, Reset, Position, Reward, Done, CurStep, InPossession, Inbounding>>({});
    
    // builder.addToGraph<ParallelForNode<Engine, moveBallRandomly,
    //     Position, RandomMovement>>({});

    auto processGrabNode = builder.addToGraph<ParallelForNode<Engine, processGrab,
        Entity, Action, Position, InPossession, Team>>({});

    auto passSystemNode = builder.addToGraph<ParallelForNode<Engine, passSystem,
        Entity, Action, Orientation, InPossession, Inbounding>>({});

    auto moveBallSystemNode = builder.addToGraph<ParallelForNode<Engine, moveBallSystem,
        Position, BallPhysics, Grabbed>>({processGrabNode});

    auto outOfBoundsSystemNode = builder.addToGraph<ParallelForNode<Engine, outOfBoundsSystem,
        Entity, Position, Grabbed, BallPhysics>>({passSystemNode, moveBallSystemNode});

    auto updateLastTouchSystemNode = builder.addToGraph<ParallelForNode<Engine, updateLastTouchSystem,
        Position, BallPhysics>>({});

    auto shootSystemNode = builder.addToGraph<ParallelForNode<Engine, shootSystem,
        Entity, Action, Position, Orientation, Inbounding, InPossession, Team>>({});
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
        .inboundingInProgress = false,
        .liveBall = true,
        .period = 1,
        .teamInPossession = 0,
        .team0Score = 0,
        .team1Score = 0,
        .gameClock = 720.0f,
        .shotClock = 24.0f
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


    for (int i = 0; i < NUM_HOOPS; i++) 
    {
        Entity hoop = ctx.makeEntity<Hoop>();
        Position hoop_pos;
        if (i == 0) 
        {
            // Left hoop (3 meters from left edge, center court)
            hoop_pos = Position { 3.0f, grid->height / 2.0f, 0.f };
        } 
        else if (i == 1) 
        {
            // Right hoop (3 meters from right edge, center court)  
            hoop_pos = Position { grid->width - 3.0f, grid->height / 2.0f, 0.f };
        } 
        else 
        {
            // Additional hoops (if NUM_HOOPS > 2)
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
            1.0f, // Radius of scoring zone (1 meter)
            2.0f, // Height of scoring zone (2 meters)
            Vector3{hoop_pos.x, hoop_pos.y, hoop_pos.z} // Center of the scoring zone
        };
        

        // Keep random movement commented out as requested
        // ctx.get<RandomMovement>(basketball) = RandomMovement {
        //     0.f,
        //     1.f + i * 2.f  // Different movement intervals: 1s, 3s, 5s...
        // };
    }

}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
