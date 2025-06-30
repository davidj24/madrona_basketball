#include "sim.hpp"
#include "types.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <cstdlib>
#include <vector>
#include <random>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    // ================================================== Singletons ==================================================
    registry.registerSingleton<GameState>();




    // ================================================== General Components ==================================================
    registry.registerComponent<Reset>();
    registry.registerComponent<GridPos>();
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
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::AgentPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
    registry.exportColumn<Agent, InPossession>((uint32_t)ExportID::AgentPossession);
    registry.exportColumn<Agent, Team>((uint32_t)ExportID::TeamData);
    registry.exportColumn<Agent, Orientation>((uint32_t)ExportID::Orientation);

    registry.exportColumn<Basketball, GridPos>((uint32_t)ExportID::BasketballPos);
    registry.exportColumn<Basketball, BallPhysics>((uint32_t)ExportID::BallPhysicsData);
    registry.exportColumn<Basketball, Grabbed>((uint32_t)ExportID::BallGrabbed);

    registry.exportColumn<Hoop, GridPos>((uint32_t)ExportID::HoopPos);

    // Singleton exports
    registry.exportSingleton<GameState>((uint32_t)ExportID::GameState);
    
    // Export entity IDs for debugging
    registry.exportColumn<Agent, madrona::Entity>((uint32_t)ExportID::AgentEntityID);
    registry.exportColumn<Basketball, madrona::Entity>((uint32_t)ExportID::BallEntityID);
    
}




//=================================================== Ball Systems ===================================================
inline void moveBallRandomly(Engine &ctx,
                    GridPos &ball_pos,
                    RandomMovement &random_movement)
{
    random_movement.moveTimer ++;
    if (random_movement.moveTimer >= random_movement.moveInterval) 
    {
        random_movement.moveTimer = 0.f;
        const GridState *grid = ctx.data().grid;

        int dx = (rand() % 3) - 1; // -1, 0, or 1
        int dy = (rand() % 3) - 1; // -1, 0, or 1

        int32_t new_x = ball_pos.x + dx;
        int32_t new_y = ball_pos.y + dy;

        new_x = std::clamp(new_x, 0, grid->width - 1);
        new_y = std::clamp(new_y, 0, grid->height - 1);

        ball_pos.x = new_x;
        ball_pos.y = new_y;
    } 
}



inline void moveBallSystem(Engine &ctx,
                           GridPos &ball_pos,
                           BallPhysics &ball_physics,
                           Grabbed &grabbed)
{
    auto holder_query = ctx.query<Entity, GridPos, InPossession>();
    ctx.iterateQuery(holder_query, [&](Entity &agent_entity, GridPos &agent_pos, InPossession &in_possession)
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
    int32_t new_x = ball_pos.x + (int32_t)ball_physics.velocity[0];
    int32_t new_y = ball_pos.y + (int32_t)ball_physics.velocity[1];
    int32_t new_z = ball_pos.z + (int32_t)ball_physics.velocity[2];

    new_x = std::clamp(new_x, 0, (int32_t)grid->width - 1);
    new_y = std::clamp(new_y, 0, (int32_t)grid->height - 1);
    // new_z = std::clamp(new_z, 0, grid->depth - 1);
    
    // Wall collision (if needed)
    GridPos test_pos = {new_x, new_y, new_z};
    const Cell &new_cell = grid->cells[test_pos.y * grid->width + test_pos.x];
    
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
                        GridPos &agent_pos,
                        InPossession &in_possession,
                        Team &team)
{
    GameState &gameState = ctx.singleton<GameState>();
    auto basketball_query = ctx.query<Entity, GridPos, Grabbed, BallPhysics>();
    ctx.iterateQuery(basketball_query, [&](Entity ball_entity, GridPos &basketball_pos, Grabbed &grabbed, BallPhysics &ball_physics) 
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

        // Otherwise, try to grab a ball at current position
        if (basketball_pos.x == agent_pos.x && basketball_pos.y == agent_pos.y) 
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
            ball_physics.velocity = agent_orientation.orientation.rotateVec(Vector3{0, 2, 0}); // Setting the ball's velocity to have the same direction as the agent's orientation
                                                                                               // Note: we use 0, 2, 0 because that's forward in our simulation specifically
            ball_physics.inFlight = true;
            gameState.inboundingInProgress = false;
        }
    });
}


inline void shootSystem(Engine &ctx,
                        Entity agent_entity,
                        Action &action,
                        GridPos agent_pos,
                        Orientation &agent_orientation,
                        InPossession &in_possession,
                        Inbounding &inbounding,
                        Team &team)
{
    if (action.shoot == 0 || !in_possession.hasBall) {return;}

    // Find the attacking hoop (not defendingHoopID)
    auto hoop_query = ctx.query<Entity, GridPos, ScoringZone>();
    GridPos attacking_hoop_pos = {0, 0, 0};
    ctx.iterateQuery(hoop_query, [&](Entity hoop_entity, GridPos &hoop_pos, ScoringZone &scoring_zone) {
        if ((uint32_t)hoop_entity.id != team.defendingHoopID) 
        {
            attacking_hoop_pos = hoop_pos;
            return;
        }
    });

    // Calculate vector to attacking hoop
    Vector3 hoop_vec = Vector3{
        static_cast<float>(attacking_hoop_pos.x - agent_pos.x),
        static_cast<float>(attacking_hoop_pos.y - agent_pos.y),
        0.f
    };

    Vector3 shot_vector = Vector3{
        static_cast<float>(agent_pos.x - attacking_hoop_pos.x),
        static_cast<float>(agent_pos.y - attacking_hoop_pos.y),
        0.f
    };

    float distance_to_hoop = shot_vector.length();

    // Calculate intended angle
    float intended_direction = std::atan2(shot_vector.x, shot_vector.y); // This points in the direction of the hoop

    // Add random deviation based on distance
    float direction_deviation_per_meter = 0.0f; // radians per unit distance (tune as desired)
    float stddev = direction_deviation_per_meter * distance_to_hoop;
    static thread_local std::mt19937 rng(std::random_device{}()); // Creates random number generator
    std::normal_distribution<float> dist(0.0f, stddev); // Creates the normal distribution
    float direction_deviation = dist(rng); // Samples from distribution to get the deviation
    float shot_direction = intended_direction + direction_deviation;

    // Set agent orientation with deviation
    agent_orientation.orientation = Quat::angleAxis(shot_direction, Vector3{0, 2, 0});

    // Shoot the ball in the (possibly deviated) direction
    Vector3 final_shot_vec = Vector3{std::sin(shot_direction), std::cos(shot_direction), 0.f};
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
            ball_physics.velocity = final_shot_vec * 2.0f; // 2.0f matches passSystem's speed
            ball_physics.inFlight = true;
        }
    });
}


inline void moveAgentSystem(Engine &ctx,
                           Action &action,
                           GridPos &agent_pos,
                           Orientation &agent_orientation)
{
    const GridState *grid = ctx.data().grid;
    if (action.rotate != 0)
    {
        float turn_angle = (pi/4.f) * action.rotate;
        Quat turn = Quat::angleAxis(turn_angle, Vector3{0, 0, 1});
        agent_orientation.orientation = turn * agent_orientation.orientation;
    }

    if (action.moveSpeed > 0)
    {
        constexpr float angle_between_directions = pi / 4.f;
        float move_angle = action.moveAngle * angle_between_directions;

        int32_t dx = (int32_t)std::round(std::sin(move_angle) * action.moveSpeed);
        int32_t dy = (int32_t)std::round(-std::cos(move_angle) * action.moveSpeed);

        int32_t new_x = agent_pos.x + dx;  
        int32_t new_y = agent_pos.y + dy;  

        // Boundary checking
        new_x = std::clamp(new_x, 0, grid->width - 1);
        new_y = std::clamp(new_y, 0, grid->height - 1);

        // Wall collision (if needed)
        GridPos test_pos = {new_x, new_y, agent_pos.z};
        const Cell &new_cell = grid->cells[test_pos.y * grid->width + test_pos.x];
        
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
                 GridPos &grid_pos,
                 Reward &reward, //add later
                 Done &done,
                 CurStep &episode_step,
                 InPossession &in_possession,
                 Inbounding &inbounding)
{
    const GridState *grid = ctx.data().grid;

    GridPos new_pos = grid_pos;

    bool episode_done = false;
    if (reset.resetNow != 0) 
    {
        reset.resetNow = 0;
        episode_done = true;
    }

    // if ((cur_cell.flags & CellFlag::End)) {
    //     episode_done = true;
    // }

    uint32_t cur_step = episode_step.step;

    if (cur_step == ctx.data().maxEpisodeLength - 1) {episode_done = true;}

    if (episode_done) 
    {
        done.episodeDone = 1.f;

        // Reset possession state
        in_possession.hasBall = false;
        inbounding.imInbounding = false;


        // Reset all basketballs when episode ends
        auto basketball_query = ctx.query<GridPos, Grabbed, BallPhysics>();
        ctx.iterateQuery(basketball_query, [&](GridPos &basketball_pos, Grabbed &grabbed, BallPhysics &ball_physics) 
        {
            grabbed.isGrabbed = false;
            grabbed.holderEntityID = ENTITY_ID_PLACEHOLDER;
            ball_physics.inFlight = false;
            ball_physics.velocity = Vector3::zero();
            // Reset basketball to start position
            basketball_pos = GridPos 
            {
                grid->startX,
                grid->startY,
                0
            };
            ball_physics.lastTouchedByID = ENTITY_ID_PLACEHOLDER;
        });

        new_pos = GridPos {
            grid->startX,
            grid->startY,
            0
        };

        episode_step.step = 0;
    } 
    else 
    {
        done.episodeDone = 0.f;
        episode_step.step = cur_step + 1;
    }

    grid_pos = new_pos;

    // Calculate reward based on current position
    const Cell &cur_cell = grid->cells[grid_pos.y * grid->width + grid_pos.x];
    reward.r = cur_cell.reward;
}



inline void updateLastTouchSystem(Engine &ctx,
                                  GridPos &ball_pos,
                                  BallPhysics &ball_physics)
{
    auto touched_agent_query = ctx.query<GridPos, Team>();
    ctx.iterateQuery(touched_agent_query, [&] (GridPos &agent_pos, Team &team)
    {
        if (ball_pos.x == agent_pos.x && ball_pos.y == agent_pos.y && ball_pos.z == agent_pos.z) 
        {
            ball_physics.lastTouchedByID = (uint32_t)team.teamIndex;
        }
    });
}



inline void outOfBoundsSystem(Engine &ctx,
                              Entity ball_entity,
                              GridPos &ball_pos,
                              Grabbed &grabbed,
                              BallPhysics &ball_physics)
{
    const GridState *grid = ctx.data().grid;
    GameState &gameState = ctx.singleton<GameState>();
    

    // Check if the ball is out of bounds
    constexpr int COURT_SIDELINE_LENGTH = 1;
    if (ball_pos.x < COURT_SIDELINE_LENGTH || ball_pos.x >= grid->width - COURT_SIDELINE_LENGTH ||
        ball_pos.y < COURT_SIDELINE_LENGTH || ball_pos.y >= grid->height - COURT_SIDELINE_LENGTH)
        // ball_pos.z < 0 || ball_pos.z >= grid->depth) for when we're in 3D later
    {
        // Reset the ball physics
        ball_physics.inFlight = false;
        ball_physics.velocity = Vector3::zero();

        auto agent_query = ctx.query<Entity, Team, InPossession, GridPos, Inbounding>();
        ctx.iterateQuery(agent_query, [&] (Entity agent_entity, Team &agent_team, InPossession &in_possession, GridPos &agent_pos, Inbounding &inbounding)
        {
            // If an agent has the ball, we need to reset their position
            if (in_possession.ballEntityID == ball_entity.id && agent_team.teamIndex == ball_physics.lastTouchedByID && inbounding.imInbounding == false)
            {
                agent_pos = GridPos {
                    grid->startX,
                    grid->startY,
                    0
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
        Action, GridPos, Orientation>>({});

    auto tickNode = builder.addToGraph<ParallelForNode<Engine, tick,
        Entity, Reset, GridPos, Reward, Done, CurStep, InPossession, Inbounding>>({});
    
    // builder.addToGraph<ParallelForNode<Engine, moveBallRandomly,
    //     GridPos, RandomMovement>>({});

    auto processGrabNode = builder.addToGraph<ParallelForNode<Engine, processGrab,
        Entity, Action, GridPos, InPossession, Team>>({});

    auto passSystemNode = builder.addToGraph<ParallelForNode<Engine, passSystem,
        Entity, Action, Orientation, InPossession, Inbounding>>({});

    auto moveBallSystemNode = builder.addToGraph<ParallelForNode<Engine, moveBallSystem,
        GridPos, BallPhysics, Grabbed>>({processGrabNode});

    auto outOfBoundsSystemNode = builder.addToGraph<ParallelForNode<Engine, outOfBoundsSystem,
        Entity, GridPos, Grabbed, BallPhysics>>({passSystemNode, moveBallSystemNode});

    auto updateLastTouchSystemNode = builder.addToGraph<ParallelForNode<Engine, updateLastTouchSystem,
        GridPos, BallPhysics>>({});

    auto shootSystemNode = builder.addToGraph<ParallelForNode<Engine, shootSystem,
        Entity, Action, GridPos, Orientation, InPossession, Inbounding, Team>>({});
}

// =================================================== Sim Creation ===================================================

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    ctx.singleton<GameState>() = GameState // CHANGE LATER TO INCLUDE OTHER ATTRIBUTES
    {
        .inboundingInProgress = false,
        .period = 1,
        .gameClock = 720,
        .shotClock = 24
    };

    std::vector<Vector3> team_colors = {Vector3{0, 100, 255}, Vector3{255, 0, 100}};
    for (int i = 0; i < NUM_AGENTS; i++) 
    {
        Entity agent = ctx.makeEntity<Agent>();
        ctx.get<Action>(agent) = Action{0, 0, 0, 0, 0, 0}; // Initialize with no action
        ctx.get<GridPos>(agent) = GridPos 
        {
            (grid->startX + i - 5),
            grid->startY,
            0
        };
        ctx.get<Reset>(agent) = Reset{0}; // Initialize reset component
        ctx.get<Inbounding>(agent) = {false};
        ctx.get<Reward>(agent).r = 0.f;
        ctx.get<Done>(agent).episodeDone = 0.f;
        ctx.get<CurStep>(agent).step = 0;
        ctx.get<InPossession>(agent) = {false, ENTITY_ID_PLACEHOLDER};
        ctx.get<Orientation>(agent) = Orientation {Quat::id()};
        ctx.get<Team>(agent) = {i % 2, team_colors[i % 2]}, i % 2; // Alternates agent teams and colors
    };

    

    for (int i = 0; i < NUM_BASKETBALLS; i++) 
    {
        Entity basketball = ctx.makeEntity<Basketball>();
        ctx.get<GridPos>(basketball) = GridPos 
        {
            grid->startX,   
            grid->startY,  
            0
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
        GridPos hoop_pos;
        if (i == 0) 
        {
            // Left hoop
            hoop_pos = GridPos { 3, 17, 0 };
        } 
        else if (i == 1) 
        {
            // Right hoop  
            hoop_pos = GridPos { 47, 17, 0 };
        } 
        else 
        {
            // Additional hoops (if NUM_HOOPS > 2)
            hoop_pos = GridPos 
            { 
                grid->startX + 10 + i * 5,   
                grid->startY + 10,  
                0
            };
        }

        ctx.get<GridPos>(hoop) = hoop_pos;
        ctx.get<Reset>(hoop) = Reset{0};
        ctx.get<Done>(hoop).episodeDone = 0.f;
        ctx.get<CurStep>(hoop).step = 0;
        ctx.get<ImAHoop>(hoop) = ImAHoop{};
        ctx.get<ScoringZone>(hoop) = ScoringZone 
        {
            1.f, // Radius of scoring zone
            2.f, // Height of scoring zone
            Vector3{static_cast<float>(hoop_pos.x), static_cast<float>(hoop_pos.y), static_cast<float>(hoop_pos.z)} // Center of the scoring zone
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
