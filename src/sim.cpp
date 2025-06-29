#include "sim.hpp"
#include "types.hpp"
#include <madrona/mw_gpu_entry.hpp>
#include <cstdlib>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

// ================================================== Components ==================================================
void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<GridPos>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();
    registry.registerComponent<RandomMovement>();
    registry.registerComponent<InPossession>();
    registry.registerComponent<Grabbed>();
    registry.registerComponent<Orientation>();
    registry.registerComponent<BallPhysics>();


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

    registry.exportColumn<Basketball, GridPos>((uint32_t)ExportID::BasketballPos);
    registry.exportColumn<Basketball, BallPhysics>((uint32_t)ExportID::BallPhysicsData);
    registry.exportColumn<Basketball, Grabbed>((uint32_t)ExportID::BallGrabbed);

    registry.exportColumn<Hoop, GridPos>((uint32_t)ExportID::HoopPos);
    
    // Export entity IDs for debugging
    registry.exportColumn<Agent, madrona::Entity>((uint32_t)ExportID::AgentEntityID);
    registry.exportColumn<Basketball, madrona::Entity>((uint32_t)ExportID::BallEntityID);
    
}


//=================================================== Systems ===================================================

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


inline void processGrab(Engine &ctx,
                        Entity agent_entity,
                        Action &action,
                        GridPos &agent_pos,
                        InPossession &in_possession)
{

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
            in_possession.ballEntityID = INVALID_ENTITY_ID;
            in_possession.hasBall = false;
            grabbed.holderEntityID = INVALID_ENTITY_ID;
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
                    other_in_possession.ballEntityID = INVALID_ENTITY_ID;
                }
            });

            in_possession.hasBall = true;
            in_possession.ballEntityID = ball_entity.id;
            grabbed.holderEntityID = (uint32_t)agent_entity.id;
            grabbed.isGrabbed = true;
            ball_physics.in_flight = false; // Make it so the ball isn't "in flight" anymore
            ball_physics.velocity = Vector3::zero(); // And change its velocity to be zero
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





inline void passSystem(Engine &ctx,
                       Entity agent_entity,
                       Action &action,
                       Orientation &agent_orientation,
                       InPossession &in_possession)
{
    if (action.pass == 0) {return;}

    auto held_ball_query = ctx.query<Grabbed, BallPhysics>();
    ctx.iterateQuery(held_ball_query, [&] (Grabbed &grabbed, BallPhysics &ball_physics)
    {
        if (grabbed.holderEntityID == agent_entity.id)
        {
            grabbed.isGrabbed = false;  // Ball is no longer grabbed
            grabbed.holderEntityID = INVALID_ENTITY_ID; // Ball is no longer held by anyone
            in_possession.hasBall = false; // Since agents can only hold 1 ball at a time, if they pass it they can't be holding one anymore
            in_possession.ballEntityID = INVALID_ENTITY_ID; // Whoever passed the ball is no longer in possession of it
            //Affect the ball's movement system here???
            // Don't forget to make the ball's velocity vector equal to the orientation of the agent
            ball_physics.velocity = agent_orientation.orientation.rotateVec(Vector3{0, 2, 0}); // Setting the ball's velocity to have the same direction as the agent's orientation
                                                                                               // Note: we use 0, 2, 0 because that's forward in our simulation specifically
            ball_physics.in_flight = true;
        }
    });
}


                        





// =================================== Tick System =========================================
inline void tick(Engine &ctx,
                 Entity agent_entity,
                 Reset &reset,
                 GridPos &grid_pos,
                 Reward &reward, //add later
                 Done &done,
                 CurStep &episode_step,
                 InPossession &in_possession)
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

        // Reset all basketballs when episode ends
        auto basketball_query = ctx.query<GridPos, Grabbed, BallPhysics>();
        ctx.iterateQuery(basketball_query, [&](GridPos &basketball_pos, Grabbed &grabbed, BallPhysics &ball_physics) 
        {
            grabbed.isGrabbed = false;
            grabbed.holderEntityID = INVALID_ENTITY_ID;
            ball_physics.in_flight = false;
            ball_physics.velocity = Vector3::zero();
            // Reset basketball to start position
            basketball_pos = GridPos 
            {
                grid->startX,
                grid->startY,
                0
            };
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







// =================================================== Task Graph ===================================================
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);

    auto moveAgentSystemNode = builder.addToGraph<ParallelForNode<Engine, moveAgentSystem,
        Action, GridPos, Orientation>>({});

    auto tickNode = builder.addToGraph<ParallelForNode<Engine, tick,
        Entity, Reset, GridPos, Reward, Done, CurStep, InPossession>>({});
    
    // builder.addToGraph<ParallelForNode<Engine, moveBallRandomly,
    //     GridPos, RandomMovement>>({});
    auto passSystemNode = builder.addToGraph<ParallelForNode<Engine, passSystem,
        Entity, Action, Orientation, InPossession>>({});

    auto processGrabNode = builder.addToGraph<ParallelForNode<Engine, processGrab,
        Entity, Action, GridPos, InPossession>>({});

    auto moveBallSystemNode = builder.addToGraph<ParallelForNode<Engine, moveBallSystem,
        GridPos, BallPhysics, Grabbed>>({passSystemNode, processGrabNode});
}





// =================================================== Sim Creation ===================================================
Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    for (int i = 0; i < NUM_AGENTS; i++) 
    {
        Entity agent = ctx.makeEntity<Agent>();
        ctx.get<Action>(agent) = Action{0, 0, 0, 0, 0}; // Initialize with no action
        ctx.get<GridPos>(agent) = GridPos 
        {
            (grid->startX + i - 5),
            grid->startY,
            0
        };
        ctx.get<Reset>(agent) = Reset{0}; // Initialize reset component
        ctx.get<Reward>(agent).r = 0.f;
        ctx.get<Done>(agent).episodeDone = 0.f;
        ctx.get<CurStep>(agent).step = 0;
        ctx.get<InPossession>(agent) = {false, INVALID_ENTITY_ID};
        ctx.get<Orientation>(agent) = Orientation {Quat::id()};
    };

    

    for (int i = 0; i < NUM_BASKETBALLS; i++) {
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
        ctx.get<Grabbed>(basketball) = Grabbed {false, INVALID_ENTITY_ID};
        ctx.get<BallPhysics>(basketball) = BallPhysics {false, Vector3::zero()};
        

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
        if (i == 0) {
            // Left hoop
            hoop_pos = GridPos { 3, 17, 0 };
        } else if (i == 1) {
            // Right hoop  
            hoop_pos = GridPos { 47, 17, 0 };
        } else {
            // Additional hoops (if NUM_HOOPS > 2)
            hoop_pos = GridPos { 
                grid->startX + 10 + i * 5,   
                grid->startY + 10,  
                0
            };
        }

        ctx.get<GridPos>(hoop) = hoop_pos;
        ctx.get<Reset>(hoop) = Reset{0};
        ctx.get<Done>(hoop).episodeDone = 0.f;
        ctx.get<CurStep>(hoop).step = 0;
        

        // Keep random movement commented out as requested
        // ctx.get<RandomMovement>(basketball) = RandomMovement {
        //     0.f,
        //     1.f + i * 2.f  // Different movement intervals: 1s, 3s, 5s...
        // };
    }

}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
