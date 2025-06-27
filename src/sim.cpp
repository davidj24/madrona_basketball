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


    // ================================================= Archetypes ================================================= 
    registry.registerArchetype<Agent>();
    registry.registerArchetype<Basketball>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::AgentPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);

    registry.exportColumn<Basketball, GridPos>((uint32_t)ExportID::BasketballPos);
    
}


//=================================================== Systems ===================================================

inline void moveBall(Engine &ctx,
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


// Removed complex grabBall and moveBallWithAgent systems
// Grab functionality is now handled directly in the tick system





// =================================== Tick System =========================================
inline void tick(Engine &ctx,
                 Entity agent_entity,
                 Action &action,
                 Reset &reset,
                 GridPos &grid_pos,
                 Reward &reward,
                 Done &done,
                 CurStep &episode_step,
                 InPossession &in_possession)
{
    const GridState *grid = ctx.data().grid;

    GridPos new_pos = grid_pos;

    switch (action) 
    {
        case Action::Up: {new_pos.y += 1;} break;
        case Action::Down: {new_pos.y -= 1;} break;
        case Action::Left: {new_pos.x -= 1;} break;
        case Action::Right: {new_pos.x += 1;} break;
        case Action::Grab: {
            // If agent already has a ball, drop it
            if (in_possession.hasBall) {
                auto basketball_query = ctx.query<GridPos, Grabbed>();
                ctx.iterateQuery(basketball_query, [&](GridPos &basketball_pos, Grabbed &grabbed) 
                {
                    if (grabbed.isGrabbed && grabbed.holderEntityID == (uint32_t)agent_entity.id) 
                    {
                        // Let the ball go if I'm the one holding it
                        in_possession.hasBall = false;
                        grabbed.isGrabbed = false;
                        grabbed.holderEntityID = 0;
                        in_possession.ballEntityID = 0;
                    }
                });
            }
            // Otherwise, try to grab a ball at current position
            else {
                bool ball_grabbed = false;  // Flag to ensure only one ball is grabbed
                auto basketball_query = ctx.query<GridPos, Grabbed>();
                ctx.iterateQuery(basketball_query, [&](GridPos &basketball_pos, Grabbed &grabbed) {
                    // Only grab if basketball is at same position as agent and not already grabbed
                    if (!ball_grabbed && basketball_pos.x == grid_pos.x && basketball_pos.y == grid_pos.y && !grabbed.isGrabbed) 
                    {
                        
                        in_possession.hasBall = true;
                        grabbed.isGrabbed = true;
                        grabbed.holderEntityID = (uint32_t)agent_entity.id;
                        ball_grabbed = true;  // Prevent grabbing multiple balls
                    }
                });
            }
            // Clear the grab action immediately to prevent repeated execution
            action = Action::None;
        } break;
        default: break;
    }

    

    if (new_pos.x < 0) {new_pos.x = 0;}
    if (new_pos.x >= grid->width) {new_pos.x = grid->width - 1;}
    if (new_pos.y < 0) {new_pos.y = 0;}
    if (new_pos.y >= grid->height) {new_pos.y = grid->height -1;}

    {
        const Cell &new_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

        if ((new_cell.flags & CellFlag::Wall)) {new_pos = grid_pos;}
    }

    // Move basketball with agent if holding one
    if (in_possession.hasBall) {
        auto basketball_query = ctx.query<GridPos, Grabbed>();
        ctx.iterateQuery(basketball_query, [&](GridPos &basketball_pos, Grabbed &grabbed) {
            if (grabbed.isGrabbed && grabbed.holderEntityID == (uint32_t)agent_entity.id) {
                basketball_pos = new_pos;  // Move basketball to agent's new position
            }
        });
    }

    const Cell &cur_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

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
        auto basketball_query = ctx.query<GridPos, Grabbed>();
        ctx.iterateQuery(basketball_query, [&](GridPos &basketball_pos, Grabbed &grabbed) {
            if (grabbed.isGrabbed && grabbed.holderEntityID == (uint32_t)agent_entity.id) {
                grabbed.isGrabbed = false;
                grabbed.holderEntityID = 0;
                // Reset basketball to start position
                basketball_pos = GridPos {
                    grid->startX,
                    grid->startY,
                    0
                };
            }
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
    
    // Clear movement actions after processing (grab actions are cleared immediately)
    if (action != Action::Grab) {
        action = Action::None;
    }

    reward.r = cur_cell.reward;
}









// =================================================== Task Graph ===================================================
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);
    builder.addToGraph<ParallelForNode<Engine, tick,
        Entity, Action, Reset, GridPos, Reward, Done, CurStep, InPossession>>({});
    
    // Keep ball movement commented out since RandomMovement is not in Basketball archetype
    // builder.addToGraph<ParallelForNode<Engine, moveBall,
    //     GridPos, RandomMovement>>({});

    // Removed complex grab systems - grab logic is now in tick system
    // builder.addToGraph<ParallelForNode<Engine, grabBall,
    //     Entity, Action, GridPos, InPossession>>({});
    // builder.addToGraph<ParallelForNode<Engine, moveBallWithAgent,
    //     Entity, GridPos, Grabbed>>({});
}



// =================================================== Sim Creation ===================================================
Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    Entity agent = ctx.makeEntity<Agent>();
    ctx.get<Action>(agent) = Action::None;
    ctx.get<GridPos>(agent) = GridPos {
        (grid->startX - 5),
        grid->startY,
        0
    };
    ctx.get<Reward>(agent).r = 0.f;
    ctx.get<Done>(agent).episodeDone = 0.f;
    ctx.get<CurStep>(agent).step = 0;
    ctx.get<InPossession>(agent) = {false, 0};


    Entity agent2 = ctx.makeEntity<Agent>();
    ctx.get<Action>(agent2) = Action::None;
    ctx.get<GridPos>(agent2) = GridPos {
        (grid->startX + 5),
        grid->startY,
        0
    };
    ctx.get<Reward>(agent2).r = 0.f;
    ctx.get<Done>(agent2).episodeDone = 0.f;
    ctx.get<CurStep>(agent2).step = 0;
    ctx.get<InPossession>(agent2) = {false, 0};
    

    for (int i = 0; i < NUM_BASKETBALLS; i++) {
        Entity basketball = ctx.makeEntity<Basketball>();
        ctx.get<GridPos>(basketball) = GridPos {
            grid->startX,   
            grid->startY,  
            0
        };

        ctx.get<Grabbed>(basketball) = Grabbed {false, 0};

        // Keep random movement commented out as requested
        // ctx.get<RandomMovement>(basketball) = RandomMovement {
        //     0.f,
        //     1.f + i * 2.f  // Different movement intervals: 1s, 3s, 5s...
        // };
    }
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
