#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "types.hpp"
#include "gen.hpp"
#include "game.hpp"

using namespace madrona;
using namespace madrona::math;



namespace madBasketball {
void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    // Singletons
    registry.registerSingleton<GameState>();
    registry.registerSingleton<WorldClock>();

    // General Components
    registry.registerComponent<Reset>();
    registry.registerComponent<Position>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();
    registry.registerComponent<RandomMovement>();

    // Agent Components
    registry.registerComponent<Action>();
    registry.registerComponent<Observations>();
    registry.registerComponent<ActionMask>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Inbounding>();
    registry.registerComponent<InPossession>();
    registry.registerComponent<Orientation>();
    registry.registerComponent<Team>();
    registry.registerComponent<GrabCooldown>();
    registry.registerComponent<Stats>();
    registry.registerComponent<Attributes>();

    // Ball Components
    registry.registerComponent<BallPhysics>();
    registry.registerComponent<Grabbed>();

    // Hoop Components
    registry.registerComponent<ImAHoop>();
    registry.registerComponent<ScoringZone>();

    // Archetypes
    registry.registerArchetype<Agent>();
    registry.registerArchetype<Basketball>();
    registry.registerArchetype<Hoop>();

    // Tensor Exports
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, Observations>((uint32_t)ExportID::Observations);
    registry.exportColumn<Agent, ActionMask>((uint32_t)ExportID::ActionMask);
    registry.exportColumn<Agent, Position>((uint32_t)ExportID::AgentPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
    registry.exportColumn<Agent, InPossession>((uint32_t)ExportID::AgentPossession);
    registry.exportColumn<Agent, Team>((uint32_t)ExportID::TeamData);
    registry.exportColumn<Agent, Orientation>((uint32_t)ExportID::Orientation);
    registry.exportColumn<Agent, Stats>((uint32_t)ExportID::AgentStats);

    registry.exportColumn<Basketball, Position>((uint32_t)ExportID::BasketballPos);
    registry.exportColumn<Basketball, BallPhysics>((uint32_t)ExportID::BallPhysicsData);
    registry.exportColumn<Basketball, Grabbed>((uint32_t)ExportID::BallGrabbed);

    registry.exportColumn<Hoop, Position>((uint32_t)ExportID::HoopPos);

    // Singleton exports
    registry.exportSingleton<GameState>((uint32_t)ExportID::GameState);

    // Export entity IDs for debugging
    registry.exportColumn<Agent, Entity>((uint32_t)ExportID::AgentEntityID);
    registry.exportColumn<Basketball, Entity>((uint32_t)ExportID::BallEntityID);
}


Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
    initRandKey(cfg.initRandKey),
    rng(rand::split_i(ctx.data().initRandKey, 0, 0)),
    episodeMgr(init.episodeMgr),
    grid(init.grid),
    maxEpisodeLength(cfg.maxEpisodeLength)
{
    // Generate the world - defined in gen.cpp
    generateWorld(ctx);
}

// On GPU, we need to sort by world any entities whose data is exported
#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraphNodeID queueSortByWorld(TaskGraphBuilder &builder,
                                 Span<const TaskGraphNodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}

static TaskGraphNodeID sortEntities(TaskGraphBuilder &builder,
                                    Span<const TaskGraphNodeID> deps)
{
    auto sort_sys = queueSortByWorld<Agent>(
        builder, deps);
    sort_sys = queueSortByWorld<Basketball>(
        builder, {sort_sys});
    sort_sys = queueSortByWorld<Hoop>(
        builder, {sort_sys});
    return sort_sys;
}
#endif


void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);

    auto cur_node = setupGameStepTasks(builder, {});

#ifdef MADRONA_GPU_MODE
    cur_node = sortEntities(builder, {cur_node});
#endif

}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
