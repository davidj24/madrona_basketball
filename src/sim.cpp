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

    // General Components
    registry.registerComponent<Reset>();
    registry.registerComponent<Position>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();
    registry.registerComponent<RandomMovement>();
    registry.registerComponent<IsWorldClock>();

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
    registry.registerArchetype<WorldClock>();

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
    ballQuery = ctx.query<Entity, Position, Grabbed, BallPhysics>();
    hoopQuery = ctx.query<Entity, Position, ImAHoop>();
    agentQuery = ctx.query<Entity, Team, InPossession, Position, Orientation, Inbounding, GrabCooldown>();

    // Generate the world - defined in gen.cpp
    generateWorld(ctx);
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);

    auto game_step = setupGameStepTasks(builder, {});

}

}
