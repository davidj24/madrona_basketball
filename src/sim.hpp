#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/rand.hpp>
#include <madrona/custom_context.hpp>

#include "types.hpp"
#include "init.hpp"

using namespace madrona;

namespace madBasketball {

class Engine;

struct Sim : public WorldBase {
    struct Config {
        RandKey initRandKey;
        uint32_t maxEpisodeLength;
        bool enableViewer;
    };

    static auto registerTypes(ECSRegistry& registry,
                              const Config& cfg) -> void;

    static void setupTasks(TaskGraphManager &taskgraph_mgr,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    RandKey initRandKey;
    RNG rng;

    EpisodeManager *episodeMgr;
    const GridState *grid;
    uint32_t maxEpisodeLength;

    // Queries for entities
    Query<Entity, Position, Grabbed, BallPhysics> ballQuery;
    Query<Entity, Position, ImAHoop> hoopQuery;
    Query<Entity, Team, InPossession, Position, Orientation, Inbounding, GrabCooldown> agentQuery;
};

class Engine : public CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
