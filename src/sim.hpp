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

    Entity balls[NUM_BASKETBALLS];
    Entity hoops[NUM_HOOPS];
    Entity agents[NUM_AGENTS];
};

class Engine : public CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
