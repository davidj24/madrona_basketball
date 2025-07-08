#pragma once

#include <madrona/sync.hpp>

#include "grid.hpp"

namespace madBasketball {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    const GridState *grid;
};

}
