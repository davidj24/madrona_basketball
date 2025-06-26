#pragma once

#include <madrona/components.hpp>

namespace madsimple {

enum class ExportID : uint32_t {
    Reset,
    Action,
    AgentPos,
    BasketballPos,
    Reward,
    Done,
    NumExports,
};

struct Reset {
    int32_t resetNow;
};

enum class Action : int32_t {
    Up    = 0,
    Down  = 1,
    Left  = 2,
    Right = 3,
    None,
};

struct GridPos {
    int32_t x;
    int32_t y;
    int32_t z;
};

struct Reward {
    float r;
};

struct Done {
    float episodeDone;
};



// ================================================ Archetypes ================================================

struct CurStep {
    uint32_t step;
};

struct Agent : public madrona::Archetype<
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep
> {};


struct Basketball : public madrona::Archetype<
    Reset,
    GridPos,
    Done,
    CurStep
> {};

}


