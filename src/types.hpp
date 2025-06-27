#pragma once

#include <madrona/components.hpp>

using namespace madrona::math;

namespace madsimple {

// ================================================ Config Constants ================================================
    constexpr int32_t NUM_AGENTS = 6;
    constexpr int32_t NUM_BASKETBALLS = 1;

    enum class ExportID : uint32_t 
    {
        Reset,
        Action,
        AgentPos,
        BasketballPos,
        Reward,
        Done,
        NumExports,
    };

struct Reset 
{
    int32_t resetNow;
};

struct Action 
{
    int32_t moveSpeed; // [0, 3] - how fast to move
    int32_t moveAngle;  // [0, 7] - which direction (8 directions)
    int32_t rotate;     // [-2, 2] - turning
    int32_t grab;       // 0/1 - grab action
};

struct GridPos 
{
    int32_t x;
    int32_t y;
    int32_t z;
};

struct Orientation
{
    Quat orientation;
};

struct Reward 
{
    float r;
};

struct Done 
{
    float episodeDone;
};

struct RandomMovement 
{
    float moveTimer;
    float moveInterval;
};

struct InPossession 
{
    bool hasBall;
    uint32_t ballEntityID;
};

struct Grabbed 
{
    bool isGrabbed;
    uint32_t holderEntityID;
};



// ================================================ Archetypes ================================================

struct CurStep 
{
    uint32_t step;
};

struct Agent : public madrona::Archetype<
    Reset,
    Action,
    GridPos,
    Reward,
    Done,
    CurStep,
    InPossession,
    Orientation
> {};


struct Basketball : public madrona::Archetype<
    Reset,
    GridPos,
    // RandomMovement,
    Done,
    CurStep,
    Grabbed
> {};

}


