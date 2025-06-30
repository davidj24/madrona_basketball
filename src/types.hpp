#pragma once

#include <madrona/components.hpp>

using namespace madrona::math;

namespace madsimple {

// ================================================ Config Constants ================================================
    constexpr int32_t NUM_AGENTS = 4;
    constexpr int32_t NUM_BASKETBALLS = 1;
    constexpr int32_t NUM_HOOPS = 2;
    constexpr uint32_t ENTITY_ID_PLACEHOLDER = UINT32_MAX;  // Use max value as invalid/null entity ID

    enum class ExportID : uint32_t 
    {
        Reset,
        Action,
        AgentPos,
        BasketballPos,
        HoopPos,
        Reward,
        Done,
        BallPhysicsData,
        AgentEntityID,
        BallEntityID,
        AgentPossession,
        BallGrabbed,
        TeamData,
        GameState,
        GameStateInbounding,
        NumExports,
    };


// ================================================ Singletons ================================================
    struct GameState
    {
        bool inboundingInProgress;
        uint32_t period;
        float gameClock; // Time left, figure out if this is in seconds or timesteps, and how it should work with tickSystem
        float shotClock;

        // Maybe add states for free throws, fouls, and jump balls later
    };



// ================================================ Components ================================================

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
        int32_t pass;       // 0/1 - pass action
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

    struct Team
    {
        int32_t teamIndex;
        Vector3 teamColor;
    };

    struct BallPhysics
    {
        bool in_flight; // I should've camelCased this but now it's too late and I don't want to find every relevant instance and replace it bc it's also used in other files I didn't make
        Vector3 velocity;
        uint32_t lastTouchedByID; // This is an entity ID of which entity last touched the ball
    };

    struct Inbounding
    {
        bool imInbounding;
        bool allowedToMove;
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
        Orientation,
        Inbounding,
        Team
    > {};


    struct Basketball : public madrona::Archetype<
        Reset,
        GridPos,
        BallPhysics,
        // RandomMovement,
        Done,
        CurStep,
        Grabbed
    > {};


    struct Hoop : public madrona::Archetype<
        Reset,
        GridPos,
        // RandomMovement,
        Done,
        CurStep
    > {};
}


