#pragma once

#include <madrona/components.hpp>

using namespace madrona::math;

namespace madsimple {

// ================================================ Config Constants ================================================
    constexpr int32_t NUM_AGENTS = 8;
    constexpr int32_t NUM_BASKETBALLS = 1;
    constexpr int32_t NUM_HOOPS = 2;
    constexpr uint32_t ENTITY_ID_PLACEHOLDER = UINT32_MAX;  // Use max value as invalid/null entity ID

    enum class ExportID : uint32_t 
    {
        // General Exports
        Reset,
        GameState,

        // Agent Exports
        Action,
        ActionMask,
        AgentPos,
        Reward,
        Done,
        AgentEntityID,
        AgentPossession,
        Orientation,
        TeamData,

        // Basketball Exports
        BasketballPos,
        BallPhysicsData,
        BallEntityID,
        BallGrabbed,

        // Hoop Exports
        HoopPos,



        NumExports,
    };


// ================================================ Singletons ================================================
    struct GameState
    {
        float inboundingInProgress; // 0.0f if false, 1.0f if true
        float liveBall; // 0.0f if dead ball, 1.0f if live ball

        float period;
        float teamInPossession; // The index of the team that is currently in possession of the ball
        float team0Hoop; // Entity id of team 0's hoop (will switch at half time)
        float team0Score;
        float team1Hoop;
        float team1Score;

        float gameClock; // Time left, figure out if this is in seconds or timesteps, and how it should work with tickSystem
        float shotClock;

        float scoredBaskets;
        float outOfBoundsCount;

        float inboundClock;
    };



// ======================================================================================================= General Components =======================================================================================================

    struct Reset 
    {
        int32_t resetNow;
    };

    struct Position 
    {
        Vector3 position;
    };

    struct RandomMovement 
    {
        float moveTimer;
        float moveInterval;
    };

    struct IsWorldClock {};

// ======================================================================================================= Agent Components =======================================================================================================

    struct Action 
    {
        // General Actions
        int32_t moveSpeed;  // [0, 3] - how fast to move
        int32_t moveAngle;  // [0, 7] - which direction (8 directions)
        int32_t rotate;     // [-2, 2] - turning
        int32_t grab;       // 0/1 - grab action

        // Offensive Actions
        int32_t pass;       // 0/1 - pass action
        int32_t shoot;      // Currently Adding

        // Defensive Actions
        // int32_t take charge <--- later
    };

    struct ActionMask
    {
        float can_move;
        float can_grab;
        float can_pass;
        float can_shoot;
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

    struct InPossession 
    {
        bool hasBall;
        uint32_t ballEntityID;
    };

    struct Inbounding
    {
        bool imInbounding;
        bool allowedToMove;
    };

    struct Team
    {
        int32_t teamIndex;
        Vector3 teamColor;
        uint32_t defendingHoopID; // The ID of the hoop that the agent is defending
    };

    struct GrabCooldown
    {
        float cooldown;
    };

// ======================================================================================================= Ball Components =======================================================================================================

    struct Grabbed 
    {
        bool isGrabbed;
        uint32_t holderEntityID;
    };

    struct BallPhysics
    {
        bool inFlight;
        Vector3 velocity;
        uint32_t lastTouchedByID; // This is a team ID of which entity last touched the ball 
        int32_t pointsWorth; // The amount of points the ball is worth (2 or 3)
    };

    
// ======================================================================================================= Hoop Components =======================================================================================================
    struct ImAHoop{}; // This component is just a tag to differenitate that a hoop entity is a hoop

    // If the ball is FULLY within the scoring zone, it should count as a made basket
    struct ScoringZone
    {
        float radius;
        float height; // The scoring zone should be a cyliner
        Vector3 center;
    };


// ================================================ Archetypes ================================================

    struct CurStep 
    {
        uint32_t step;
    };

    struct Agent : public madrona::Archetype<
        Reset,
        Action,
        ActionMask,
        GrabCooldown,
        Position,
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
        Position,
        BallPhysics,
        // RandomMovement,
        Done,
        CurStep,
        Grabbed
    > {};

    struct Hoop : public madrona::Archetype<
        Reset,
        Position,
        ImAHoop,
        ScoringZone,
        // RandomMovement,
        Done,
        CurStep
    > {};

    struct WorldClock : public madrona::Archetype<
        IsWorldClock,
        Reset
    > {};
}


