#pragma once

#include <madrona/components.hpp>
#include "constants.hpp"

using namespace madrona::math;

namespace madBasketball {

    enum class ExportID : uint32_t 
    {
        // General Exports
        Reset,
        GameState,

        // Agent Exports
        Action,
        ActionMask,
        AgentPos,
        Observations,
        Reward,
        Done,
        AgentEntityID,
        AgentPossession,
        Orientation,
        TeamData,
        AgentStats,

        // Basketball Exports
        BasketballPos,
        BallPhysicsData,
        BallEntityID,
        BallGrabbed,
        BallVelocity,

        // Hoop Exports
        HoopPos,



        NumExports,
    };


// ================================================ Singletons ================================================
    struct GameState
    {
        int32_t inboundingInProgress; // 0 if false, 1 if true
        int32_t liveBall; // 0 if dead ball, 1 if live ball

        float period;
        float teamInPossession; // The index of the team that is currently in possession of the ball
        int32_t team0Hoop; // Entity id of team 0's hoop (will switch at half time) - converted to int32_t to match entity IDs
        float team0Score;
        int32_t team1Hoop; // Entity id of team 1's hoop - converted to int32_t to match entity IDs
        float team1Score;

        float gameClock; // Time left, figure out if this is in seconds or timesteps, and how it should work with tickSystem
        float shotClock;

        float scoredBaskets;
        float outOfBoundsCount;

        float inboundClock;
        int32_t isOneOnOne; // 0 if false, 1 if true

    };

    struct WorldClock {
        int32_t resetNow;
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
    
    struct Velocity
    {
        Vector3 velocity;
    };
    
    struct RandomMovement 
    {
        float moveTimer;
        float moveInterval;
    };

// ======================================================================================================= Agent Components =======================================================================================================

    struct Action 
    {
        // General Actions
        int32_t move;  // 0/1 - move action
        int32_t moveAngle;  // [0, 7] - which direction (8 directions)
        int32_t rotate;     // [-1, 0, 1] - turning
        int32_t grab;       // 0/1 - grab action

        // Offensive Actions
        int32_t pass;       // 0/1 - pass action
        int32_t shoot;      // 0/1 - shoot action

        // Defensive Actions
        // int32_t take charge <--- later
    };

    struct ActionMask
    {
        int32_t can_move;
        int32_t can_grab;
        int32_t can_pass;
        int32_t can_shoot;
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
        int32_t hasBall;
        int32_t ballEntityID;
        int32_t pointsWorth; // Points this agent would get if they scored from their current position
    };

    struct Inbounding
    {
        int32_t imInbounding;
        int32_t allowedToMove;
    };

    struct Team
    {
        int32_t teamIndex;
        Vector3 teamColor;
        int32_t defendingHoopID; // The ID of the hoop that the agent is defending
    };

    struct GrabCooldown
    {
        float cooldown;
    };

    struct Observations
    {
        std::array<float, 128> observationsArray; // Change size as needed
    };

    struct Stats
    {
        float points;
        float fouls;
        // Add assists later if necessary
    };

    struct Attributes // Per Agent stats like how fast a given agent can move etc
    {
        float maxSpeed;
        float quickness; // How much they can accelerate
        float shooting;
        float freeThrowPercentage;
        float reactionSpeed;
        Vector3 currentTargetPosition;
        float currentShotPercentage; // The likelihood of a shot going in from the agent's current state
    };

    // ======================================================================================================= Ball Components =======================================================================================================

    struct Grabbed 
    {
        int32_t isGrabbed;
        int32_t holderEntityID;
    };

    struct BallPhysics
    {
        int32_t inFlight;
        int32_t lastTouchedByAgentID; // Entity ID of the specific agent who last touched the ball 
        int32_t lastTouchedByTeamID; // Team ID of the team that last touched the ball 
        int32_t shotByAgentID; // Entity ID of the agent who shot the ball (doesn't change after touching)
        int32_t shotByTeamID; // Team ID of the team that shot the ball (doesn't change after touching)
        int32_t shotPointValue; // Point value of the shot when it was taken (2 or 3)
        int32_t shotIsGoingIn; // Calculated at moment of release to let agent know if shot will score
    };

    
// ======================================================================================================= Hoop Components =======================================================================================================
    struct ImAHoop{}; // This component is just a tag to differenitate that a hoop entity is a hoop

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
        Observations,
        GrabCooldown,
        Position,
        Reward,
        Done,
        CurStep,
        InPossession,
        Orientation,
        Inbounding,
        Team,
        Stats,
        Attributes,
        Velocity
    > {};

    struct Basketball : public madrona::Archetype<
        Reset,
        Position,
        BallPhysics,
        // RandomMovement,
        Done,
        CurStep,
        Grabbed,
        Velocity
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
}


