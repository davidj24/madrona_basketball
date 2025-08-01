#include "types.hpp"
#include "constants.hpp"
#include "helper.hpp"
#include "gen.hpp"

using namespace madrona;
using namespace madrona::math;



namespace madBasketball {

void generateWorld(Engine &ctx) {
    ctx.singleton<GameState>() = GameState
    {
        .inboundingInProgress = 0,
        .liveBall = 1,
        .period = 1.0f,
        .teamInPossession = 0.0f,
        .team0Hoop = 0,
        .team0Score = 0.0f,
        .team1Hoop = 1,
        .team1Score = 0.0f,
        .gameClock = TIME_PER_PERIOD,
        .shotClock = 24.0f,
        .scoredBaskets = 0.f,
        .outOfBoundsCount = 0.f,
        .inboundClock = 0.0f,
        .isOneOnOne = ONE_ON_ONE,
    };

    // Make sure to add the Reset component to the WorldClock entity
    WorldClock worldClock = ctx.singleton<WorldClock>();
    worldClock.resetNow = 0;

    // Initialize GameState
    const GridState* grid = ctx.data().grid;
    GameState &gameState = ctx.singleton<GameState>();

    // Create hoops
    // for (int i = 0; i < NUM_HOOPS; i++) {
    //     Entity hoop = ctx.makeEntity<Hoop>();
    //     ctx.data().hoops[i] = hoop;
    //     Position hoop_pos;

    //     float court_start_x = (grid->width - COURT_LENGTH_M) / 2.0f;
    //     float court_center_y = grid->height / 2.0f;

    //     if (i == 0) {
    //         gameState.team0Hoop = hoop.id;
    //         hoop_pos = Position {
    //             Vector3{
    //                 court_start_x + HOOP_FROM_BASELINE_M,
    //                 court_center_y,
    //                 0.f
    //             }
    //         };
    //     }
    //     else if (i == 1) {
    //         gameState.team1Hoop = hoop.id;
    //         hoop_pos = Position {
    //             Vector3{
    //                 court_start_x + COURT_LENGTH_M - HOOP_FROM_BASELINE_M,
    //                 court_center_y,
    //                 0.f
    //             }
    //         };
    //     }
    //     else {
    //         hoop_pos = Position
    //         {
    //             Vector3{
    //                 grid->startX + 10.0f + i * 5.0f,
    //                 grid->startY + 10.0f,
    //                 0.f
    //             }
    //         };
    //     }

    //     ctx.get<Position>(hoop) = hoop_pos;
    //     ctx.get<Reset>(hoop) = Reset{0};
    //     ctx.get<Done>(hoop).episodeDone = 0.f;
    //     ctx.get<CurStep>(hoop).step = 0;
    //     ctx.get<ImAHoop>(hoop) = ImAHoop{};
    //     ctx.get<ScoringZone>(hoop) = ScoringZone
    //     {
    //         HOOP_SCORE_ZONE_SIZE,
    //         .1f,
    //         Vector3{hoop_pos.position.x, hoop_pos.position.y, hoop_pos.position.z}
    //     };
    // }


    
    
    float court_start_x = (grid->width - COURT_LENGTH_M) / 2.0f;
    float court_center_y = grid->height / 2.0f;
    
    
    // Create hoop 0
    Entity hoop = ctx.makeEntity<Hoop>();
    ctx.data().hoops[0] = hoop;
    gameState.team0Hoop = hoop.id;
    Position hoop0_pos;
    hoop0_pos = Position 
    {
        Vector3{
            court_start_x + HOOP_FROM_BASELINE_M,
            court_center_y,
            0.f
        }
    };


    ctx.get<Position>(hoop) = hoop0_pos;


    ctx.get<Reset>(hoop) = Reset{0};
    ctx.get<Done>(hoop).episodeDone = 0.f;
    ctx.get<CurStep>(hoop).step = 0;
    ctx.get<ImAHoop>(hoop) = ImAHoop{};
    ctx.get<ScoringZone>(hoop) = ScoringZone
    {
        HOOP_SCORE_ZONE_SIZE,
        .1f,
        Vector3{hoop0_pos.position.x, hoop0_pos.position.y, hoop0_pos.position.z}
    };


    // create hoop 1
    Entity hoop1 = ctx.makeEntity<Hoop>();
    ctx.data().hoops[1] = hoop1;
    gameState.team1Hoop = hoop1.id;
    Position hoop1_pos;
    hoop1_pos = Position {
        Vector3{
            court_start_x + COURT_LENGTH_M - HOOP_FROM_BASELINE_M,
            court_center_y, 
            0.f
        }
    };

    
    ctx.get<Position>(hoop1) = hoop1_pos;


    ctx.get<Reset>(hoop1) = Reset{0};
    ctx.get<Done>(hoop1).episodeDone = 0.f;
    ctx.get<CurStep>(hoop1).step = 0;
    ctx.get<ImAHoop>(hoop1) = ImAHoop{};
    ctx.get<ScoringZone>(hoop1) = ScoringZone
    {
        HOOP_SCORE_ZONE_SIZE,
        .1f,
        Vector3{hoop1_pos.position.x, hoop1_pos.position.y, hoop1_pos.position.z}
    };





    




    Entity basketball = ctx.makeEntity<Basketball>();
    ctx.data().balls[0] = basketball;
    ctx.get<Position>(basketball) = Position { Vector3{grid->startX, grid->startY, 0.f} };
    ctx.get<Reset>(basketball) = Reset{0};
    ctx.get<Done>(basketball).episodeDone = 0.f;
    ctx.get<CurStep>(basketball).step = 0;
    // We will set the Grabbed component later, after we know the agent's ID.
    ctx.get<Grabbed>(basketball) = Grabbed {0, ENTITY_ID_PLACEHOLDER};
    ctx.get<BallPhysics>(basketball) = BallPhysics {0, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, 2, 0};
    ctx.get<Velocity>(basketball).velocity = Vector3::zero();




    // Now create agents with proper hoop references
    int32_t offensive_agent_id = ENTITY_ID_PLACEHOLDER;
    Position agent_pos_for_ball = {grid->startX, grid->startY, 0.f};
    
    // Create agent entities first
    for (int i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.makeEntity<Agent>();
        ctx.data().agents[i] = agent;
        ctx.get<Action>(agent) = Action{0, 0, 0, 0, 0, 0};
        ctx.get<ActionMask>(agent) = ActionMask{0, 0, 0, 0};
        ctx.get<Reset>(agent) = Reset{0};
        ctx.get<Inbounding>(agent) = Inbounding{0, 1};
        ctx.get<Reward>(agent).r = 0.f;
        ctx.get<Done>(agent).episodeDone = 0.f;
        ctx.get<CurStep>(agent).step = 0;
        ctx.get<Orientation>(agent) = (i % 2 == 0) ? Orientation{Quat::angleAxis(-madrona::math::pi / 2.0f, {0.f, 0.f, 1.f})} : Orientation{Quat::angleAxis(madrona::math::pi / 2.0f, {0.f, 0.f, 1.f})};
        ctx.get<GrabCooldown>(agent) = GrabCooldown{0.f};
        ctx.get<Stats>(agent) = {0.f, 0.f};
        ctx.get<Velocity>(agent).velocity = Vector3::zero();
        
        // Use actual hoop entity IDs from gameState
        int32_t defending_hoop_id = (i % 2 == 0) ? gameState.team0Hoop : gameState.team1Hoop;
        // Use team color constants
        const madrona::math::Vector3 team_color = (i % 2 == 0) ? TEAM0_COLOR : TEAM1_COLOR;
        ctx.get<Team>(agent) = Team{i % 2, team_color, defending_hoop_id};
    }
    
    // Use helper function to setup positions and ball possession
    setupAgentPositions(ctx, basketball, offensive_agent_id, agent_pos_for_ball);

    if (gameState.isOneOnOne == 1.f) {
        ctx.get<Grabbed>(basketball) = {1, offensive_agent_id};
    }
}

void resetWorld(Engine &ctx) {
    GameState &gameState = ctx.singleton<GameState>();
    const GridState *grid = ctx.data().grid;

    // --- Part 1: Reset GameState Singleton ---
    if (gameState.gameClock <= 0.f && gameState.isOneOnOne == 0.f)
    {
        // End-of-quarter logic (only for full games)
        if (gameState.period < 4 || gameState.team0Score == gameState.team1Score)
        {
            gameState.period++;
            gameState.gameClock = TIME_PER_PERIOD;
            gameState.shotClock = 24.0f;
            gameState.liveBall = 1.0f;
            gameState.inboundingInProgress = 0.f;
        }
        else
        {
            gameState.liveBall = 0.f; // Game over
        }
    }
    else
    {
        // This was a manual reset or a reset after a score in 1v1 mode.
        gameState = GameState {
            .inboundingInProgress = 0,
            .liveBall = 1,
            .period = 1.0f,
            .teamInPossession = 0.0f,
            .team0Hoop = gameState.team0Hoop, // Preserve existing hoop IDs
            .team0Score = 0.0f,
            .team1Hoop = gameState.team1Hoop, // Preserve existing hoop IDs
            .team1Score = 0.0f,
            .gameClock = TIME_PER_PERIOD,
            .shotClock = 24.0f,
            .scoredBaskets = 0.f,
            .outOfBoundsCount = 0.f,
            .inboundClock = 0.0f,
            .isOneOnOne = gameState.isOneOnOne // Preserve the 1v1 mode setting
        };
    }

    Vector3 team_colors[2] = {Vector3{0, 100, 255}, Vector3{255, 0, 100}};
    Entity basketball_entity;
    for (CountT i = 0; i < NUM_BASKETBALLS; i++) {
        Entity ball = ctx.data().balls[i];
        basketball_entity = ball;
    }
    int32_t offensive_agent_id = ENTITY_ID_PLACEHOLDER;
    Position agent_pos_for_ball = {grid->startX, grid->startY, 0.f};
    
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];

        // Reset all components to their default state
        ctx.get<Action>(agent) = Action{0, 0, 0, 0, 0, 0};
        ctx.get<ActionMask>(agent) = ActionMask{0, 0, 0, 0};
        ctx.get<Reset>(agent) = Reset{0};
        ctx.get<Inbounding>(agent) = Inbounding{0, 1};
        ctx.get<Done>(agent).episodeDone = 1.f;
        ctx.get<CurStep>(agent).step = 0;
        ctx.get<Orientation>(agent) = (i % 2 == 0) ? Orientation{Quat::angleAxis(-madrona::math::pi / 2.0f, {0.f, 0.f, 1.f})} : Orientation{Quat::angleAxis(madrona::math::pi / 2.0f, {0.f, 0.f, 1.f})};
        ctx.get<GrabCooldown>(agent) = GrabCooldown{0.f};
        ctx.get<Stats>(agent) = Stats{0.f, 0.f};
        ctx.get<Velocity>(agent).velocity = Vector3::zero();
        
        int32_t defending_hoop_id = (i % 2 == 0) ? gameState.team0Hoop : gameState.team1Hoop;
        ctx.get<Team>(agent) = Team{static_cast<int32_t>(i % 2), team_colors[i % 2], defending_hoop_id};
    }
    
    // Use helper function to setup positions and ball possession consistently
    setupAgentPositions(ctx, basketball_entity, offensive_agent_id, agent_pos_for_ball);

    // --- Part 3: Reset All Basketballs ---
    for (CountT i = 0; i < NUM_BASKETBALLS; i++) {
        Entity ball = ctx.data().balls[i];
        ctx.get<Position>(ball) = agent_pos_for_ball;
        ctx.get<Reset>(ball).resetNow = 0;
        ctx.get<Done>(ball).episodeDone = 1.f;
        ctx.get<CurStep>(ball).step = 0;
        ctx.get<BallPhysics>(ball) = BallPhysics {0, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, 2, 0};
        ctx.get<Velocity>(ball).velocity = Vector3::zero();
        Grabbed &grabbed = ctx.get<Grabbed>(ball);
        if (gameState.isOneOnOne == 1.f)
        {
            grabbed = Grabbed{1, offensive_agent_id};
        }
        else
        {
            grabbed = Grabbed{0, ENTITY_ID_PLACEHOLDER};
        }
    }

    // --- Part 4: Reset Hoops ---
    for (CountT i = 0; i < NUM_HOOPS; i++) {
        Entity hoop = ctx.data().hoops[i];
        ctx.get<Reset>(hoop).resetNow = 0;
        ctx.get<Done>(hoop).episodeDone = 1.f;
        ctx.get<CurStep>(hoop).step = 0;
    }
}

}
