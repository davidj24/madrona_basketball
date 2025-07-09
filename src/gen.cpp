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
        .inboundingInProgress = 0.0f,
        .liveBall = 1.0f,
        .period = 1.0f,
        .teamInPossession = 0.0f,
        .team0Hoop = 0.0f,
        .team0Score = 0.0f,
        .team1Hoop = 1.0f,
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
    worldClock.resetNow = false;

    // Initialize GameState and create hoops first
    const GridState* grid = ctx.data().grid;
    GameState &gameState = ctx.singleton<GameState>();
    for (int i = 0; i < NUM_HOOPS; i++) {
        Entity hoop = ctx.makeEntity<Hoop>();
        ctx.data().hoops[i] = hoop;
        Position hoop_pos;

        float court_start_x = (grid->width - COURT_LENGTH_M) / 2.0f;
        float court_center_y = grid->height / 2.0f;

        if (i == 0) {
            gameState.team0Hoop = hoop.id;
            hoop_pos = Position {
                Vector3{
                    court_start_x + HOOP_FROM_BASELINE_M,
                    court_center_y,
                    0.f
                }
            };
        }
        else if (i == 1) {
            gameState.team1Hoop = hoop.id;
            hoop_pos = Position {
                Vector3{
                    court_start_x + COURT_LENGTH_M - HOOP_FROM_BASELINE_M,
                    court_center_y,
                    0.f
                }
            };
        }
        else {
            hoop_pos = Position
            {
                Vector3{
                    grid->startX + 10.0f + i * 5.0f,
                    grid->startY + 10.0f,
                    0.f
                }
            };
        }

        ctx.get<Position>(hoop) = hoop_pos;
        ctx.get<Reset>(hoop) = Reset{0};
        ctx.get<Done>(hoop).episodeDone = 0.f;
        ctx.get<CurStep>(hoop).step = 0;
        ctx.get<ImAHoop>(hoop) = ImAHoop{};
        ctx.get<ScoringZone>(hoop) = ScoringZone
        {
            HOOP_SCORE_ZONE_SIZE,
            .1f,
            Vector3{hoop_pos.position.x, hoop_pos.position.y, hoop_pos.position.z}
        };
    }




    Entity basketball = ctx.makeEntity<Basketball>();
    ctx.data().balls[0] = basketball;
    ctx.get<Position>(basketball) = Position { Vector3{grid->startX, grid->startY, 0.f} };
    ctx.get<Reset>(basketball) = Reset{0};
    ctx.get<Done>(basketball).episodeDone = 0.f;
    ctx.get<CurStep>(basketball).step = 0;
    // We will set the Grabbed component later, after we know the agent's ID.
    ctx.get<Grabbed>(basketball) = Grabbed {false, ENTITY_ID_PLACEHOLDER};
    ctx.get<BallPhysics>(basketball) = BallPhysics {false, Vector3::zero(), ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, 2};

    // Now create agents with proper hoop references
    uint32_t offensive_agent_id = ENTITY_ID_PLACEHOLDER;
    Vector3 team_colors[2] = {Vector3{0, 100, 255}, Vector3{255, 0, 100}};
    for (int i = 0; i < NUM_AGENTS; i++)
    {
        Entity agent = ctx.makeEntity<Agent>();
        ctx.data().agents[i] = agent;
        ctx.get<Action>(agent) = Action{0, 0, 0, 0, 0, 0};
        ctx.get<ActionMask>(agent) = ActionMask{0, 0, 0, 0};
        Position &agent_pos = ctx.get<Position>(agent);
        if (gameState.isOneOnOne == 1.f)
        {
            // Calculate the base starting position
            Vector3 base_pos = {
                grid->startX + (i * 2.f), // Spread them out for 1v1
                grid->startY,
                0.f
            };

            // Generate a random deviation
            float x_dev = sampleUniform(ctx, -START_POS_STDDEV, START_POS_STDDEV);
            float y_dev = sampleUniform(ctx, -START_POS_STDDEV, START_POS_STDDEV);

            // Add the deviation to the base position
            agent_pos.position = base_pos + Vector3{x_dev, y_dev, 0.f};

            agent_pos.position.x = clamp(agent_pos.position.x, 0.f, grid->width);
            agent_pos.position.y = clamp(agent_pos.position.y, 0.f, grid->height);

            if (i == 0)
            {
                offensive_agent_id = agent.id;
                // Give them possession of the ball we just created.
                ctx.get<InPossession>(agent) = {true, static_cast<uint32_t>(basketball.id), 2};
            }
            else
            {
                // All other agents in 1v1 mode start without the ball.
                ctx.get<InPossession>(agent) = {false, ENTITY_ID_PLACEHOLDER, 2};
            }
        }
        else
        {
            // Original 5v5 starting positions
            agent_pos = Position {
                Vector3{
                    grid->startX - 1 - (-2*(i % 2)),
                    grid->startY - 2 + i/2,
                    0.f
                }
            };
        }

        ctx.get<Reset>(agent) = Reset{0};
        ctx.get<Inbounding>(agent) = Inbounding{false, true};
        ctx.get<Reward>(agent).r = 0.f;
        ctx.get<Done>(agent).episodeDone = 0.f;
        ctx.get<CurStep>(agent).step = 0;
        ctx.get<Orientation>(agent) = Orientation {Quat::id()};
        ctx.get<GrabCooldown>(agent) = GrabCooldown{0.f};
        ctx.get<Stats>(agent) = {0.f, 0.f};
        ctx.get<Attributes>(agent) = {1 - i*DEFENDER_SLOWDOWN, 0.f, 0.f, 6.5f, ctx.get<Position>(agent).position};

        // Use actual hoop entity IDs from gameState
        uint32_t defending_hoop_id = (i % 2 == 0) ? gameState.team0Hoop : gameState.team1Hoop;
        ctx.get<Team>(agent) = Team{i % 2, team_colors[i % 2], defending_hoop_id};
    };

    if (gameState.isOneOnOne == 1.f)
    {
        ctx.get<Grabbed>(basketball) = {true, offensive_agent_id};
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
            .inboundingInProgress = 0.0f,
            .liveBall = 1.0f,
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

    // --- Part 2: Reset All Agents ---
    Vector3 team_colors[2] = {Vector3{0, 100, 255}, Vector3{255, 0, 100}};
    Entity basketball_entity;
    for (CountT i = 0; i < NUM_BASKETBALLS; i++) {
        Entity ball = ctx.data().balls[i];
        basketball_entity = ball;
    }
    uint32_t offensive_agent_id = ENTITY_ID_PLACEHOLDER;

    int agent_i = 0;
    for (CountT i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];

        // Reset all components to their default state
        ctx.get<Action>(agent) = Action{0, 0, 0, 0, 0, 0};
        ctx.get<ActionMask>(agent) = ActionMask{0, 0, 0, 0};
        ctx.get<Reset>(agent) = Reset{0};
        ctx.get<Inbounding>(agent) = Inbounding{false, true};

        // reward.r = 0.f;
        ctx.get<Done>(agent).episodeDone = 1.f;
        ctx.get<CurStep>(agent).step = 0;
        ctx.get<InPossession>(agent) = InPossession{false, ENTITY_ID_PLACEHOLDER, 2};
        ctx.get<Orientation>(agent) = Orientation{Quat::id()};
        ctx.get<GrabCooldown>(agent) = GrabCooldown{0.f};
        ctx.get<Stats>(agent) = Stats{0.f, 0.f};

        Position &pos = ctx.get<Position>(agent);
        InPossession &in_pos = ctx.get<InPossession>(agent);
        if (gameState.isOneOnOne == 1.f)
        {
            Vector3 base_pos = { grid->startX + (agent_i * 2.f), grid->startY, 0.f };
            float x_dev = sampleUniform(ctx, -START_POS_STDDEV, START_POS_STDDEV);
            float y_dev = sampleUniform(ctx, -START_POS_STDDEV, START_POS_STDDEV);
            pos.position = base_pos + Vector3{x_dev, y_dev, 0.f};
            pos.position.x = clamp(pos.position.x, 0.f, grid->width);
            pos.position.y = clamp(pos.position.y, 0.f, grid->height);

            if (agent_i == 0)
            {
                offensive_agent_id = agent.id;
                in_pos = {true, (uint32_t)basketball_entity.id, 2};
            }
            else
            {
                pos = Position { Vector3{ grid->startX - 1 - (-2*(agent_i % 2)), grid->startY - 2 + agent_i/2, 0.f } };
                in_pos = {false, ENTITY_ID_PLACEHOLDER, 2};
            }
        }
        else
        {
            pos = Position { Vector3{ grid->startX - 1 - (-2*(agent_i % 2)), grid->startY - 2 + agent_i/2, 0.f } };
        }

        ctx.get<Attributes>(agent) = {1.f - agent_i*DEFENDER_SLOWDOWN, 0.f, 0.f, 6.5f, pos.position};

        uint32_t defending_hoop_id = (agent_i % 2 == 0) ? gameState.team0Hoop : gameState.team1Hoop;
        ctx.get<Team>(agent) = Team{agent_i % 2, team_colors[agent_i % 2], defending_hoop_id};

        agent_i++;
    }

    // --- Part 3: Reset All Basketballs ---
    for (CountT i = 0; i < NUM_BASKETBALLS; i++) {
        Entity ball = ctx.data().balls[i];
        ctx.get<Position>(ball) = Position { Vector3{grid->startX, grid->startY, 0.f} };
        ctx.get<Reset>(ball).resetNow = 0;
        ctx.get<Done>(ball).episodeDone = 1.f;
        ctx.get<CurStep>(ball).step = 0;
        ctx.get<BallPhysics>(ball) = BallPhysics {false, Vector3::zero(), ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, ENTITY_ID_PLACEHOLDER, 2};
        Grabbed &grabbed = ctx.get<Grabbed>(ball);
        if (gameState.isOneOnOne == 1.f)
        {
            grabbed = Grabbed{true, offensive_agent_id};
        }
        else
        {
            grabbed = Grabbed{false, ENTITY_ID_PLACEHOLDER};
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
