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
    Entity worldClock = ctx.makeEntity<WorldClock>();
    ctx.get<IsWorldClock>(worldClock) = {};
    ctx.get<Reset>(worldClock) = {0}; // Initialize resetNow to 0

    // Initialize GameState and create hoops first
    const GridState* grid = ctx.data().grid;
    GameState &gameState = ctx.singleton<GameState>();
    for (int i = 0; i < NUM_HOOPS; i++) {
        Entity hoop = ctx.makeEntity<Hoop>();
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

    // Now create agents with proper hoop references
    Vector3 team_colors[2] = {Vector3{0, 100, 255}, Vector3{255, 0, 100}};
    for (int i = 0; i < NUM_AGENTS; i++)
    {
        Entity agent = ctx.makeEntity<Agent>();
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

            // It's good practice to clamp the final position to the world boundaries
            agent_pos.position.x = clamp(agent_pos.position.x, 0.f, grid->width);
            agent_pos.position.y = clamp(agent_pos.position.y, 0.f, grid->height);
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
        ctx.get<InPossession>(agent) = {false, ENTITY_ID_PLACEHOLDER, 2};
        ctx.get<Orientation>(agent) = Orientation {Quat::id()};
        ctx.get<GrabCooldown>(agent) = GrabCooldown{0.f};
        ctx.get<Stats>(agent) = {0.f, 0.f};
        ctx.get<Attributes>(agent) = {1 - i*DEFENDER_SLOWDOWN, 0.f, 0.f, 6.5f, ctx.get<Position>(agent).position};

        // Use actual hoop entity IDs from gameState
        uint32_t defending_hoop_id = (i % 2 == 0) ? gameState.team0Hoop : gameState.team1Hoop;
        ctx.get<Team>(agent) = Team{i % 2, team_colors[i % 2], defending_hoop_id};
    };


    for (int i = 0; i < NUM_BASKETBALLS; i++)
    {
        Entity basketball = ctx.makeEntity<Basketball>();
        ctx.get<Position>(basketball) = Position { Vector3{grid->startX, grid->startY, 0.f} };
        ctx.get<Reset>(basketball) = Reset{0};
        ctx.get<Done>(basketball).episodeDone = 0.f;
        ctx.get<CurStep>(basketball).step = 0;
        ctx.get<Grabbed>(basketball) = Grabbed {false, ENTITY_ID_PLACEHOLDER};
        ctx.get<BallPhysics>(basketball) = BallPhysics {
            false,
            Vector3::zero(),
            ENTITY_ID_PLACEHOLDER,
            ENTITY_ID_PLACEHOLDER,
            ENTITY_ID_PLACEHOLDER,
            ENTITY_ID_PLACEHOLDER,
            2
        };
    }

}
}
