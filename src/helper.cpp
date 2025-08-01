#include "helper.hpp"

using namespace madrona;
using namespace madrona::math;

namespace madBasketball {

float sampleUniform(Engine &ctx, const float min, const float max)
{
    return min + (max - min) * ctx.data().rng.sampleUniform();
}

// Computes the rotation needed to align the 'start' vector with the 'target' vector.
Quat findRotationBetweenVectors(Vector3 start, Vector3 target)
{
    // Ensure the vectors are normalized (unit length)
    start = start.normalize();
    target = target.normalize();

    float dot_product = dot(start, target);

    // Case 1: If the vectors are already aligned, no rotation is needed.
    if (dot_product > 0.999999f) {
        return Quat::id();
    }

    // Case 2: If the vectors are in opposite directions, we need a 180-degree rotation.
    // For a 2D game, the most stable axis for a 180-degree turn is the Z-axis.
    if (dot_product < -0.999999f) {
        return Quat::angleAxis(pi, Vector3{0.f, 0.f, 1.f});
    }

    // Case 3: The general case.
    // The axis of rotation is the cross product of the two vectors.
    Vector3 rotation_axis = cross(start, target);
    rotation_axis = rotation_axis.normalize();

    // The angle is the arccosine of the dot product.
    float rotation_angle = acosf(dot_product);

    return Quat::angleAxis(rotation_angle, rotation_axis);
}

Vector3 findVectorToCenter(Engine &ctx, Position pos)
{
    const GridState *grid = ctx.data().grid;
    return (Vector3{grid->startX, grid->startY, 0.f} - pos.position).normalize();
}

int32_t getShotPointValue(Position shot_pos, Vector3 hoop_score_zone)
{
    float distance_to_hoop = (shot_pos.position - hoop_score_zone).length();

    // 1. Check if the shot is in the corner lane, relative to the court's position.
    bool isInCornerLane = (shot_pos.position.y < COURT_MIN_Y + CORNER_3_FROM_SIDELINE_M ||
                        shot_pos.position.y > COURT_MIN_Y + COURT_WIDTH_M - CORNER_3_FROM_SIDELINE_M);

    if (isInCornerLane) {
        // 2. If so, check if the shot is within the corner's length, relative to the court's position.
        bool isShootingAtLeftHoop = hoop_score_zone.x < WORLD_WIDTH_M / 2.0f;

        if (isShootingAtLeftHoop) {
            if (shot_pos.position.x <= COURT_MIN_X + CORNER_3_LENGTH_FROM_BASELINE_M) {
                return 3;
            }
        } else { // Shooting at the right hoop
            if (shot_pos.position.x >= COURT_MIN_X + COURT_LENGTH_M - CORNER_3_LENGTH_FROM_BASELINE_M) {
                return 3;
            }
        }
    }

    // 3. If not a valid corner 3, check the distance against the arc.
    // If the shot is beyond the 3-point arc distance, it's a 3-pointer
    if (distance_to_hoop >= ARC_RADIUS_M) {
        return 3;
    }

    // 4. If none of the 3-point conditions are met, it is a 2-point shot.
    return 2;
}


// Helper function to project the vertices of a rectangle onto an axis
Projection projectRectangle(const Vector3* vertices, const Vector3& axis) {
    Projection p;
    p.min = dot(vertices[0], axis);
    p.max = p.min;

    for (int i = 1; i < 4; i++) {
        float proj = dot(vertices[i], axis);
        if (proj < p.min) {
            p.min = proj;
        }
        if (proj > p.max) {
            p.max = proj;
        }
    }
    return p;
}

// Helper function to check if two projections overlap
bool projectionsOverlap(const Projection& p1, const Projection& p2) {
    return p1.max > p2.min && p2.max > p1.min;
}

// Helper function to setup agent positions and ball possession consistently
void setupAgentPositions(Engine &ctx, Entity basketball_entity, int32_t &offensive_agent_id, Position &agent_pos_for_ball) {
    const GridState *grid = ctx.data().grid;
    GameState &gameState = ctx.singleton<GameState>();
    
    for (int i = 0; i < NUM_AGENTS; i++) {
        Entity agent = ctx.data().agents[i];
        Position &pos = ctx.get<Position>(agent);
        InPossession &in_pos = ctx.get<InPossession>(agent);
        
        if (gameState.isOneOnOne == 1.f) {
            if (i == 0) {
                // Agent 0 (offensive) - base position with random deviation
                Vector3 base_pos = { grid->startX + (i * 2.f), grid->startY, 0.f };
                float x_dev = sampleUniform(ctx, -START_POS_STDDEV, START_POS_STDDEV);
                float y_dev = sampleUniform(ctx, -START_POS_STDDEV, START_POS_STDDEV);
                pos.position = base_pos + Vector3{x_dev, y_dev, 0.f};
                pos.position.x = clamp(pos.position.x, 0.f, grid->width);
                pos.position.y = clamp(pos.position.y, 0.f, grid->height);
                
                agent_pos_for_ball = pos;
                offensive_agent_id = agent.id;
                in_pos = {1, basketball_entity.id, 2};
            } else {
                // Agent 1 (defensive) - spawn at radius away from agent 0
                float random_angle = sampleUniform(ctx, 0.f, 2.f * madrona::math::pi);
                
                Vector3 offset = {
                    DEFENDER_SPAWN_RADIUS * cosf(random_angle),
                    DEFENDER_SPAWN_RADIUS * sinf(random_angle),
                    0.f
                };
                
                pos.position = agent_pos_for_ball.position + offset;
                pos.position.x = clamp(pos.position.x, 0.f, grid->width);
                pos.position.y = clamp(pos.position.y, 0.f, grid->height);
                
                in_pos = {0, ENTITY_ID_PLACEHOLDER, 2};
            }
        } else {
            // Original 5v5 starting positions
            pos = Position { Vector3{ grid->startX - 1 - (-2*(i % 2)), grid->startY - 2 + i/2, 0.f } };
            if (i == 0) {
                offensive_agent_id = agent.id;
                in_pos = {1, basketball_entity.id, 2};
            } else {
                in_pos = {0, ENTITY_ID_PLACEHOLDER, 2};
            }
        }
        
        // Set common attributes for all agents
        ctx.get<Attributes>(agent) = {DEFAULT_SPEED - i*DEFENDER_SLOWDOWN, 1.f, 0.f, 0.f, i*DEFENDER_REACTION, pos.position, 0.f};
    }
}

}
