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

int32_t getShotPointValue(Position shot_pos, Position hoop_pos)
{
    float distance_to_hoop = (shot_pos.position - hoop_pos.position).length();

    // 1. Check if the shot is in the corner lane, relative to the court's position.
    bool isInCornerLane = (shot_pos.position.y < COURT_MIN_Y + CORNER_3_FROM_SIDELINE_M ||
                        shot_pos.position.y > COURT_MIN_Y + COURT_WIDTH_M - CORNER_3_FROM_SIDELINE_M);

    if (isInCornerLane) {
        // 2. If so, check if the shot is within the corner's length, relative to the court's position.
        bool isShootingAtLeftHoop = hoop_pos.position.x < WORLD_WIDTH_M / 2.0f;

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

}
