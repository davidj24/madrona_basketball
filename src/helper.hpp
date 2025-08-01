#pragma once

#include "sim.hpp"

namespace madBasketball 
{

    float sampleUniform(Engine &ctx, float min, float max);
    int32_t getShotPointValue(Position shot_pos, Vector3 hoop_score_zone);
    Vector3 findVectorToCenter(Engine &ctx, Position pos);
    Quat findRotationBetweenVectors(Vector3 start, Vector3 target);
    void setupAgentPositions(Engine &ctx, Entity basketball_entity, int32_t &offensive_agent_id, Position &agent_pos_for_ball);
    struct Projection 
    {
        float min;
        float max;
    };
    Projection projectRectangle(const Vector3* vertices, const Vector3& axis);
    bool projectionsOverlap(const Projection& p1, const Projection& p2);

}