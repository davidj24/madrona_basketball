#pragma once

#include "sim.hpp"

namespace madBasketball {

float sampleUniform(Engine &ctx, float min, float max);
int32_t getShotPointValue(Position shot_pos, Position hoop_pos);
Vector3 findVectorToCenter(Engine &ctx, Position pos);
Quat findRotationBetweenVectors(Vector3 start, Vector3 target);

}