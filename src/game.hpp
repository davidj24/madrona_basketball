#pragma once
#include "sim.hpp"

using namespace madrona;

namespace madBasketball {

TaskGraphNodeID setupGameStepTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps);

}