#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace madBasketball {
static Cell * setupCells(int64_t grid_x, int64_t grid_y)

{
    Cell *cells = new Cell[grid_x * grid_y]();
    return cells;
}

NB_MODULE(madrona_basketball, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<Manager> (m, "SimpleGridworldSimulator")
        .def("__init__", [](Manager *self,
                            int64_t discrete_x,
                            int64_t discrete_y,
                            float start_x,  // Changed to float
                            float start_y,  // Changed to float
                            int64_t max_episode_length,
                            madrona::py::PyExecMode exec_mode,
                            int64_t num_worlds,
                            int64_t gpu_id) {

            Cell *cells = setupCells(discrete_x, discrete_y);

            // Convert discrete grid to continuous dimensions
            // Assuming 1 cell per meter resolution
            int32_t cells_per_meter = 1;
            float width_meters = (float) discrete_x / (float) cells_per_meter;
            float height_meters = (float) discrete_y / (float) cells_per_meter;

            new (self) Manager(Manager::Config {
                .maxEpisodeLength = (uint32_t)max_episode_length,
                .execMode = exec_mode,
                .numWorlds = (uint32_t)num_worlds,
                .gpuID = (int)gpu_id,
            }, GridState {
                .cells = cells,
                .startX = start_x,
                .startY = start_y,
                .width = width_meters,
                .height = height_meters,
                .cellsPerMeter = cells_per_meter,
                .discreteWidth = (int32_t)discrete_x,
                .discreteHeight = (int32_t)discrete_y,
            });

            delete[] cells;
        }, nb::arg("discrete_x"),
           nb::arg("discrete_y"),
           nb::arg("start_x"),
           nb::arg("start_y"),
           nb::arg("max_episode_length"),
           nb::arg("exec_mode"),
           nb::arg("num_worlds"),
           nb::arg("gpu_id") = -1)


        //=================================================== General Tensors ===================================================
        .def("step", &Manager::step)
        .def("set_action", &Manager::setAction,
             "Set enhanced action with move_speed, move_angle, rotate, grab, pass, shoot",
             nb::arg("world_idx"), nb::arg("agent_idx"), nb::arg("move_speed"), 
             nb::arg("move_angle"), nb::arg("rotate"), nb::arg("grab"), nb::arg("pass"), nb::arg("shoot"))
        .def("trigger_reset", &Manager::triggerReset)
        .def("reset_tensor", &Manager::resetTensor)
        .def("game_state_tensor", &Manager::gameStateTensor)


        //=================================================== Agent Tensors ===================================================
        .def("action_tensor", &Manager::actionTensor)
        .def("action_mask_tensor", &Manager::actionMaskTensor)
        .def("agent_pos_tensor", &Manager::agentPosTensor)
        .def("observations_tensor", &Manager::observationsTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("agent_possession_tensor", &Manager::agentPossessionTensor)
        .def("agent_entity_id_tensor", &Manager::agentEntityIDTensor)
        .def("agent_team_tensor", &Manager::agentTeamTensor)
        .def("orientation_tensor", &Manager::orientationTensor)
        .def("agent_stats_tensor", &Manager::agentStatsTensor)
        


        //=================================================== Basketball Tensors ===================================================
        .def("basketball_pos_tensor", &Manager::basketballPosTensor)
        .def("ball_physics_tensor", &Manager::ballPhysicsTensor)
        .def("ball_grabbed_tensor", &Manager::ballGrabbedTensor)
        .def("ball_entity_id_tensor", &Manager::ballEntityIDTensor)


        //=================================================== Hoop Tensors ===================================================
        .def("hoop_pos_tensor", &Manager::hoopPosTensor)

    ;
}

}
