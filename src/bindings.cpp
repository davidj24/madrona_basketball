#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace madsimple {

static void setRewards(Cell *cells,
                       const float *rewards,
                       int64_t grid_x,
                       int64_t grid_y)
{
    for (int64_t y = 0; y < grid_y; y++) {
        for (int64_t x = 0; x < grid_x; x++) {
            int64_t idx = y * grid_x + x;
            cells[idx].reward = rewards[idx];
        }
    }
}

static void tagWalls(Cell *cells,
                     const bool *walls,
                     int64_t grid_x,
                     int64_t grid_y)
{
    for (int64_t y = 0; y < grid_y; y++) {
        for (int64_t x = 0; x < grid_x; x++) {
            int64_t idx = y * grid_x + x;

            if (walls[idx]) {
                cells[idx].flags |= CellFlag::Wall;
            }
        }
    }
}

static Cell * setupCellData(
    const nb::ndarray<bool, nb::shape<-1, -1>,
        nb::c_contig, nb::device::cpu> &walls,
    const nb::ndarray<float, nb::shape<-1, -1>,
        nb::c_contig, nb::device::cpu> &rewards,
    int64_t grid_x,
    int64_t grid_y)

{
    Cell *cells = new Cell[grid_x * grid_y]();

    setRewards(cells, rewards.data(), grid_x, grid_y);
    tagWalls(cells, walls.data(), grid_x, grid_y);
    
    return cells;
}

NB_MODULE(_madrona_simple_example_cpp, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<Manager> (m, "SimpleGridworldSimulator")
        .def("__init__", [](Manager *self,
                            nb::ndarray<bool, nb::shape<-1, -1>,
                                nb::c_contig, nb::device::cpu> walls,
                            nb::ndarray<float, nb::shape<-1, -1>,
                                nb::c_contig, nb::device::cpu> rewards,
                            float start_x,  // Changed to float
                            float start_y,  // Changed to float
                            int64_t max_episode_length,
                            madrona::py::PyExecMode exec_mode,
                            int64_t num_worlds,
                            int64_t gpu_id) {
            int64_t discrete_y = (int64_t)walls.shape(0);
            int64_t discrete_x = (int64_t)walls.shape(1);

            if ((int64_t)rewards.shape(0) != discrete_y ||
                (int64_t)rewards.shape(1) != discrete_x) {
                throw std::runtime_error("walls and rewards shapes don't match");
            }

            Cell *cells =
                setupCellData(walls, rewards, discrete_x, discrete_y); 

            // Convert discrete grid to continuous dimensions
            // Assuming 1 cell per meter resolution
            int32_t cells_per_meter = 1;
            float width_meters = (float)discrete_x / cells_per_meter;
            float height_meters = (float)discrete_y / cells_per_meter;

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
        }, nb::arg("walls"),
           nb::arg("rewards"),
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
        .def("game_state_inbounding_tensor", &Manager::gameStateInboundingTensor)


        //=================================================== Agent Tensors ===================================================
        .def("action_tensor", &Manager::actionTensor)
        .def("observation_tensor", &Manager::observationTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("agent_possession_tensor", &Manager::agentPossessionTensor)
        .def("agent_entity_id_tensor", &Manager::agentEntityIDTensor)
        .def("agent_team_tensor", &Manager::agentTeamTensor)
        .def("orientation_tensor", &Manager::orientationTensor)


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
