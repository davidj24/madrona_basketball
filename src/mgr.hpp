#pragma once
#ifdef gridworld_madrona_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include "grid.hpp"

namespace madsimple {

class Manager {
public:
    struct Config {
        uint32_t maxEpisodeLength;
        madrona::ExecMode execMode;
        uint32_t numWorlds;
        int gpuID;
    };

    MGR_EXPORT Manager(const Config &cfg, const GridState &src_grid);
    MGR_EXPORT ~Manager();

    MGR_EXPORT void step();

    // Input injection interface (following escape room pattern)
    MGR_EXPORT void setAction(int32_t world_idx, int32_t agent_idx, 
                             int32_t move_speed, int32_t move_angle, 
                             int32_t rotate, int32_t grab, int32_t pass);
    MGR_EXPORT void triggerReset(int32_t world_idx);

    // State extraction interface (tensor outputs)
    MGR_EXPORT madrona::py::Tensor resetTensor() const;
    MGR_EXPORT madrona::py::Tensor actionTensor() const;
    MGR_EXPORT madrona::py::Tensor observationTensor() const;
    MGR_EXPORT madrona::py::Tensor rewardTensor() const;
    MGR_EXPORT madrona::py::Tensor doneTensor() const;
    MGR_EXPORT madrona::py::Tensor basketballPosTensor() const;
    MGR_EXPORT madrona::py::Tensor hoopPosTensor() const;

    

private:
    struct Impl;
    struct CPUImpl;
    struct GPUImpl;

    std::unique_ptr<Impl> impl_;
};

}
