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

namespace madBasketball {

class Manager {
public:
    struct Config {
        uint32_t randSeed;
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
                             int32_t rotate, int32_t grab, int32_t pass, int32_t shoot);
    MGR_EXPORT void triggerReset(int32_t world_idx);




    //=================================================== General Tensors ===================================================
    MGR_EXPORT madrona::py::Tensor resetTensor() const;
    MGR_EXPORT madrona::py::Tensor gameStateTensor() const;


    //=================================================== Agent Tensors ===================================================
    MGR_EXPORT madrona::py::Tensor actionTensor() const;
    MGR_EXPORT madrona::py::Tensor actionMaskTensor() const;
    MGR_EXPORT madrona::py::Tensor agentPosTensor() const;
    MGR_EXPORT madrona::py::Tensor observationsTensor() const;
    MGR_EXPORT madrona::py::Tensor orientationTensor() const;
    MGR_EXPORT madrona::py::Tensor agentTeamTensor() const;
    MGR_EXPORT madrona::py::Tensor rewardTensor() const;
    MGR_EXPORT madrona::py::Tensor doneTensor() const;
    MGR_EXPORT madrona::py::Tensor agentPossessionTensor() const;
    MGR_EXPORT madrona::py::Tensor agentEntityIDTensor() const;
    MGR_EXPORT madrona::py::Tensor agentStatsTensor() const;
    


    //=================================================== Basketball Tensors ===================================================
    MGR_EXPORT madrona::py::Tensor basketballPosTensor() const;
    MGR_EXPORT madrona::py::Tensor ballPhysicsTensor() const;
    MGR_EXPORT madrona::py::Tensor ballGrabbedTensor() const;
    MGR_EXPORT madrona::py::Tensor ballEntityIDTensor() const;
    MGR_EXPORT madrona::py::Tensor ballVelocityTensor() const;

    
    
    //=================================================== Hoop Tensors ===================================================
    MGR_EXPORT madrona::py::Tensor hoopPosTensor() const;
    


private:
    struct Impl;
    struct CPUImpl;
    struct GPUImpl;

    std::unique_ptr<Impl> impl_;
};

}
