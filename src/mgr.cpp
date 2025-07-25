#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_cpu.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include "types.hpp"

// ============================== Config Constants ==============================
using namespace madrona;
using namespace madrona::py;

namespace madBasketball {

    struct Manager::Impl 
    {
        Config cfg;
        EpisodeManager *episodeMgr;
        GridState *gridData;

        inline Impl(const Config &c,
                    EpisodeManager *ep_mgr,
                    GridState *grid_data)
            : cfg(c),
            episodeMgr(ep_mgr),
            gridData(grid_data)
        {}

        inline virtual ~Impl() {}

        virtual void run() = 0;
        virtual Tensor exportTensor(ExportID slot, TensorElementType type,
                                    Span<const int64_t> dims) = 0;

        static inline Impl * init(const Config &cfg, const GridState &src_grid);
    };

    struct Manager::CPUImpl final : Manager::Impl 
    {
        using ExecT = TaskGraphExecutor<Engine, Sim, Sim::Config, WorldInit>;
        ExecT cpuExec;

        inline CPUImpl(const Manager::Config &mgr_cfg,
                    const Sim::Config &sim_cfg,
                    EpisodeManager *episode_mgr,
                    GridState *grid_data,
                    WorldInit *world_inits)
            : Impl(mgr_cfg, episode_mgr, grid_data),
            cpuExec({
                    .numWorlds = mgr_cfg.numWorlds,
                    .numExportedBuffers = (uint32_t)ExportID::NumExports,
                }, sim_cfg, world_inits, 1)
        {}

        inline virtual ~CPUImpl() final 
        {
            delete episodeMgr;
            free(gridData);
        }

        inline virtual void run() final { cpuExec.run(); }
        
        inline virtual Tensor exportTensor(ExportID slot,
                                        TensorElementType type,
                                        Span<const int64_t> dims) final
        {
            void *dev_ptr = cpuExec.getExported((uint32_t)slot);
            return Tensor(dev_ptr, type, dims, Optional<int>::none());
        }
    };

    #ifdef MADRONA_CUDA_SUPPORT
    struct Manager::GPUImpl final : Manager::Impl {
        MWCudaExecutor gpuExec;
        MWCudaLaunchGraph stepGraph;

        inline GPUImpl(CUcontext cu_ctx,
                    const Manager::Config &mgr_cfg,
                    const Sim::Config &sim_cfg,
                    EpisodeManager *episode_mgr,
                    GridState *grid_data,
                    WorldInit *world_inits)
            : Impl(mgr_cfg, episode_mgr, grid_data),
            gpuExec({
                    .worldInitPtr = world_inits,
                    .numWorldInitBytes = sizeof(WorldInit),
                    .userConfigPtr = (void *)&sim_cfg,
                    .numUserConfigBytes = sizeof(Sim::Config),
                    .numWorldDataBytes = sizeof(Sim),
                    .worldDataAlignment = alignof(Sim),
                    .numWorlds = mgr_cfg.numWorlds,
                    .numTaskGraphs = 1,
                    .numExportedBuffers = (uint32_t)ExportID::NumExports, 
                }, {
                    { BASKETBALL_SRC_LIST },
                    { BASKETBALL_COMPILE_FLAGS },
                    CompileConfig::OptMode::LTO,
                }, cu_ctx),
            stepGraph(gpuExec.buildLaunchGraph(0))
            
        {}

        inline virtual ~GPUImpl() final {
            REQ_CUDA(cudaFree(episodeMgr));
            REQ_CUDA(cudaFree(gridData));
        }

        inline virtual void run() final { gpuExec.run(stepGraph); }
        
        virtual inline Tensor exportTensor(ExportID slot, TensorElementType type,
                                        Span<const int64_t> dims) final
        {
            void *dev_ptr = gpuExec.getExported((uint32_t)slot);
            return Tensor(dev_ptr, type, dims, cfg.gpuID);
        }
    };
    #endif

    static HeapArray<WorldInit> setupWorldInitData(int64_t num_worlds,
                                                EpisodeManager *episode_mgr,
                                                const GridState *grid)
    {
        HeapArray<WorldInit> world_inits(num_worlds);

        for (int64_t i = 0; i < num_worlds; i++) 
        {
            world_inits[i] = WorldInit {
                episode_mgr,
                grid,
            };
        }

        return world_inits;
    }

    Manager::Impl * Manager::Impl::init(const Config &cfg,
                                        const GridState &src_grid)
    {
        static_assert(sizeof(GridState) % alignof(Cell) == 0);

        Sim::Config sim_cfg {
            .maxEpisodeLength = cfg.maxEpisodeLength,
            .enableViewer = false,
            .initRandKey = rand::initKey(cfg.randSeed),
        };

        switch (cfg.execMode) {
        case ExecMode::CPU: {
            EpisodeManager *episode_mgr = new EpisodeManager { 0 };
            
            uint64_t num_cell_bytes =
                sizeof(Cell) * src_grid.discreteWidth * src_grid.discreteHeight;

            auto *grid_data =
                (char *)malloc(sizeof(GridState) + num_cell_bytes);
            Cell *cpu_cell_data = (Cell *)(grid_data + sizeof(GridState));

        GridState *cpu_grid = (GridState *)grid_data;
        *cpu_grid = GridState {
            .cells = cpu_cell_data,
            .startX = src_grid.startX,
            .startY = src_grid.startY,
            .width = src_grid.width,
            .height = src_grid.height,
            .cellsPerMeter = src_grid.cellsPerMeter,
            .discreteWidth = src_grid.discreteWidth,
            .discreteHeight = src_grid.discreteHeight,
        };

            memcpy(cpu_cell_data, src_grid.cells, num_cell_bytes);

            HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
                episode_mgr, cpu_grid);

            return new CPUImpl(cfg, sim_cfg, episode_mgr, cpu_grid,
                            world_inits.data());
        } break;
        case ExecMode::CUDA: {
    #ifndef MADRONA_CUDA_SUPPORT
            FATAL("CUDA support not compiled in!");
    #else
            CUcontext cu_ctx = MWCudaExecutor::initCUDA(cfg.gpuID);

            EpisodeManager *episode_mgr = 
                (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
            // Set the current episode count to 0
            REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

            uint64_t num_cell_bytes =
                sizeof(Cell) * src_grid.discreteWidth * src_grid.discreteHeight;

            auto *grid_data =
                (char *)cu::allocGPU(sizeof(GridState) + num_cell_bytes);

            Cell *gpu_cell_data = (Cell *)(grid_data + sizeof(GridState));
            GridState grid_staging {
                .cells = gpu_cell_data,
                .startX = src_grid.startX,
                .startY = src_grid.startY,
                .width = src_grid.width,
                .height = src_grid.height,
                .cellsPerMeter = src_grid.cellsPerMeter,
                .discreteWidth = src_grid.discreteWidth,
                .discreteHeight = src_grid.discreteHeight,
            };

            cudaMemcpy(grid_data, &grid_staging, sizeof(GridState),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_cell_data, src_grid.cells, num_cell_bytes,
                    cudaMemcpyHostToDevice);

            GridState *gpu_grid = (GridState *)grid_data;

            HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
                episode_mgr, gpu_grid);

            return new GPUImpl(cu_ctx, cfg, sim_cfg, episode_mgr, gpu_grid,
                            world_inits.data());
    #endif
        } break;
        default: return nullptr;
        }
    }

    Manager::Manager(const Config &cfg,
                    const GridState &src_grid)
        : impl_(Impl::init(cfg, src_grid))
    {}

    Manager::~Manager() {}

    void Manager::step()
    {
        impl_->run();
    }

    // This is the old setAction just in case we need it
    // void Manager::setAction(int32_t world_idx, int32_t agent_idx, int32_t action)
    // {
    //     // Get the action tensor and set the action for the specified world/agent
    //     Tensor action_tensor = actionTensor();
    //     int32_t *action_data = static_cast<int32_t *>(action_tensor.devicePtr());

        
    //     if (world_idx >= 0 && world_idx < impl_->cfg.numWorlds && 
    //         agent_idx >= 0 && agent_idx < NUM_AGENTS) 
    //         {
    //         int32_t index = world_idx * NUM_AGENTS + agent_idx;
    //         action_data[index] = action;
            
    //     } else {
    //         printf("ERROR: Invalid indices! world=%d (max=%d), agent=%d (max=%d)\n",
    //                world_idx, impl_->cfg.numWorlds-1, agent_idx, NUM_AGENTS-1);
    //     }
    // }


    // New more robust setAction, easily adaptable to 3D
    void Manager::setAction(int32_t world_idx, int32_t agent_idx, int32_t move_speed, int32_t move_angle, int32_t rotate, int32_t grab, int32_t pass, int32_t shoot)
    {
        // Get the action tensor and set the action for the specified world/agent
        Tensor action_tensor = actionTensor();
        Action *action_data = static_cast<Action *>(action_tensor.devicePtr());

        
        if (world_idx >= 0 && world_idx < (int32_t)impl_->cfg.numWorlds && 
            agent_idx >= 0 && agent_idx < NUM_AGENTS) 
        {
            int32_t index = world_idx * NUM_AGENTS + agent_idx;
            action_data[index].move = move_speed;
            action_data[index].moveAngle = move_angle;
            action_data[index].rotate = rotate;
            action_data[index].grab = grab;
            action_data[index].pass = pass;
            action_data[index].shoot = shoot; 
        } 
        else 
        {
            printf("ERROR: Invalid indices! world=%d (max=%d), agent=%d (max=%d)\n",
                world_idx, (int32_t)impl_->cfg.numWorlds-1, agent_idx, NUM_AGENTS-1);
        }
    }



    void Manager::triggerReset(int32_t world_idx)
    {
        // Get the reset tensor and trigger reset for ALL agents in the specified world
        Tensor reset_tensor = resetTensor();
        int32_t *reset_data = static_cast<int32_t *>(reset_tensor.devicePtr());
        
        if (world_idx >= 0 && world_idx < (int32_t)impl_->cfg.numWorlds) 
        {
            // Set reset flag for all agents in this world (tensor is numWorlds x NUM_AGENTS x 1)
            for (int32_t agent_idx = 0; agent_idx < NUM_AGENTS; agent_idx++) {
                int32_t global_idx = (world_idx * NUM_AGENTS + agent_idx) * 1; // *1 because Reset has 1 field
                reset_data[global_idx] = 1;
            }
        }
    }



    //=================================================== General Tensors ===================================================

    Tensor Manager::resetTensor() const
    {
        return impl_->exportTensor(ExportID::Reset, TensorElementType::Int32,
                                {impl_->cfg.numWorlds, NUM_AGENTS, 1}); // Reset component has 1 int32_t field
    }

    Tensor Manager::gameStateTensor() const
    {
        return impl_->exportTensor(ExportID::GameState, TensorElementType::Float32,
            {impl_->cfg.numWorlds, 14}); // 14 fields: inboundingInProgress, liveBall, period, teamInPossession, team0Hoop, team0Score, team1Hoop, team1Score, gameClock, shotClock, scoredBaskets, outOfBoundsCount, inboundClock, isOneOnOne
    }


    //=================================================== Agent Tensors ===================================================

    Tensor Manager::actionTensor() const
    {
        return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
            {impl_->cfg.numWorlds, NUM_AGENTS, 6}); // 8 actions: move, moveAngle, rotate, grab, pass, shoot, steal, contest
    }

    Tensor Manager::actionMaskTensor() const
    {
        return impl_->exportTensor(ExportID::ActionMask, TensorElementType::Int32,
            {impl_->cfg.numWorlds, NUM_AGENTS, 4});
    }

    Tensor Manager::agentPosTensor() const
    {
        return impl_->exportTensor(ExportID::AgentPos, TensorElementType::Float32,
            {impl_->cfg.numWorlds, NUM_AGENTS, 3});
    }

    Tensor Manager::observationsTensor() const
    {
        return impl_->exportTensor(ExportID::Observations, TensorElementType::Float32,
            {impl_->cfg.numWorlds, NUM_AGENTS, sizeof(Observations) / sizeof(float)});
    }


    Tensor Manager::orientationTensor() const
    {
        return impl_->exportTensor(ExportID::Orientation, TensorElementType::Float32,
            {impl_->cfg.numWorlds, NUM_AGENTS, 4}); // 4 entries of eahc Quaternion
    }


    Tensor Manager::rewardTensor() const
    {
        return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
            {impl_->cfg.numWorlds, NUM_AGENTS});
    }


    Tensor Manager::doneTensor() const
    {
        return impl_->exportTensor(ExportID::Done, TensorElementType::Float32,
            {impl_->cfg.numWorlds, NUM_AGENTS});
    }


    Tensor Manager::agentPossessionTensor() const
    {
        return impl_->exportTensor(ExportID::AgentPossession, TensorElementType::Int32,
            {impl_->cfg.numWorlds, NUM_AGENTS, 3});  // hasBall, ballEntityID, pointsWorth
    }


    Tensor Manager::agentEntityIDTensor() const
    {
        return impl_->exportTensor(ExportID::AgentEntityID, TensorElementType::Int32,
            {impl_->cfg.numWorlds, NUM_AGENTS});
    }


    Tensor Manager::agentTeamTensor() const
    {
        return impl_->exportTensor(ExportID::TeamData, TensorElementType::Int32,
            {impl_->cfg.numWorlds, NUM_AGENTS, 5});  // teamIndex, teamColor (3 components), defendingHoopID
    }


    Tensor Manager::agentStatsTensor() const
    {
        return impl_->exportTensor(ExportID::AgentStats, TensorElementType::Int32,
            {impl_->cfg.numWorlds, NUM_AGENTS, 2});  // points and fouls
    }

    //=================================================== Basketball Tensors ===================================================
    Tensor Manager::basketballPosTensor() const
    {
        return impl_->exportTensor(ExportID::BasketballPos, TensorElementType::Float32,
            {impl_->cfg.numWorlds, NUM_BASKETBALLS, 3});
    }


    Tensor Manager::ballPhysicsTensor() const
    {
        return impl_->exportTensor(ExportID::BallPhysicsData, TensorElementType::Int32,
            {impl_->cfg.numWorlds, NUM_BASKETBALLS, 7});
    }


    Tensor Manager::ballGrabbedTensor() const
    {
        return impl_->exportTensor(ExportID::BallGrabbed, TensorElementType::Int32,
            {impl_->cfg.numWorlds, NUM_BASKETBALLS, 2});  // isGrabbed, holderEntityID
    }


    Tensor Manager::ballEntityIDTensor() const
    {
        return impl_->exportTensor(ExportID::BallEntityID, TensorElementType::Int32,
            {impl_->cfg.numWorlds, NUM_BASKETBALLS});
    }


    //=================================================== Hoop Tensors ===================================================
    Tensor Manager::hoopPosTensor() const
    {
        return impl_->exportTensor(ExportID::HoopPos, TensorElementType::Float32,
            {impl_->cfg.numWorlds, NUM_HOOPS, 3});
    }
}