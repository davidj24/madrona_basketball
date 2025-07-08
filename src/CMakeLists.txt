set(SIMULATOR_SRCS
    types.hpp sim.hpp sim.cpp
)

add_library(madrona_basketball_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(madrona_basketball_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
)

add_library(madrona_basketball_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(madrona_basketball_mgr PRIVATE
    madrona_python_utils
    madrona_common
    madrona_importer
    madrona_mw_cpu
    madrona_basketball_cpu_impl
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            madrona_basketball_gpu_srcs
        SOURCES_DEFN
            BASKETBALL_SRC_LIST
        FLAGS_DEFN
            BASKETBALL_COMPILE_FLAGS
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(madrona_basketball_mgr PRIVATE
        madrona_basketball_gpu_srcs
        madrona_mw_gpu
    )
endif()

target_compile_definitions(madrona_basketball_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(madrona_basketball
    bindings.cpp
)

target_link_libraries(madrona_basketball PRIVATE
    madrona_hdrs
    madrona_basketball_mgr
)
