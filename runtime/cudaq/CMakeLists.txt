# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(spin)

set(LIBRARY_NAME cudaq)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

# Create the CUDA Quantum Library
add_library(${LIBRARY_NAME} 
         SHARED cudaq.cpp 
                algorithms/state.cpp
                platform/quantum_platform.cpp 
                utils/cudaq_utils.cpp)

target_include_directories(${LIBRARY_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
           $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
  PRIVATE .)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC dl cudaq-spin cudaq-common cudaq-nlopt cudaq-ensmallen
  PRIVATE fmt::fmt-header-only)

cudaq_library_set_rpath(${LIBRARY_NAME})

add_subdirectory(qis/managers)
add_subdirectory(algorithms)
add_subdirectory(platform)
add_subdirectory(builder)

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-targets DESTINATION lib)

install(EXPORT cudaq-targets
        FILE CUDAQTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
