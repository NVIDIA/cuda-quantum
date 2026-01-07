# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-em-photonics)

add_library(${LIBRARY_NAME} SHARED PhotonicsExecutionManager.cpp)

target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
       $<INSTALL_INTERFACE:include>)

set (PHOTONICS_DEPENDENCIES "")
list(APPEND PHOTONICS_DEPENDENCIES cudaq-common libqpp fmt::fmt-header-only)
add_openmp_configurations(${LIBRARY_NAME} PHOTONICS_DEPENDENCIES)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC cudaq-operator
  PRIVATE ${PHOTONICS_DEPENDENCIES}
)

install(TARGETS ${LIBRARY_NAME}
  EXPORT cudaq-em-photonics-targets
  DESTINATION lib)

install(EXPORT cudaq-em-photonics-targets
        FILE CUDAQEmPhotonicsTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)

add_target_config(orca-photonics)
