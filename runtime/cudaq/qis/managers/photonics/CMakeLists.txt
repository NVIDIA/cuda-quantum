# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
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

find_package(OpenMP)

if (OpenMP_CXX_FOUND)
  target_link_libraries(${LIBRARY_NAME}
    PUBLIC cudaq-spin
    PRIVATE cudaq-common libqpp fmt::fmt-header-only OpenMP::OpenMP_CXX
  )
  target_compile_definitions(${LIBRARY_NAME} PRIVATE -DHAS_OPENMP=1)
else()
  target_link_libraries(${LIBRARY_NAME}
    PUBLIC cudaq-spin
    PRIVATE cudaq-common libqpp fmt::fmt-header-only)
endif()

install(TARGETS ${LIBRARY_NAME}
  EXPORT cudaq-em-photonics-targets
  DESTINATION lib)

install(EXPORT cudaq-em-photonics-targets
        FILE CUDAQEmPhotonicsTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)

add_target_config(photonics)
