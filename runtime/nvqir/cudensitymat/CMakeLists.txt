# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME nvqir-dynamics)
message (STATUS "cuDensityMat_FOUND - building dynamics NVQIR backend.")
message(STATUS "Found cudensitymat version: ${cuDensityMat_VERSION}")

# Ensure cuDensityMat and cuTensor library directories are on the install RPATH.
get_target_property(_cudm_loc cuDensityMat::cuDensityMat IMPORTED_LOCATION)
get_target_property(_cut_loc cuTensor::cuTensor IMPORTED_LOCATION)
if(_cudm_loc)
  get_filename_component(_cudm_dir "${_cudm_loc}" DIRECTORY)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_cudm_dir}")
endif()
if(_cut_loc)
  get_filename_component(_cut_dir "${_cut_loc}" DIRECTORY)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_cut_dir}")
endif()

# We need cudensitymat v0.4.0
if (${cuDensityMat_VERSION} VERSION_GREATER_EQUAL "0.4")

  add_library(${LIBRARY_NAME} SHARED
    CuDensityMatSim.cpp
    mpi_support.cpp
    CuDensityMatTimeStepper.cpp
    RungeKuttaIntegrator.cpp
    CrankNicolsonIntegrator.cpp
    MagnusIntegrator.cpp
    CuDensityMatExpectation.cpp
    CuDensityMatEvolution.cpp
    CuDensityMatState.cpp
    CuDensityMatContext.cpp
    CuDensityMatOpConverter.cpp
    CuDensityMatCallbackConverter.cpp
    CuDensityMatSuperOpCtor.cpp
    CuDensityMatUtils.cpp
  )

  message("CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
  target_include_directories(${LIBRARY_NAME}
    PRIVATE 
      . .. 
      ${CUDAToolkit_INCLUDE_DIRS} 
      ${CMAKE_SOURCE_DIR}/runtime/common
  )

  target_include_directories(${LIBRARY_NAME}
    PUBLIC
      . ..
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CMAKE_SOURCE_DIR}/runtime/common
  )

  target_link_libraries(${LIBRARY_NAME}
                        PRIVATE
                          fmt::fmt-header-only 
                          cudaq-logger
                          cudaq-common
                          cuDensityMat::cuDensityMat
                          cuTensor::cuTensor
                          cuTensorNet::cuTensorNet
                          CUDA::cudart_static
                      )
  target_link_libraries(${LIBRARY_NAME}
                      PUBLIC
                        cudaq-operator
                        cuDensityMat::cuDensityMat
                      )
  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
  add_target_config(dynamics)
else()
  message(FATAL_ERROR "cuDensityMat version ${cuDensityMat_VERSION} is not supported. Please use version 0.4.0 or higher.")
endif()
