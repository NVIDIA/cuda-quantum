# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)
set(LIBRARY_NAME nvqir-dynamics)
message (STATUS "CUDENSITYMAT_ROOT and CUDA_FOUND - building dynamics NVQIR backend.")

find_library(CUTENSOR_LIB
    NAMES   cutensor libcutensor.so.2
    HINTS   
        ${CUTENSOR_ROOT}/lib64
        ${CUTENSOR_ROOT}/lib
        ${CUTENSOR_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
        ${CUTENSOR_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}
)

find_library(CUTENSORNET_LIB
    NAMES   cutensornet libcutensornet.so.2
    HINTS   
        ${CUTENSORNET_ROOT}/lib64
        ${CUTENSORNET_ROOT}/lib
        ${CUTENSORNET_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
        ${CUTENSORNET_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}
)

# Find cudensitymat header (for MPI plugin interface typedef)
find_file(CUDENSITYMAT_INC
    NAMES   cudensitymat.h
    HINTS   
        $ENV{CUQUANTUM_INSTALL_PREFIX}/include      
        /usr/include    
        ENV CPATH
    REQUIRED
)

find_library(CUDENSITYMAT_LIB
    NAMES cudensitymat libcudensitymat.so.0
    HINTS
    ${CUDENSITYMAT_ROOT}/lib64
    ${CUDENSITYMAT_ROOT}/lib
    ${CUDENSITYMAT_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
    ${CUDENSITYMAT_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR})

if(NOT CUTENSOR_LIB)
  message(FATAL_ERROR "\nUnable to find cutensor installation. Please ensure it is correctly installed and set and define CUTENSOR_ROOT if necessary (currently set to: ${CUTENSOR_ROOT}).")
endif()
message(STATUS "CUTENSOR_LIB: ${CUTENSOR_LIB}")

if(NOT CUDENSITYMAT_LIB OR NOT CUDENSITYMAT_INC)
  message(FATAL_ERROR "\nUnable to find cudensitymat installation. Please ensure it is correctly installed and set and define CUDENSITYMAT_ROOT if necessary (currently set to: ${CUDENSITYMAT_ROOT}).")
endif()
message(STATUS "CUDENSITYMAT_INC: ${CUDENSITYMAT_INC}")
message(STATUS "CUDENSITYMAT_LIB: ${CUDENSITYMAT_LIB}")

# Determine cudensitymat version
file(READ "${CUDENSITYMAT_INC}" cudensitymat_header)
string(REGEX MATCH "CUDENSITYMAT_MAJOR ([0-9]*)" _ ${cudensitymat_header})
set(CUDENSITYMAT_MAJOR ${CMAKE_MATCH_1})

string(REGEX MATCH "CUDENSITYMAT_MINOR ([0-9]*)" _ ${cudensitymat_header})
set(CUDENSITYMAT_MINOR ${CMAKE_MATCH_1})

string(REGEX MATCH "CUDENSITYMAT_PATCH ([0-9]*)" _ ${cudensitymat_header})
set(CUDENSITYMAT_PATCH ${CMAKE_MATCH_1})

set(CUDENSITYMAT_VERSION ${CUDENSITYMAT_MAJOR}.${CUDENSITYMAT_MINOR}.${CUDENSITYMAT_PATCH})
message(STATUS "Found cudensitymat version: ${CUDENSITYMAT_VERSION}")

# We need cudensitymat v0.2.0 
if (${CUDENSITYMAT_VERSION} VERSION_GREATER_EQUAL "0.2")

  get_filename_component(CUDENSITYMAT_INCLUDE_DIR ${CUDENSITYMAT_INC} DIRECTORY)
  get_filename_component(CUDENSITYMAT_LIB_DIR ${CUDENSITYMAT_LIB} DIRECTORY)
  get_filename_component(CUTENSOR_LIB_DIR ${CUTENSOR_LIB} DIRECTORY)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${CUDENSITYMAT_LIB_DIR}:${CUTENSOR_LIB_DIR}")

  add_library(${LIBRARY_NAME} SHARED
    CuDensityMatSim.cpp
    mpi_support.cpp
    CuDensityMatTimeStepper.cpp
    RungeKuttaIntegrator.cpp
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
      ${CUDENSITYMAT_INCLUDE_DIR}
  )

  target_include_directories(${LIBRARY_NAME}
    PUBLIC
      . ..
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CMAKE_SOURCE_DIR}/runtime/common
      ${CUDENSITYMAT_INCLUDE_DIR}
  )

  target_link_libraries(${LIBRARY_NAME}
                        PRIVATE
                          fmt::fmt-header-only
                          cudaq-common
                          ${CUDENSITYMAT_LIB}
                          ${CUTENSOR_LIB}
                          ${CUTENSORNET_LIB}
                          CUDA::cudart_static
                      )
  target_link_libraries(${LIBRARY_NAME}
                      PUBLIC
                        cudaq-operator
                        ${CUDENSITYMAT_LIB}
                      )
  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
  add_target_config(dynamics)
else()
  message(FATAL_ERROR "Cudensitymat version ${CUDENSITYMAT_VERSION} is not supported. Please use version 0.2.0 or higher.")
endif()