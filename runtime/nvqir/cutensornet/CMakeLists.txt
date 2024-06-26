# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Find CUDA Toolkit for CUDA libs, e.g., cudart.
find_package(CUDAToolkit REQUIRED)

find_library(CUTENSOR_LIB
    NAMES   cutensor
    HINTS   
        ${CUTENSOR_ROOT}/lib64
        ${CUTENSOR_ROOT}/lib
        ${CUTENSOR_ROOT}/lib64/${CUDA_VERSION_MAJOR}
        ${CUTENSOR_ROOT}/lib/${CUDA_VERSION_MAJOR}
)

find_library(CUTENSORNET_LIB
    NAMES   cutensornet
    HINTS   
        ${CUTENSORNET_ROOT}/lib64
        ${CUTENSORNET_ROOT}/lib
        ${CUTENSORNET_ROOT}/lib64/${CUDA_VERSION_MAJOR}
        ${CUTENSORNET_ROOT}/lib/${CUDA_VERSION_MAJOR}
)

find_file(CUTENSORNET_INC
    NAMES   cutensornet.h
    HINTS   
        ${CUTENSORNET_ROOT}/include      
        /usr/include    
        ENV CPATH
)

if(NOT CUTENSOR_LIB)
  message(FATAL_ERROR "\nUnable to find cutensor installation. Please ensure it is correctly installed and set and define CUTENSOR_ROOT if necessary (currently set to: ${CUTENSOR_ROOT}).")
endif()
message(STATUS "CUTENSOR_LIB: ${CUTENSOR_LIB}")

if(NOT CUTENSORNET_LIB OR NOT CUTENSORNET_INC)
  message(FATAL_ERROR "\nUnable to find the cutensornet installation. Please ensure it is correctly installed and define CUTENSORNET_ROOT if necessary (currently set to: ${CUTENSORNET_ROOT}).")
endif()
message(STATUS "CUTENSORNET_LIB: ${CUTENSORNET_LIB}")
message(STATUS "CUTENSORNET_INC: ${CUTENSORNET_INC}")

# Determine cutensornet version
file(READ "${CUTENSORNET_INC}" cutensornet_header)
string(REGEX MATCH "CUTENSORNET_MAJOR ([0-9]*)" _ ${cutensornet_header})
set(CUTENSORNET_MAJOR ${CMAKE_MATCH_1})

string(REGEX MATCH "CUTENSORNET_MINOR ([0-9]*)" _ ${cutensornet_header})
set(CUTENSORNET_MINOR ${CMAKE_MATCH_1})

string(REGEX MATCH "CUTENSORNET_PATCH ([0-9]*)" _ ${cutensornet_header})
set(CUTENSORNET_PATCH ${CMAKE_MATCH_1})

set(CUTENSORNET_VERSION ${CUTENSORNET_MAJOR}.${CUTENSORNET_MINOR}.${CUTENSORNET_PATCH})
message(STATUS "Found cutensornet version: ${CUTENSORNET_VERSION}")
# We need cutensornet v2.3+
if (${CUTENSORNET_VERSION} VERSION_GREATER_EQUAL "2.3")
  set (BASE_TENSOR_BACKEND_SRS  
        simulator_cutensornet.cpp 
        tensornet_spin_op.cpp
        tensornet_state.cpp)
  get_filename_component(CUTENSORNET_INCLUDE_DIR ${CUTENSORNET_INC} DIRECTORY)
  get_filename_component(CUTENSORNET_LIB_DIR ${CUTENSORNET_LIB} DIRECTORY)
  get_filename_component(CUTENSOR_LIB_DIR ${CUTENSOR_LIB} DIRECTORY)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${CUTENSORNET_LIB_DIR}:${CUTENSOR_LIB_DIR}")

  # Helper macro to add cutensornet-based backends
  macro (nvqir_create_cutn_plugin LIBRARY_NAME)
    # This will create a target named ${LIBRARY_NAME}
    add_library(nvqir-${LIBRARY_NAME} SHARED ${ARGN})
    target_include_directories(nvqir-${LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/runtime/common ${CMAKE_SOURCE_DIR}/runtime/nvqir ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUTENSORNET_INCLUDE_DIR})
    target_link_libraries(nvqir-${LIBRARY_NAME} PRIVATE fmt::fmt-header-only cudaq cudaq-common ${CUTENSORNET_LIB} ${CUTENSOR_LIB} CUDA::cudart)
    install(TARGETS nvqir-${LIBRARY_NAME} DESTINATION lib)
    file (WRITE ${CMAKE_BINARY_DIR}/targets/${LIBRARY_NAME}.config "NVQIR_SIMULATION_BACKEND=${LIBRARY_NAME}\nGPU_REQUIREMENTS=\"true\"\nPREPROCESSOR_DEFINES=\"\${PREPROCESSOR_DEFINES} -D CUDAQ_SIMULATION_SCALAR_FP64\"\n")
    install(FILES ${CMAKE_BINARY_DIR}/targets/${LIBRARY_NAME}.config DESTINATION targets)
  endmacro()

  nvqir_create_cutn_plugin(tensornet ${BASE_TENSOR_BACKEND_SRS} simulator_tensornet_register.cpp tn_simulation_state.cpp)
  nvqir_create_cutn_plugin(tensornet-mps ${BASE_TENSOR_BACKEND_SRS} simulator_mps_register.cpp mps_simulation_state.cpp)
  add_library(tensornet-mpi-util OBJECT mpi_support.cpp)
  target_include_directories(tensornet-mpi-util PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUTENSORNET_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/runtime)
  target_link_libraries(tensornet-mpi-util PRIVATE cudaq-common fmt::fmt-header-only)
  # Note: only tensornet backend supports MPI at cutensornet level (distributed tensor computation)
  target_link_libraries(nvqir-tensornet PRIVATE tensornet-mpi-util)
  
  # Check if the cutensornet plugin lib for expectation value is provided
  if(NOT CUDAQ_CUTENSORNET_PLUGIN_LIB AND EXISTS "$ENV{CUDAQ_CUTENSORNET_PLUGIN_LIB}")
    set(CUDAQ_CUTENSORNET_PLUGIN_LIB "$ENV{CUDAQ_CUTENSORNET_PLUGIN_LIB}")
  endif()
  if (EXISTS "${CUDAQ_CUTENSORNET_PLUGIN_LIB}")
    message(STATUS "CUDA Quantum tensornet backends: link additional external plugin ${CUDAQ_CUTENSORNET_PLUGIN_LIB}")
    target_link_libraries(nvqir-tensornet PRIVATE -Wl,--whole-archive ${CUDAQ_CUTENSORNET_PLUGIN_LIB} -Wl,--no-whole-archive)
    target_link_libraries(nvqir-tensornet-mps PRIVATE -Wl,--whole-archive ${CUDAQ_CUTENSORNET_PLUGIN_LIB} -Wl,--no-whole-archive)
  endif()
else()
  message(WARNING "Skipped tensornet backend due to incompatible cutensornet version. Please install cutensornet v2.3.0+.")
endif()
