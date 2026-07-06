# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

message (STATUS "cuStateVec_FOUND - building custatevec NVQIR backend.")

# Ensure the cuStateVec library directory is on the install RPATH.
get_target_property(_cusv_loc cuStateVec::cuStateVec IMPORTED_LOCATION)
if(_cusv_loc)
  get_filename_component(_cusv_dir "${_cusv_loc}" DIRECTORY)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_cusv_dir}")
endif()

add_library(nvqir-custatevec-components STATIC
  CuStateVecBatch.cpp
  CuStateVecCommunicator.cpp
  CuStateVecDevice.cpp
  CuStateVecConfig.cpp
  CuStateVecGateEngine.cpp
  CuStateVecRandom.cpp
  CuStateVecSimulationState.cpp
  CuStateVecState.cpp)
set_target_properties(nvqir-custatevec-components PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvqir-custatevec-components
  PUBLIC
    .
    ..
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(nvqir-custatevec-components
  PUBLIC
    cuStateVec::cuStateVec
    CUDA::cublas
    CUDA::curand
    CUDA::cudart_static)

# Build the external communicator module once and embed it into both MGPU
# simulator libraries. This is a build-time object target, not an installed
# runtime library.
add_library(nvqir-custatevec-communicator-obj OBJECT
  CuStateVecCommunicatorPlugin.cpp)
set_target_properties(nvqir-custatevec-communicator-obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvqir-custatevec-communicator-obj PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime
  ${CMAKE_SOURCE_DIR}/runtime/include
  ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(nvqir-custatevec-communicator-obj PRIVATE
  cuStateVec::cuStateVec)

macro (nvqir_create_cusv_plugin LIBRARY_NAME SOURCE_FILENAME)
  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILENAME})

  target_include_directories(${LIBRARY_NAME}
    PRIVATE
      . ..
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CUDAToolkit_INCLUDE_DIRS}/cccl
      ${CMAKE_SOURCE_DIR}/runtime/common)

  # Prevent libc++ from defining the conflicting new/delete helpers
  target_compile_definitions(${LIBRARY_NAME} PRIVATE
    _LIBCPP_DISABLE_NEW_DELETE_DEFINITIONS
  )

  # Tell clang not to emit aligned new/delete operators
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${LIBRARY_NAME} PRIVATE -fno-aligned-allocation)
  endif()

  target_link_libraries(${LIBRARY_NAME}
                        PRIVATE
                          cudaq-logger
                          cudaq-common
                          nvqir-custatevec-components
                          cuStateVec::cuStateVec
                          CUDA::cudart_static
                          CUDA::cublas)
  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
endmacro()

# Generate a cuStateVec plugin source from the CuStateVecCircuitSimulator.cpp.in
# template and build it into a shared library. The template placeholders are
# filled from the arguments:
#   LIBRARY_NAME     - name of the resulting shared library target
#   SOURCE_FILENAME  - name of the generated .cpp file (in the build directory)
#   SIMULATOR_HEADER - header declaring the simulator class (CUSV_SIMULATOR_HEADER)
#   SIMULATOR_CLASS  - simulator class to instantiate (CUSV_SIMULATOR_CLASS)
#   SCALAR_TYPE      - floating-point scalar type, e.g. double or float (CUSV_SCALAR_TYPE)
#   REGISTER_NAME    - NVQIR backend registration name (CUSV_REGISTER_NAME)
function(nvqir_create_configured_cusv_plugin
         LIBRARY_NAME SOURCE_FILENAME SIMULATOR_HEADER SIMULATOR_CLASS
         SCALAR_TYPE REGISTER_NAME)
  set(CUSV_SIMULATOR_HEADER ${SIMULATOR_HEADER})
  set(CUSV_SIMULATOR_CLASS ${SIMULATOR_CLASS})
  set(CUSV_SCALAR_TYPE ${SCALAR_TYPE})
  set(CUSV_REGISTER_NAME ${REGISTER_NAME})
  set(generated_source "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILENAME}")
  configure_file(CuStateVecCircuitSimulator.cpp.in ${generated_source} @ONLY)
  nvqir_create_cusv_plugin(${LIBRARY_NAME} ${generated_source})

  # Select the scalar-precision compile definition from the scalar type.
  if(CUSV_SCALAR_TYPE STREQUAL "double")
    target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDAQ_SIMULATION_SCALAR_FP64)
  elseif(CUSV_SCALAR_TYPE STREQUAL "float")
    target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDAQ_SIMULATION_SCALAR_FP32)
  else()
    message(FATAL_ERROR "Unsupported cuStateVec scalar type: ${CUSV_SCALAR_TYPE}")
  endif()
endfunction()

# Single-GPU cuStateVec simulator, double precision.
nvqir_create_configured_cusv_plugin(
  nvqir-custatevec-fp64 CuStateVecCircuitSimulator.cpp
  CuStateVecCircuitSimulatorEx.h CuStateVecCircuitSimulator double
  custatevec_fp64)
# Single-GPU cuStateVec simulator, single precision.
nvqir_create_configured_cusv_plugin(
  nvqir-custatevec-fp32 CuStateVecCircuitSimulatorF32.cpp
  CuStateVecCircuitSimulatorEx.h CuStateVecCircuitSimulator float
  custatevec_fp32)
# Multi-GPU multi-node (MPI-distributed) cuStateVec simulator, double precision.
nvqir_create_configured_cusv_plugin(
  nvqir-nvidia-mgpu CuStateVecCircuitSimulatorMgpu.cpp
  CuStateVecMpiCircuitSimulator.h CuStateVecMpiCircuitSimulator double
  nvidia_mgpu)
# Multi-GPU multi-node (MPI-distributed) cuStateVec simulator, single precision.
nvqir_create_configured_cusv_plugin(
  nvqir-nvidia-mgpu-fp32 CuStateVecCircuitSimulatorMgpuF32.cpp
  CuStateVecMpiCircuitSimulator.h CuStateVecMpiCircuitSimulator float
  nvidia_mgpu_fp32)
# The multi-GPU multi-node simulators run distributed, so they need MPI plugin
# support. Only these libraries embed the communicator bridge and link cudaq;
# the single-GPU simulators above are non-distributed and require neither.

# Embed the communicator bridge (CuStateVecCommunicatorPlugin.cpp), which adapts
# the cuStateVecEx communicator ABI to the CUDA-Q distributed MPI plugin.
target_sources(nvqir-nvidia-mgpu PRIVATE
  $<TARGET_OBJECTS:nvqir-custatevec-communicator-obj>)
target_sources(nvqir-nvidia-mgpu-fp32 PRIVATE
  $<TARGET_OBJECTS:nvqir-custatevec-communicator-obj>)
# Link cudaq for the distributed MPI runtime (cudaq::mpi::getMpiPlugin, rank,
# etc.) used by the communicator bridge and the MPI simulator.
target_link_libraries(nvqir-nvidia-mgpu PRIVATE cudaq)
target_link_libraries(nvqir-nvidia-mgpu-fp32 PRIVATE cudaq)
