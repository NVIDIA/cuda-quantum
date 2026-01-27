# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_filename_component(CUDAQ_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

include(CMakeFindDependencyMacro)
list(APPEND CMAKE_MODULE_PATH "${CUDAQ_CMAKE_DIR}")

set (CUDAQOperator_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQOperator REQUIRED)

set (CUDAQCommon_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQCommon REQUIRED)

set (CUDAQEmDefault_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQEmDefault REQUIRED)

set (CUDAQPlatformDefault_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQPlatformDefault REQUIRED)

set (CUDAQNlopt_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQNlopt REQUIRED)

set (CUDAQEnsmallen_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQEnsmallen REQUIRED)

set (CUDAQPythonInterop_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQPythonInterop REQUIRED)

get_filename_component(PARENT_DIRECTORY ${CUDAQ_CMAKE_DIR} DIRECTORY)
get_filename_component(CUDAQ_LIBRARY_DIR ${PARENT_DIRECTORY} DIRECTORY)
get_filename_component(CUDAQ_INSTALL_DIR ${CUDAQ_LIBRARY_DIR} DIRECTORY)
set(CUDAQ_INCLUDE_DIR ${CUDAQ_INSTALL_DIR}/include)

set (NVQIR_DIR "${PARENT_DIRECTORY}/nvqir")
find_dependency(NVQIR REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if (NOT CUDAQ_LIBRARY_MODE)
  enable_language(CUDAQ)
endif() 

# ---- TARGET EXPORTS ----
# Prefer cusvsim libraries if they are present
set (__base_nvtarget_name "custatevec") 
find_library(CUDAQ_CUSVSIM_PATH NAMES cusvsim-fp32 HINTS ${CUDAQ_LIBRARY_DIR})
if (CUDAQ_CUSVSIM_PATH)
  set(__base_nvtarget_name "cusvsim")
endif() 

# Default Target
add_library(cudaq::cudaq-default-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-default-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-${__base_nvtarget_name}-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-${__base_nvtarget_name}-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

# NVIDIA Target
add_library(cudaq::cudaq-nvidia-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-nvidia-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-${__base_nvtarget_name}-fp32${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-${__base_nvtarget_name}-fp32${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

# NVIDIA FP64 Target
add_library(cudaq::cudaq-nvidia-fp64-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-nvidia-fp64-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-${__base_nvtarget_name}-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-${__base_nvtarget_name}-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

# NVIDIA MGPU Target
add_library(cudaq::cudaq-nvidia-mgpu-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-nvidia-mgpu-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-mgpu-fp32${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-mgpu-fp32${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

# NVIDIA MGPU-FP64 Target
add_library(cudaq::cudaq-nvidia-mgpu-fp64-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-nvidia-mgpu-fp64-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-mgpu-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-mgpu-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

# NVIDIA MQPU Target
add_library(cudaq::cudaq-nvidia-mqpu-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-nvidia-mqpu-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-${__base_nvtarget_name}${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-${__base_nvtarget_name}${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-mqpu;cudaq::cudaq-em-default")

# NVIDIA MQPU FP64 Target
add_library(cudaq::cudaq-nvidia-mqpu-fp64-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-nvidia-mqpu-fp64-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-${__base_nvtarget_name}-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-${__base_nvtarget_name}-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-mqpu;cudaq::cudaq-em-default")

# QPP CPU Target
add_library(cudaq::cudaq-qpp-cpu-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-qpp-cpu-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-qpp${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-qpp${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

# QPP CPU DensityMatrix Target
add_library(cudaq::cudaq-qpp-density-matrix-cpu-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-qpp-density-matrix-cpu-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-dm${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-dm${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

# Stim Target
add_library(cudaq::cudaq-stim-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-stim-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-stim${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-stim${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")
# -------------------------

if(NOT TARGET cudaq::cudaq)
    include("${CUDAQ_CMAKE_DIR}/CUDAQTargets.cmake")
endif()

function(cudaq_set_target TARGETNAME)
  message(STATUS "CUDA Quantum Target = ${TARGETNAME}")
  target_link_libraries(cudaq::cudaq INTERFACE cudaq::cudaq-${TARGETNAME}-target)
endfunction()

add_library(cudaq::cudaq-builder SHARED IMPORTED)
set_target_properties(cudaq::cudaq-builder PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libcudaq-builder${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libcudaq-builder${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Check for the presence of NVIDIA GPUs, if none
# are found set the default target to qpp-cpu
set(__tmp_cudaq_target "qpp-cpu")
find_program(NVIDIA_SMI "nvidia-smi")
if(NVIDIA_SMI)
  execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (${NGPUS} GREATER_EQUAL 1)
    # We have NVIDIA GPUs, set the NVIDIA target as the default
    message(STATUS "Number of NVIDIA GPUs detected: ${NGPUS}")
    set(__tmp_cudaq_target "nvidia")
  endif()
endif()

set(CUDAQ_TARGET ${__tmp_cudaq_target} CACHE STRING "The CUDA Quantum target to compile for and execute on. Defaults to `${__tmp_cudaq_target}`")
cudaq_set_target(${CUDAQ_TARGET})
