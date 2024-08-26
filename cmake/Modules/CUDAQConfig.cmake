# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_filename_component(CUDAQ_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

include(CMakeFindDependencyMacro)
list(APPEND CMAKE_MODULE_PATH "${CUDAQ_CMAKE_DIR}")

set (CUDAQSpin_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQSpin REQUIRED)

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
add_library(cudaq::cudaq-nvidia-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-nvidia-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-custatevec-fp32${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-custatevec-fp32${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

add_library(cudaq::cudaq-nvidia-fp64-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-nvidia-fp64-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-custatevec-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-custatevec-fp64${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_LINK_INTERFACE_LIBRARIES "cudaq::cudaq-platform-default;cudaq::cudaq-em-default")

add_library(cudaq::cudaq-qpp-cpu-target SHARED IMPORTED)
set_target_properties(cudaq::cudaq-qpp-cpu-target PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libnvqir-qpp${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libnvqir-qpp${CMAKE_SHARED_LIBRARY_SUFFIX}"
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
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
  set(CUDA_TEST_SOURCE "
    #include <cuda_runtime.h>
    #include <stdio.h>
    int main() {
      int deviceCount;
      cudaError_t error = cudaGetDeviceCount(&deviceCount);
      if (error != cudaSuccess) {
        printf(\"cudaGetDeviceCount returned error %d: %s\\n\", error, cudaGetErrorString(error));
        return 1;
      }
      printf(\"%d\", deviceCount);
      return 0;
    }
  ")

  set(CUDA_TEST_FILE "${CMAKE_BINARY_DIR}/platform/cuda_test.cu")
  file(WRITE "${CUDA_TEST_FILE}" "${CUDA_TEST_SOURCE}")

  try_run(RUN_RESULT COMPILE_RESULT
    "${CMAKE_BINARY_DIR}" "${CUDA_TEST_FILE}"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    LINK_LIBRARIES ${CMAKE_CUDA_RUNTIME_LIBRARY}
    RUN_OUTPUT_VARIABLE GPU_COUNT
  )

  if(COMPILE_RESULT AND RUN_RESULT EQUAL 0)
    # We have NVIDIA GPUs, set the NVIDIA target as the default
    message(STATUS "Number of NVIDIA GPUs detected: ${GPU_COUNT}")
    set(__tmp_cudaq_target "nvidia")
  endif()
endif() 

set(CUDAQ_TARGET ${__tmp_cudaq_target} CACHE STRING "The CUDA Quantum target to compile for and execute on. Defaults to `${__tmp_cudaq_target}`")
cudaq_set_target(${CUDAQ_TARGET})