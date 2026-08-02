# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #

# GPU RoCE (GpuRoceTransceiver) bridge implementation
# ==============================================================================
# These targets are gated by CUDAQ_REALTIME_ENABLE_HSB_TOOLS and require
# a pre-built holoscan-sensor-bridge (HSB) with DOCA support.


if (NOT HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR)
  message(FATAL_ERROR
    "HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR must be set when building HSB tools.")
endif()
if (NOT HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR)
  message(FATAL_ERROR
    "HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR must be set when building HSB tools.")
endif()

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

# --------------------------------------------------------------------------- #
# Find HSB core library
# --------------------------------------------------------------------------- #

find_library(HSB_CORE_LIB
  NAMES hololink_core
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/core"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

if (NOT HSB_CORE_LIB)
  message(FATAL_ERROR
    "Could not find hololink_core library under ${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}.")
endif()

# --------------------------------------------------------------------------- #
# Find GPU RoCE Transceiver library
# --------------------------------------------------------------------------- #

find_library(GPU_ROCE_TRANSCEIVER_LIB
  NAMES gpu_roce_transceiver
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/operators/gpu_roce_transceiver"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

if (NOT GPU_ROCE_TRANSCEIVER_LIB)
  message(WARNING
    "Could not find gpu_roce_transceiver library. "
    "gpu_roce_bridge will not be built.")
endif()

# --------------------------------------------------------------------------- #
# Find transitive HSB libraries
# --------------------------------------------------------------------------- #

find_library(HSB_COMMON_LIB
  NAMES hololink
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/common"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

find_library(BASE_RECEIVER_OP_LIB
  NAMES base_receiver_op
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/operators"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

find_library(IBVERBS_LIB NAMES ibverbs)

# --------------------------------------------------------------------------- #
# Find DOCA libraries
# --------------------------------------------------------------------------- #

set(DOCA_PATH "/opt/mellanox/doca")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)")
  set(DOCA_LIB_DIR "${DOCA_PATH}/lib/x86_64-linux-gnu")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
  set(DOCA_LIB_DIR "${DOCA_PATH}/lib/aarch64-linux-gnu")
else()
  set(DOCA_LIB_DIR "${DOCA_PATH}/lib")
endif()

find_path(DOCA_INCLUDE_DIR doca_verbs.h
  PATHS ${DOCA_PATH}/include
  NO_DEFAULT_PATH)

# RHEL may have DOCA libraries under ${DOCA_PATH}/lib64  
find_library(DOCA_VERBS_LIB doca_verbs
  PATHS ${DOCA_LIB_DIR} ${DOCA_PATH}/lib/ ${DOCA_PATH}/lib64/
  NO_DEFAULT_PATH)

find_library(DOCA_GPUNETIO_LIB doca_gpunetio
  PATHS ${DOCA_LIB_DIR} ${DOCA_PATH}/lib/ ${DOCA_PATH}/lib64/
  NO_DEFAULT_PATH)

find_library(DOCA_COMMON_LIB doca_common
  PATHS ${DOCA_LIB_DIR} ${DOCA_PATH}/lib/ ${DOCA_PATH}/lib64/
  NO_DEFAULT_PATH)

# --------------------------------------------------------------------------- #
# Find Holoscan (required by gpu_roce_transceiver -> holoscan::core)
# --------------------------------------------------------------------------- #

find_package(holoscan QUIET)

# --------------------------------------------------------------------------- #
# Find fmt (transitive dependency of HSB logging)
# --------------------------------------------------------------------------- #

find_path(FMT_INCLUDE_DIR
  NAMES fmt/format.h
  PATHS /opt/nvidia/holoscan /usr/local/cudaq /usr /usr/local
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)

# =========================================================================== #
# gpu_roce_bridge library
# =========================================================================== #

if (GPU_ROCE_TRANSCEIVER_LIB AND
    DOCA_INCLUDE_DIR AND DOCA_VERBS_LIB AND DOCA_COMMON_LIB AND
    DOCA_GPUNETIO_LIB)

  message(STATUS "Building gpu_roce_bridge interface library with the following dependencies:")
  message(STATUS "  GPU RoCE Transceiver: ${GPU_ROCE_TRANSCEIVER_LIB}")

  find_path(DOCA_GPU_INCLUDE_DIR doca_gpunetio_dev_verbs_common.cuh
    HINTS /opt/mellanox/doca/include
  )

  find_library(CUDADEVRT_LIBRARY cudadevrt
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED
  )

  # Library implementing the gpu_roce bridge interface (includes unified kernel)
  add_library(cudaq-realtime-bridge-gpu-roce SHARED
    bridge_impl.cpp
    gpu_roce_wrapper.cpp
    unified_dispatch_kernel.cu)

  set_target_properties(cudaq-realtime-bridge-gpu-roce PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

  target_include_directories(cudaq-realtime-bridge-gpu-roce
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CUDAQ_REALTIME_INCLUDE_DIR}
      "${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src"
      "${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src/hololink/operators/gpu_roce_transceiver"
      ${DOCA_INCLUDE_DIR}
      ${DOCA_GPU_INCLUDE_DIR}
      ${FMT_INCLUDE_DIR})

  target_link_libraries(cudaq-realtime-bridge-gpu-roce
    PRIVATE
      ${GPU_ROCE_TRANSCEIVER_LIB}
      ${BASE_RECEIVER_OP_LIB}
      ${HSB_CORE_LIB}
      ${HSB_COMMON_LIB}
      cudaq-realtime
      CUDA::cudart
      CUDA::cuda_driver
      ${CUDADEVRT_LIBRARY}
      ${DOCA_VERBS_LIB}
      ${DOCA_GPUNETIO_LIB}
      ${DOCA_COMMON_LIB}
      ${IBVERBS_LIB}
      Threads::Threads
      ${CMAKE_DL_LIBS})

  target_link_options(cudaq-realtime-bridge-gpu-roce PRIVATE
    "LINKER:--allow-multiple-definition")

  if (holoscan_FOUND)
    target_link_libraries(cudaq-realtime-bridge-gpu-roce PRIVATE holoscan::core)
  endif()

  # Set RPATH for shared libraries
  set_target_properties(cudaq-realtime-bridge-gpu-roce PROPERTIES
    BUILD_RPATH "${DOCA_LIB_DIR}"
    INSTALL_RPATH "${DOCA_LIB_DIR}")
  # Build output directory (/lib)
  set_target_properties(cudaq-realtime-bridge-gpu-roce PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  install(TARGETS cudaq-realtime-bridge-gpu-roce DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
  if (NOT GPU_ROCE_TRANSCEIVER_LIB)
    message(WARNING "gpu_roce_transceiver library not found. "
                    "gpu_roce_bridge will not be built.")
  endif()
  if (NOT DOCA_INCLUDE_DIR OR NOT DOCA_VERBS_LIB)
    message(WARNING "DOCA libraries not found. "
                    "gpu_roce_bridge requires DOCA.")
  endif()
endif()
