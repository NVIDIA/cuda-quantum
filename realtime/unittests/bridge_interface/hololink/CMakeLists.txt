# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #

# Hololink bridge and playback tools
# ==============================================================================
# These targets are gated by CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS 

# =========================================================================== #
# hololink_bridge  (generic increment bridge)
# =========================================================================== #

message(STATUS "Building hololink_bridge (generic increment)")
find_library(IBVERBS_LIB NAMES ibverbs REQUIRED)


  # Bridge executable (.cpp, linked with CUDA)
add_executable(hololink_app
  hololink_bridge.cpp init_rpc_increment_function_table.cu)

set_target_properties(hololink_app PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(hololink_app
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})

# Link order: static archives first, then shared
# Note: no direct dependency on hololink needed.
target_link_libraries(hololink_app
  PRIVATE
    cudaq-realtime-dispatch
    cudaq-realtime)

# =========================================================================== #
# hololink_fpga_playback  (no GPU / DOCA dependency)
# =========================================================================== #

add_executable(hololink_fpga_playback_app
  hololink_fpga_playback.cpp)

target_include_directories(hololink_fpga_playback_app
  PRIVATE ${CUDAQ_REALTIME_INCLUDE_DIR})

target_link_libraries(hololink_fpga_playback_app
  PRIVATE Threads::Threads)

# =========================================================================== #
# hololink_fpga_emulator  (software FPGA, libibverbs only)
# =========================================================================== #

message(STATUS "Building hololink_fpga_emulator")

add_executable(hololink_fpga_emulator_app
  hololink_fpga_emulator.cpp)

target_link_libraries(hololink_fpga_emulator_app
  PRIVATE
    ${IBVERBS_LIB}
    Threads::Threads)

configure_file(hololink_test.sh.in hololink_test.sh @ONLY)
