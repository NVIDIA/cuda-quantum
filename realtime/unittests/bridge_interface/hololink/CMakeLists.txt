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

# Demo application that uses the hololink bridge interface to implement a simple "increment" RPC function.
# This is the same as the one in cuda-quantum/realtime/unittests/utils 
# (which is implemented end-to-end without using the bridge pluggable interface).
add_executable(hololink_app
  hololink_bridge.cpp)

set_target_properties(hololink_app PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(hololink_app
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})

# Note: no direct dependency on hololink needed.
target_link_libraries(hololink_app
  PRIVATE
    cudaq-realtime-dispatch
    cudaq-realtime)

# Find the rpc_increment_ft target built (device code for the increment function table) and link it to the app.
if (TARGET rpc_increment_ft)
  add_dependencies(hololink_app rpc_increment_ft)
  message(STATUS "  - rpc_increment_ft target found, linking to app")
  target_link_libraries(hololink_app PRIVATE rpc_increment_ft)
endif()

# Find the `hololink_fpga_emulator` and `hololink_fpga_playback` targets built by the utils/ CMakeLists.txt, and link them to the app for easy invocation.
if (TARGET hololink_fpga_emulator)
  add_dependencies(hololink_app hololink_fpga_emulator)
  set(HOLOLINK_EMULATOR_BIN ${CMAKE_BINARY_DIR}/unittests/utils/hololink_fpga_emulator)
else()
  message(WARNING "hololink_fpga_emulator target not found. "
                  "FPGA emulation will not be available for hololink_app.")
endif()
if (TARGET hololink_fpga_playback)
  add_dependencies(hololink_app hololink_fpga_playback)
  set(HOLOLINK_PLAYBACK_BIN ${CMAKE_BINARY_DIR}/unittests/utils/hololink_fpga_playback)
else()
  message(WARNING "hololink_fpga_playback target not found. "
                  "FPGA playback will not be available for hololink_app.")
endif()


configure_file(hololink_test.sh.in hololink_test.sh @ONLY)
