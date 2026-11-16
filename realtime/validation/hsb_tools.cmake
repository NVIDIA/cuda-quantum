# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RoCE validation apps and HSB playback/emulator tools
# ==============================================================================
# These targets are gated by CUDAQ_REALTIME_ENABLE_HSB_TOOLS and require
# a pre-built HSB core library. The GPU RoCE bridge has its own DOCA checks.
# They are NOT CI tests -- they need FPGA hardware or an FPGA emulator.

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

find_library(IBVERBS_LIB NAMES ibverbs)

# --------------------------------------------------------------------------- #
# Find fmt (transitive dependency of HSB logging)
# --------------------------------------------------------------------------- #

find_path(FMT_INCLUDE_DIR
  NAMES fmt/format.h
  PATHS /opt/nvidia/holoscan /usr/local/cudaq /usr /usr/local
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)

# =========================================================================== #
# hsb_fpga_playback  (uses hololink_core for FPGA control plane)
# =========================================================================== #

add_executable(hsb_fpga_playback
  hsb_fpga_playback.cpp)

target_include_directories(hsb_fpga_playback
  PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    "${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src"
    "${CUDAToolkit_INCLUDE_DIRS}")

if (FMT_INCLUDE_DIR)
  target_include_directories(hsb_fpga_playback
    PRIVATE "${FMT_INCLUDE_DIR}")
endif()

target_link_libraries(hsb_fpga_playback
  PRIVATE ${HSB_CORE_LIB} CUDA::cuda_driver Threads::Threads)

# Ship the direct playback tool for the installed compatibility workflow.
install(TARGETS hsb_fpga_playback
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT hsb-tools)

configure_file(gpu_roce_test.sh gpu_roce_test.sh COPYONLY)
configure_file(hsb_test_cpu.sh
               hsb_test_cpu.sh COPYONLY)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/roce_test.sh
        ${CMAKE_CURRENT_BINARY_DIR}/gpu_roce_test.sh
        ${CMAKE_CURRENT_BINARY_DIR}/hsb_test_cpu.sh
        DESTINATION utils
        COMPONENT hsb-tools)

# =========================================================================== #
# hsb_fpga_emulator  (software FPGA, libibverbs only)
# =========================================================================== #

if (IBVERBS_LIB)
  message(STATUS "Building hsb_fpga_emulator")

  add_executable(hsb_fpga_emulator
    hsb_fpga_emulator.cpp)

  target_include_directories(hsb_fpga_emulator
    PRIVATE
      ${CUDAQ_REALTIME_INCLUDE_DIR})

  target_link_libraries(hsb_fpga_emulator
    PRIVATE
      ${IBVERBS_LIB}
      Threads::Threads)
  # install the emulator binary for testing the installer
  install(TARGETS hsb_fpga_emulator
          DESTINATION ${CMAKE_INSTALL_BINDIR}
          COMPONENT hsb-tools)
else()
  message(WARNING "libibverbs not found. hsb_fpga_emulator will not be built.")
endif()

# The RoCE-specific CLI shares the generic validation app’s RPC handlers.
if (TARGET rpc_increment_ft)
  foreach(app IN ITEMS roce_app)
    add_executable(${app} ${app}.cpp
      init_rpc_increment_function_table_host.cpp)
    set_target_properties(${app} PROPERTIES
      LINKER_LANGUAGE CUDA
      INSTALL_RPATH "$ORIGIN/../lib"
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_include_directories(${app} PRIVATE
      ${CUDAQ_REALTIME_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${app} PRIVATE
      cudaq-realtime-dispatch cudaq-realtime rpc_increment_ft Threads::Threads)
    if (TARGET cudaq-realtime-bridge-gpu-roce)
      add_dependencies(${app} cudaq-realtime-bridge-gpu-roce)
    endif()
    if (TARGET cudaq-realtime-bridge-cpu-roce)
      add_dependencies(${app} cudaq-realtime-bridge-cpu-roce)
    endif()
    install(TARGETS ${app} DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT hsb-tools)
  endforeach()
  # The application-provided cooperative grid dispatcher (--dispatcher=grid).
  cudaq_realtime_add_dispatch_kernel(roce_app
    ${CMAKE_CURRENT_LIST_DIR}/grid_dispatcher.cu)
endif()

# Playback/RTH integration tests are registered in ../unittests/validation, after
# both producer and consumer targets exist.
