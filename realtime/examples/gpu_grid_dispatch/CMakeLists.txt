# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

# gpu_grid_dispatch: serve an RPC with a whole resident cooperative GPU grid
# while the CPU owns the NIC (CUDA-Q Realtime's host dispatch path over the
# cpu_roce bridge).  Builds against an installed CUDA-Q Realtime, found
# through CUDAQ_REALTIME_ROOT (cache variable or environment), as
# examples/gpu_dispatch does:
#
#   cmake -S . -B build -DCUDAQ_REALTIME_ROOT=<realtime install prefix>
#   cmake --build build
#
# gdrcopy is optional (--notify=gdrcopy); it is looked for under GDRCOPY_HOME,
# /usr/local/gdrcopy and /opt/gdrcopy.

# Before project(): CMake fixes its own default architecture when it enables
# CUDA.  Hopper, Blackwell, Thor and Blackwell workstation / Spark.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "90;100;110;120")
endif()
project(gpu_grid_dispatch LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

if (NOT CUDAQ_REALTIME_ROOT)
  set(CUDAQ_REALTIME_ROOT "$ENV{CUDAQ_REALTIME_ROOT}" CACHE PATH
      "Path to the CUDA-Q Realtime installation")
endif()
if (NOT CUDAQ_REALTIME_ROOT)
  set(CUDAQ_REALTIME_ROOT "/opt/nvidia/cudaq/realtime")
  message(STATUS "CUDAQ_REALTIME_ROOT not set; using ${CUDAQ_REALTIME_ROOT}")
endif()

find_library(CUDAQ_REALTIME_LIB NAMES cudaq-realtime
  HINTS ${CUDAQ_REALTIME_ROOT}/lib NO_DEFAULT_PATH)
if (NOT CUDAQ_REALTIME_LIB)
  message(FATAL_ERROR
    "libcudaq-realtime not found under ${CUDAQ_REALTIME_ROOT}/lib; set CUDAQ_REALTIME_ROOT")
endif()
message(STATUS "gpu_grid_dispatch: cudaq-realtime at ${CUDAQ_REALTIME_LIB}")

# The host dispatch loop lives in libcudaq-realtime itself, so no dispatch
# archive and no relocatable device code are needed: the grid is an ordinary
# kernel this program launches.
add_executable(gpu_grid_dispatch
  gpu_grid_dispatch.cpp
  grid_kernel.cu)
set_target_properties(gpu_grid_dispatch PROPERTIES
  CUDA_STANDARD 17
  CXX_STANDARD 17
  # The bridge provider is dlopen'ed from the install's lib directory.
  BUILD_RPATH "${CUDAQ_REALTIME_ROOT}/lib"
  INSTALL_RPATH "${CUDAQ_REALTIME_ROOT}/lib")
target_include_directories(gpu_grid_dispatch PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDAQ_REALTIME_ROOT}/include
  ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(gpu_grid_dispatch PRIVATE
  GPU_GRID_BRIDGE_DIR="${CUDAQ_REALTIME_ROOT}/lib")
target_link_libraries(gpu_grid_dispatch PRIVATE
  ${CUDAQ_REALTIME_LIB}
  CUDA::cudart
  Threads::Threads)

# gdrcopy (optional): --notify=gdrcopy writes the mailbox in device memory
# from the CPU through a BAR mapping.
find_path(GDRCOPY_INCLUDE_DIR gdrapi.h
  HINTS "$ENV{GDRCOPY_HOME}" /usr/local/gdrcopy /opt/gdrcopy
  PATH_SUFFIXES include)
find_library(GDRCOPY_LIBRARY NAMES gdrapi
  HINTS "$ENV{GDRCOPY_HOME}" /usr/local/gdrcopy /opt/gdrcopy
  PATH_SUFFIXES lib lib64)
if (GDRCOPY_INCLUDE_DIR AND GDRCOPY_LIBRARY)
  target_compile_definitions(gpu_grid_dispatch PRIVATE GPU_GRID_HAVE_GDRCOPY=1)
  target_include_directories(gpu_grid_dispatch PRIVATE ${GDRCOPY_INCLUDE_DIR})
  # cuPointerSetAttribute(SYNC_MEMOPS) on the mailbox page: the driver API.
  target_link_libraries(gpu_grid_dispatch PRIVATE
    ${GDRCOPY_LIBRARY}
    CUDA::cuda_driver)
  message(STATUS "gpu_grid_dispatch: gdrcopy at ${GDRCOPY_LIBRARY}")
else()
  message(STATUS "gpu_grid_dispatch: gdrcopy not found; --notify=gdrcopy unavailable")
endif()

install(TARGETS gpu_grid_dispatch RUNTIME DESTINATION bin)
install(PROGRAMS gpu_grid_dispatch_test.sh DESTINATION bin)
