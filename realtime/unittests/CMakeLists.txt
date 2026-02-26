# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies 
# ==============================================================================

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  EXCLUDE_FROM_ALL
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PUBLIC --param=evrp-mode=legacy)
endif()
include(GoogleTest)


add_compile_options(-Wno-attributes)

# ==============================================================================
# GPU Dispatch Kernel Tests
# ==============================================================================

find_package(CUDAToolkit)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  
  add_executable(test_dispatch_kernel test_dispatch_kernel.cu)
  
  set_target_properties(test_dispatch_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
  )
  
  target_include_directories(test_dispatch_kernel PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAQ_REALTIME_INCLUDE_DIR}
  )
  
  # Find CUDA device runtime library (required for device-side API calls like cudaGraphLaunch)
  find_library(CUDADEVRT_LIBRARY cudadevrt
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED
  )
  
  target_link_libraries(test_dispatch_kernel PRIVATE 
    GTest::gtest_main 
    CUDA::cudart
    cudaq-realtime
    cudaq-realtime-dispatch
    ${CUDADEVRT_LIBRARY}
  )
  
  add_dependencies(CudaqRealtimeUnitTests test_dispatch_kernel)
  gtest_discover_tests(test_dispatch_kernel
    TEST_PREFIX "test_dispatch_kernel."
  )
  
  message(STATUS "  - test_dispatch_kernel (GPU dispatch infrastructure)")
endif()

# ==============================================================================
# Hololink bridge/emulator/playback tools (optional, not CI)
# ==============================================================================

if (CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS)
  add_subdirectory(utils)
endif()
