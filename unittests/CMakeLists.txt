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
FetchContent_MakeAvailable(googletest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PUBLIC --param=evrp-mode=legacy)
endif()
include(GoogleTest)


add_compile_options(-Wno-attributes) 

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test_config.h.in"
               "${CMAKE_BINARY_DIR}/include/test_config.h" 
               @ONLY)
include_directories(${CMAKE_BINARY_DIR}/include)

# ======= Shmem Channel Tests ==================================================
add_library(shmem-add SHARED test_device_libraries/add.cpp)

add_executable(test_cpu_shmem_device test_cpu_shmem_device.cpp)
target_link_libraries(test_cpu_shmem_device PRIVATE GTest::gtest_main cudaq-qclink)
add_dependencies(CUDAQXQCLINKUnitTests test_cpu_shmem_device)
gtest_discover_tests(test_cpu_shmem_device)
# ==============================================================================

# ======= CUDA Channel Tests ===================================================
if (CUDA_FOUND) 
  add_library(vec-add OBJECT test_device_libraries/vec_add.cu)
  set_property(TARGET vec-add PROPERTY CUDA_FATBIN_COMPILATION ON)

  add_executable(test_cuda_device test_cuda_device.cpp)
  target_link_libraries(test_cuda_device PRIVATE GTest::gtest_main cudaq-qclink)
  add_dependencies(CUDAQXQCLINKUnitTests test_cuda_device)
  gtest_discover_tests(test_cuda_device)

  add_executable(test_nv_rdma_device test_nv_rdma_device.cu)
  target_link_libraries(test_nv_rdma_device PRIVATE GTest::gtest_main cudaq-qclink)
  add_dependencies(CUDAQXQCLINKUnitTests test_nv_rdma_device)
  gtest_discover_tests(test_nv_rdma_device)
endif() 
# ==============================================================================


# ======= Shmem Channel Tests ==================================================
add_executable(test_cudaq_compiler test_cudaq_compiler.cpp)
target_link_libraries(test_cudaq_compiler PRIVATE GTest::gtest_main cudaq-qclink)
add_dependencies(CUDAQXQCLINKUnitTests test_cudaq_compiler)
gtest_discover_tests(test_cudaq_compiler)
# ==============================================================================

# ======= QCLink Integration Tests ============================================
add_executable(test_qclink test_qclink.cpp)
target_link_libraries(test_qclink PRIVATE GTest::gtest_main cudaq-qclink)
add_dependencies(CUDAQXQCLINKUnitTests test_qclink)
gtest_discover_tests(test_qclink)
# ==============================================================================


