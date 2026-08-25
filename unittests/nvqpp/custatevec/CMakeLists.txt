# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_custatevec_components
  ${CUDAQ_GTEST_MAIN}
  BatchTester.cpp
  ConfigTester.cpp
  StateTester.cpp)

target_include_directories(test_custatevec_components PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/nvqir/custatevec)

target_link_libraries(test_custatevec_components PRIVATE
  cudaq
  cudaq-common
  nvqir-custatevec-components
  cuStateVec::cuStateVec
  CUDA::cublas
  CUDA::cudart
  CUDA::curand
  gtest_main)

gtest_discover_tests(test_custatevec_components
  TEST_PREFIX "CuStateVec"
  PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu"
  DISCOVERY_TIMEOUT 120)

function(add_custatevec_backend_test target source backend precision)
  string(REPLACE "-" "_" backend_symbol ${backend})
  set(test_main ${CUDAQ_GTEST_MAIN})
  if(ARGN STREQUAL "MPI")
    set(test_main mpi_main.cpp)
  endif()
  add_executable(${target} ${test_main} ${source})
  target_include_directories(${target} PRIVATE
    ${CMAKE_SOURCE_DIR}/unittests
    ${CMAKE_SOURCE_DIR}/unittests/nvqpp)
  target_compile_definitions(${target} PRIVATE
    NVQIR_BACKEND_NAME=${backend_symbol}
    __MATH_LONG_DOUBLE_CONSTANTS
    CUDAQ_SIMULATION_SCALAR_${precision})
  if(precision STREQUAL "FP32")
    target_compile_definitions(${target} PRIVATE
      CUDAQ_BACKEND_CUSTATEVEC_FP32)
  else()
    target_compile_definitions(${target} PRIVATE
      CUDAQ_BACKEND_CUSTATEVEC_FP64)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(${target} PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
  endif()
  target_link_libraries(${target} PRIVATE
    cudaq
    cudaq-builder
    cudaq-mlir-runtime
    cudaq-platform-default
    nvqir
    nvqir-${backend}
    gtest_main)
endfunction()

add_custatevec_backend_test(custatevec_trajectory_fp32
  TrajectorySingleDeviceTester.cpp custatevec-fp32 FP32)
add_custatevec_backend_test(custatevec_trajectory_fp64
  TrajectorySingleDeviceTester.cpp custatevec-fp64 FP64)
gtest_discover_tests(custatevec_trajectory_fp32
  TEST_PREFIX "CuStateVecEx."
  PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
gtest_discover_tests(custatevec_trajectory_fp64
  TEST_PREFIX "CuStateVecEx."
  PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")


add_custatevec_backend_test(custatevec_host_migration
  HostDeviceTester.cpp custatevec-fp64 FP64)
add_test(NAME CuStateVecEx.HostMigration COMMAND ${CMAKE_COMMAND} -E env
  CUDAQ_MAX_GPU_MEMORY_GB=1 CUDAQ_MAX_CPU_MEMORY_GB=32
  $<TARGET_FILE:custatevec_host_migration>
  --gtest_filter=HostDeviceTester.*)

add_test(NAME CuStateVecEx.HostMigrationLarge COMMAND ${CMAKE_COMMAND} -E env
  CUDAQ_MAX_GPU_MEMORY_GB=16 CUDAQ_MAX_CPU_MEMORY_GB=1024
  $<TARGET_FILE:custatevec_host_migration>
  --gtest_filter=HostDeviceLargeMemTester.*)

add_test(NAME CuStateVecEx.HostMigrationExpectation COMMAND ${CMAKE_COMMAND}
  -E env CUDAQ_MAX_CPU_MEMORY_GB=1024
  $<TARGET_FILE:custatevec_host_migration>
  --gtest_filter=HostDeviceMatrixExpValTester.*)
set_tests_properties(CuStateVecEx.HostMigration
  CuStateVecEx.HostMigrationLarge CuStateVecEx.HostMigrationExpectation
  PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
set_tests_properties(CuStateVecEx.HostMigrationLarge
  CuStateVecEx.HostMigrationExpectation
  PROPERTIES LABELS "gpu_required;large_memory_required")

if(MPI_CXX_FOUND)
  # Skip tests when fewer than two GPUs are available.
  function(add_custatevec_mgpu_test test_name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "COMMAND;ENVIRONMENT")

    # Exercise both the CUDA-Q and built-in cuStateVec MPI plugins.
    foreach(test_suffix IN ITEMS "" ".OpenMPI")
      set(test_environment ${ARG_ENVIRONMENT})
      if(test_suffix)
        list(APPEND test_environment
          "CUDAQ_MGPU_COMM_PLUGIN_TYPE=OPENMPI")
      endif()

      set(full_test_name ${test_name}${test_suffix})
      add_test(NAME ${full_test_name}
        COMMAND bash -c
        "gpu_count=\$(nvidia-smi --list-gpus 2>/dev/null | wc -l); [ \"\$gpu_count\" -ge 2 ] || exit 77; exec \"\$@\""
        -- ${ARG_COMMAND})
      set_tests_properties(${full_test_name}
        PROPERTIES
          SKIP_RETURN_CODE 77
          ENVIRONMENT "${test_environment}"
          LABELS "gpu_required;mgpus_required"
          RESOURCE_LOCK "gpu")
    endforeach()
  endfunction()

  add_custatevec_backend_test(custatevec_mgpu
    MGpuTester.cpp nvidia-mgpu FP64 MPI)
  add_custatevec_backend_test(custatevec_mgpu_fp32
    MGpuTester.cpp nvidia-mgpu-fp32 FP32 MPI)
  add_custatevec_backend_test(custatevec_mgpu_host_migration
    HostDeviceTester.cpp nvidia-mgpu FP64 MPI)

  add_custatevec_mgpu_test(CuStateVecEx.MGPU
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=5"
    COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu>
    --gtest_filter=*MGPUTester.*)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUFP32
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=5"
    COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu_fp32>
    --gtest_filter=*MGPUTester.*)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUHostMigration
    ENVIRONMENT
      "CUDAQ_MGPU_NQUBITS_THRESH=5"
      "CUDAQ_MAX_GPU_MEMORY_GB=1"
      "CUDAQ_MAX_CPU_MEMORY_GB=32"
    COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu_host_migration>
    --gtest_filter=HostDeviceTester.*)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUSequentialNoise
    ENVIRONMENT
      "CUDAQ_MGPU_NQUBITS_THRESH=5"
      "CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE=1000000"
    COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu>
    --gtest_filter=*MGPUTester.checkNoise)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUStateReuse
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=25"
    COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu>
    --gtest_filter=*MGPUStateReuseTester.*)
endif()
