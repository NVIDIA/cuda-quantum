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
  ErrorTester.cpp
  RandomTester.cpp
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

add_executable(test_custatevec_communicator_packaging
  ${CUDAQ_GTEST_MAIN}
  CommunicatorPackagingTester.cpp)
target_compile_definitions(test_custatevec_communicator_packaging PRIVATE
  CUSTATEVEC_MGPU_FP32_PATH="$<TARGET_FILE:nvqir-nvidia-mgpu-fp32>"
  CUSTATEVEC_MGPU_FP64_PATH="$<TARGET_FILE:nvqir-nvidia-mgpu>"
  CUSTATEVEC_STANDALONE_COMM_PATH="$<TARGET_FILE_DIR:nvqir-nvidia-mgpu-fp32>/${CMAKE_SHARED_LIBRARY_PREFIX}nvqir-custatevec-communicator${CMAKE_SHARED_LIBRARY_SUFFIX}")
target_link_libraries(test_custatevec_communicator_packaging PRIVATE
  ${CMAKE_DL_LIBS}
  gtest_main)
add_dependencies(test_custatevec_communicator_packaging
  nvqir-nvidia-mgpu-fp32
  nvqir-nvidia-mgpu)
gtest_discover_tests(test_custatevec_communicator_packaging
  TEST_PREFIX "CuStateVecEx."
  PROPERTIES LABELS "gpu_required"
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
  trajectory_single_device_tester.cpp custatevec-fp32 FP32)
add_custatevec_backend_test(custatevec_trajectory_fp64
  trajectory_single_device_tester.cpp custatevec-fp64 FP64)
gtest_discover_tests(custatevec_trajectory_fp32
  TEST_PREFIX "CuStateVecEx."
  PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
gtest_discover_tests(custatevec_trajectory_fp64
  TEST_PREFIX "CuStateVecEx."
  PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")

add_custatevec_backend_test(custatevec_ptsbe
  PTSBETester.cpp custatevec-fp32 FP32)
target_include_directories(custatevec_ptsbe PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/nvqir/custatevec)
target_link_libraries(custatevec_ptsbe PRIVATE
  nvqir-custatevec-components
  CUDA::curand)
gtest_discover_tests(custatevec_ptsbe
  TEST_PREFIX "CuStateVecEx."
  PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")

add_custatevec_backend_test(custatevec_host_migration
  host_device_tester.cpp custatevec-fp64 FP64)
add_test(NAME CuStateVecEx.HostMigration COMMAND ${CMAKE_COMMAND} -E env
  CUDAQ_MAX_GPU_MEMORY_GB=1 CUDAQ_MAX_CPU_MEMORY_GB=32
  $<TARGET_FILE:custatevec_host_migration>)

add_custatevec_backend_test(custatevec_host_migration_large
  host_device_large_mem_tester.cpp custatevec-fp64 FP64)
add_test(NAME CuStateVecEx.HostMigrationLarge COMMAND ${CMAKE_COMMAND} -E env
  CUDAQ_MAX_GPU_MEMORY_GB=16 CUDAQ_MAX_CPU_MEMORY_GB=1024
  $<TARGET_FILE:custatevec_host_migration_large>)

add_custatevec_backend_test(custatevec_host_migration_expectation
  host_device_matrix_exp_val.cpp custatevec-fp64 FP64)
add_test(NAME CuStateVecEx.HostMigrationExpectation COMMAND ${CMAKE_COMMAND}
  -E env CUDAQ_MAX_CPU_MEMORY_GB=1024
  $<TARGET_FILE:custatevec_host_migration_expectation>)
set_tests_properties(CuStateVecEx.HostMigration
  CuStateVecEx.HostMigrationLarge CuStateVecEx.HostMigrationExpectation
  PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
set_tests_properties(CuStateVecEx.HostMigrationLarge
  CuStateVecEx.HostMigrationExpectation
  PROPERTIES LABELS "gpu_required;large_memory_required")

if(MPI_CXX_FOUND)
  # Skip tests when fewer than two GPUs are available.
  function(add_custatevec_mgpu_test test_name)
    add_test(NAME ${test_name}
      COMMAND bash -c
      "gpu_count=\$(nvidia-smi --list-gpus 2>/dev/null | wc -l); [ \"\$gpu_count\" -ge 2 ] || exit 77; exec \"\$@\""
      -- ${ARGN})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
  endfunction()

  add_custatevec_backend_test(custatevec_mgpu
    mgpu_tester.cpp nvidia-mgpu FP64 MPI)
  add_custatevec_backend_test(custatevec_mgpu_fp32
    mgpu_tester.cpp nvidia-mgpu-fp32 FP32 MPI)
  add_custatevec_backend_test(custatevec_mgpu_host_migration
    host_device_tester.cpp nvidia-mgpu FP64 MPI)
  add_custatevec_backend_test(custatevec_mgpu_state_reuse
    mgpu_state_reuse_tester.cpp nvidia-mgpu FP64 MPI)
  add_custatevec_backend_test(custatevec_mgpu_state
    MGpuStateTester.cpp nvidia-mgpu FP64 MPI)
  add_custatevec_backend_test(custatevec_mgpu_ptsbe
    PTSBEMgpuTester.cpp nvidia-mgpu FP64 MPI)
  target_include_directories(custatevec_mgpu_ptsbe PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime/nvqir/custatevec)
  target_link_libraries(custatevec_mgpu_ptsbe PRIVATE
    nvqir-custatevec-components)
  target_include_directories(custatevec_mgpu_state PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime/nvqir/custatevec)
  target_link_libraries(custatevec_mgpu_state PRIVATE
    nvqir-custatevec-components MPI::MPI_CXX)
  add_custatevec_mgpu_test(CuStateVecEx.MGPU
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu>)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUFP32
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu_fp32>)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUHostMigration
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu_host_migration>)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUSequentialNoise
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu>
    --gtest_filter=MGPUTester.checkNoise)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUStateReuse
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu_state_reuse>)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUState
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu_state>)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUStateNonPowerOfTwo
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 6
    $<TARGET_FILE:custatevec_mgpu_state>
    --gtest_filter=MGpuTesterMultiProcesses.BelowThresholdStateSupportsNonPowerOfTwoRanks)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUStateSmallTransition
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 4
    $<TARGET_FILE:custatevec_mgpu_state>
    --gtest_filter=MGpuTesterMultiProcesses.GrowsSmallReplicatedStateAcrossThreshold)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUStateMigrationLevel
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu_state>
    --gtest_filter=MGpuTesterMultiProcesses.ValidatesMigrationLevelWithoutMigrationWires)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUStateRebindConfig
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 4
    $<TARGET_FILE:custatevec_mgpu_state>
    --gtest_filter=MGpuTesterMultiProcesses.CommunicatorRebindingReinitializesState)
  add_custatevec_mgpu_test(CuStateVecEx.MGPUPTSBE
    ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
    $<TARGET_FILE:custatevec_mgpu_ptsbe>)
  set_tests_properties(CuStateVecEx.MGPU CuStateVecEx.MGPUFP32
                       CuStateVecEx.MGPUState CuStateVecEx.MGPUStateNonPowerOfTwo
                       CuStateVecEx.MGPUStateSmallTransition
    PROPERTIES
      ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=5"
      LABELS "gpu_required;mgpus_required"
      RESOURCE_LOCK "gpu")
  set_tests_properties(CuStateVecEx.MGPUStateSmallTransition PROPERTIES
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=2")
  set_tests_properties(CuStateVecEx.MGPUStateMigrationLevel PROPERTIES
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=1;CUDAQ_HOST_DEVICE_MIGRATION_LEVEL=2"
    LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
  set_tests_properties(CuStateVecEx.MGPUStateReuse PROPERTIES
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=25"
    LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
  set_tests_properties(CuStateVecEx.MGPUHostMigration PROPERTIES
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=5;CUDAQ_MAX_GPU_MEMORY_GB=1;CUDAQ_MAX_CPU_MEMORY_GB=32"
    LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
  set_tests_properties(CuStateVecEx.MGPUSequentialNoise PROPERTIES
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=5;CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE=1000000"
    LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
  set_tests_properties(CuStateVecEx.MGPUStateRebindConfig PROPERTIES
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=5;CUDAQ_GPU_FABRIC=2"
    LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
  set_tests_properties(CuStateVecEx.MGPUPTSBE PROPERTIES
    ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=5;CUDAQ_GPU_RNG_THRESHOLD=1"
    LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
endif()
