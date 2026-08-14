# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

function(add_backend_test target )
  set(singleValues BACKEND_NAME)
  set(multiValues SOURCES)
  cmake_parse_arguments(ARG "" "${singleValues}" "${multiValues}" ${ARGN})

  add_executable(${target} ${ARG_SOURCES})
  target_compile_definitions(${target} PRIVATE NVQIR_BACKEND_NAME=${ARG_BACKEND_NAME})

  target_link_libraries(${target}
      PRIVATE
      nvqir
      nvqir-${ARG_BACKEND_NAME}
      cudaq
      cudaq-logger
      fmt::fmt-header-only
      gtest_main)
  cudaq_gtest_discover_tests(${target} DISCOVERY_TIMEOUT 120)
endfunction()


add_backend_test(qpp_tester 
  BACKEND_NAME qpp
  SOURCES QPPTester.cpp)


add_backend_test(dm_tester 
  BACKEND_NAME dm
  SOURCES QPPDMTester.cpp)


add_backend_test(stim_tester 
  BACKEND_NAME stim
  SOURCES StimTester.cpp)
target_include_directories(stim_tester PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/nvqir/stim
  ${CMAKE_SOURCE_DIR}/tpls/Stim/src)
target_link_libraries(stim_tester PRIVATE libstim)

if (cuStateVec_FOUND)
  function(add_custatevec_simulator_test target backend precision)
    string(REPLACE "-" "_" backend_symbol ${backend})
    add_executable(${target}
      ${CUDAQ_GTEST_MAIN}
      CuStateVecTester.cpp)
    target_include_directories(${target} PRIVATE
      ${CMAKE_SOURCE_DIR}/unittests
      ${CMAKE_SOURCE_DIR}/runtime/nvqir/custatevec)
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
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
      target_link_options(${target} PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
    endif()
    target_link_libraries(${target} PRIVATE
      cudaq
      cudaq-builder
      cudaq-mlir-runtime
      cudaq-platform-default
      nvqir
      nvqir-${backend}
      nvqir-custatevec-components
      CUDA::curand
      gtest_main)
    gtest_discover_tests(${target}
      TEST_PREFIX "CuStateVecEx."
      PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
  endfunction()

  add_custatevec_simulator_test(custatevec_tester custatevec-fp32 FP32)
  add_custatevec_simulator_test(custatevec_tester_fp64 custatevec-fp64 FP64)
endif()

if (cuStateVec_FOUND AND MPI_CXX_FOUND)
  function(add_custatevec_mgpu_tester target backend precision)
    string(REPLACE "-" "_" backend_symbol ${backend})
    add_executable(${target}
      ${CMAKE_SOURCE_DIR}/unittests/nvqpp/custatevec/mpi_main.cpp
      CuStateVecMpiTester.cpp)
    target_include_directories(${target} PRIVATE
      ${CMAKE_SOURCE_DIR}/unittests
      ${CMAKE_SOURCE_DIR}/unittests/nvqpp
      ${CMAKE_SOURCE_DIR}/runtime/nvqir/custatevec)
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
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
      target_link_options(${target} PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
    endif()
    target_link_libraries(${target} PRIVATE
      cudaq
      cudaq-builder
      cudaq-mlir-runtime
      cudaq-platform-default
      nvqir
      nvqir-${backend}
      nvqir-custatevec-components
      gtest_main)
  endfunction()

  add_custatevec_mgpu_tester(custatevec_mgpu_tester nvidia-mgpu FP64)
  add_custatevec_mgpu_tester(custatevec_mgpu_tester_fp32
    nvidia-mgpu-fp32 FP32)

  # Skip tests when fewer than two GPUs are available.
  function(add_custatevec_backend_mgpu_test test_name)
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

  function(add_custatevec_mgpu_test_suite target suffix)
    set(state_test CuStateVecEx.MGPUState${suffix})
    set(transition_test CuStateVecEx.MGPUStateSmallTransition${suffix})
    set(migration_test CuStateVecEx.MGPUStateMigrationLevel${suffix})
    set(rebind_test CuStateVecEx.MGPUStateRebindConfig${suffix})
    set(ptsbe_test CuStateVecEx.MGPUPTSBE${suffix})

    add_custatevec_backend_mgpu_test(${state_test}
      ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=5"
      COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
      $<TARGET_FILE:${target}>
      --gtest_filter=MGpuTesterMultiProcesses.*)
    add_custatevec_backend_mgpu_test(${transition_test}
      ENVIRONMENT "CUDAQ_MGPU_NQUBITS_THRESH=2"
      COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 4
      $<TARGET_FILE:${target}>
      --gtest_filter=MGpuTesterMultiProcesses.GrowsSmallReplicatedStateAcrossThreshold)
    add_custatevec_backend_mgpu_test(${migration_test}
      ENVIRONMENT
        "CUDAQ_MGPU_NQUBITS_THRESH=1"
        "CUDAQ_HOST_DEVICE_MIGRATION_LEVEL=2"
      COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
      $<TARGET_FILE:${target}>
      --gtest_filter=MGpuTesterMultiProcesses.ValidatesMigrationLevelWithoutMigrationWires)
    add_custatevec_backend_mgpu_test(${rebind_test}
      ENVIRONMENT
        "CUDAQ_MGPU_NQUBITS_THRESH=5"
        "CUDAQ_GPU_FABRIC=2"
      COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 4
      $<TARGET_FILE:${target}>
      --gtest_filter=MGpuTesterMultiProcesses.CommunicatorRebindingReinitializesState)
    add_custatevec_backend_mgpu_test(${ptsbe_test}
      ENVIRONMENT
        "CUDAQ_MGPU_NQUBITS_THRESH=5"
        "CUDAQ_GPU_RNG_THRESHOLD=1"
      COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 2
      $<TARGET_FILE:${target}>
      --gtest_filter=PtsbeMgpuTest.*)
  endfunction()

  add_custatevec_mgpu_test_suite(custatevec_mgpu_tester "")
  add_custatevec_mgpu_test_suite(custatevec_mgpu_tester_fp32 FP32)
endif()
