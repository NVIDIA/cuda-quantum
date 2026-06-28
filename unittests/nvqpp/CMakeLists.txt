# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CUDAQ_RUNTIME_TEST_SOURCES
  # Integration tests
  integration/adjoint_tester.cpp
  integration/builder_tester.cpp
  integration/ccnot_tester.cpp
  integration/deuteron_variational_tester.cpp
  integration/draw_tester.cpp
  integration/ghz_nisq_tester.cpp
  integration/gradient_tester.cpp
  integration/grover_test.cpp
  integration/nlopt_tester.cpp
  integration/qpe_ftqc.cpp
  integration/qpe_nisq.cpp
  integration/qubit_allocation.cpp
  integration/vqe_tester.cpp
  integration/bug67_vqe_then_sample.cpp
  integration/bug77_vqe_with_shots.cpp
  integration/bug116_cusv_measure_bug.cpp
  integration/async_tester.cpp
  integration/negative_controls_tester.cpp
  integration/observe_result_tester.cpp
  integration/noise_tester.cpp
  integration/get_state_tester.cpp
  integration/encoding_tester.cpp
  integration/measure_reset_tester.cpp
  integration/mps_to_host_tester.cpp
  qir/NVQIRTester.cpp  
  qis/QubitQISTester.cpp
  integration/kernels_tester.cpp
  common/MeasureCountsTester.cpp
  common/NoiseModelTester.cpp
  common/CompiledModuleTester.cpp
  ptsbe/PTSBESampleTester.cpp
  ptsbe/PTSBEMultiBackendTester.cpp
  integration/tracer_tester.cpp
  integration/gate_library_tester.cpp
  integration/analysis_scope_tester.cpp
)

# Make it so we can get function symbols
set (CMAKE_ENABLE_EXPORTS TRUE)

## This Macro allows us to create a test_runtime executable for
## the sources in CUDAQ_RUNTIME_TEST_SOURCE for a specific backend simulator
macro (create_tests_with_backend NVQIR_BACKEND EXTRA_BACKEND_TESTER)
  set(TEST_EXE_NAME "test_runtime_${NVQIR_BACKEND}")
  string(REPLACE "-" "_" NVQIR_BACKEND_OUT ${NVQIR_BACKEND})
  add_executable(${TEST_EXE_NAME} ${CUDAQ_GTEST_MAIN} ${CUDAQ_RUNTIME_TEST_SOURCES} ${EXTRA_BACKEND_TESTER})
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DNVQIR_BACKEND_NAME=${NVQIR_BACKEND_OUT})
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE __MATH_LONG_DOUBLE_CONSTANTS)
  target_include_directories(${TEST_EXE_NAME} PRIVATE . ..)

  # On GCC, the default is --as-needed for linking, and therefore the
  # nvqir-simulation plugin may not get picked up. This works as is on clang
  # But here we add it purposefully to avoid that bug.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(${TEST_EXE_NAME} PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
  endif()
  target_link_libraries(${TEST_EXE_NAME}
    PUBLIC
    nvqir-${NVQIR_BACKEND}
    nvqir
    cudaq
    cudaq-platform-default
    cudaq-builder
    cudaq-mlir-runtime
    gtest_main)
  set(TEST_LABELS "")
  set(TEST_PROCESSORS 1)
  if (${NVQIR_BACKEND} STREQUAL "qpp" OR ${NVQIR_BACKEND} STREQUAL "dm")
    set(TEST_PROCESSORS ${CUDAQ_TEST_OMP_SLOTS})
  endif()
  if (${NVQIR_BACKEND} STREQUAL "qpp")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_SIMULATION_SCALAR_FP64)
  endif()
  if (${NVQIR_BACKEND} STREQUAL "dm")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_BACKEND_DM -DCUDAQ_SIMULATION_SCALAR_FP64)
  endif()
  if (${NVQIR_BACKEND} STREQUAL "stim")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_BACKEND_STIM -DCUDAQ_SIMULATION_SCALAR_FP64)
    target_include_directories(${TEST_EXE_NAME} PRIVATE
      ${CMAKE_SOURCE_DIR}/runtime/nvqir/stim
      ${CMAKE_SOURCE_DIR}/tpls/Stim/src)
    target_link_libraries(${TEST_EXE_NAME} PRIVATE libstim)
  endif()
  if (${NVQIR_BACKEND} STREQUAL "tensornet")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_BACKEND_TENSORNET -DCUDAQ_SIMULATION_SCALAR_FP64)
    set(TEST_LABELS "gpu_required")
  endif()
  if (${NVQIR_BACKEND} STREQUAL "tensornet-mps")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_BACKEND_TENSORNET -DCUDAQ_BACKEND_TENSORNET_MPS -DCUDAQ_SIMULATION_SCALAR_FP64)
    set(TEST_LABELS "gpu_required")
  endif()
  if (${NVQIR_BACKEND} STREQUAL "tensornet-fp32")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_BACKEND_TENSORNET -DCUDAQ_SIMULATION_SCALAR_FP32)
    set(TEST_LABELS "gpu_required")
  endif()
  if (${NVQIR_BACKEND} STREQUAL "tensornet-mps-fp32")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_BACKEND_TENSORNET -DCUDAQ_BACKEND_TENSORNET_MPS -DCUDAQ_SIMULATION_SCALAR_FP32)
    set(TEST_LABELS "gpu_required")
  endif()
  if (${NVQIR_BACKEND} STREQUAL "custatevec-fp32")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_BACKEND_CUSTATEVEC_FP32 -DCUDAQ_SIMULATION_SCALAR_FP32)
    target_link_libraries(${TEST_EXE_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
    set(TEST_LABELS "gpu_required")
  endif()
  if ("${TEST_LABELS}" STREQUAL "")
    gtest_discover_tests(${TEST_EXE_NAME} PROPERTIES PROCESSORS ${TEST_PROCESSORS} DISCOVERY_TIMEOUT 120)
  else()
    gtest_discover_tests(${TEST_EXE_NAME} PROPERTIES LABELS "${TEST_LABELS}" PROCESSORS ${TEST_PROCESSORS} RESOURCE_LOCK "gpu" DISCOVERY_TIMEOUT 120)
  endif()
endmacro()

# We will always have the QPP backend, create a tester for it
create_tests_with_backend(qpp backends/QPPTester.cpp)
create_tests_with_backend(dm backends/QPPDMTester.cpp)
create_tests_with_backend(stim backends/StimTester.cpp)

if (cuStateVec_FOUND)
  find_program(NVIDIA_SMI "nvidia-smi")
  if(${NVIDIA_SMI} STREQUAL "NVIDIA_SMI-NOTFOUND")
    # libcustatevec.so has linkage to libnvdia-ml.so.1, which is part of NVIDIA driver.
    # On a build system without NVIDIA GPUs, this lib cannot be resolved;
    # hence, linking these test executables will fail.
    # On these CPU-only build systems, we directly link the nvidia-ml-dev library, which provides
    # stub symbols for the actual driver library, to these test executables.
    # Running these tests will ultimately requires a system with GPUs, i.e., the proper nvidia-ml lib
    # will be loaded by the runtime linker/loader.
    find_package(CUDAToolkit REQUIRED)
    # make sure CUDA_TOOLKIT_ROOT_DIR is set
    if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
      set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_BIN_DIR}/..)
    endif()
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      set(NVIDIA_ML_PATH ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/stubs)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(NVIDIA_ML_PATH ${CUDA_TOOLKIT_ROOT_DIR}/targets/sbsa-linux/lib/stubs)
    else()
      message(FATAL_ERROR "Neither x86_64 nor aarch64 was detected." )
    endif()
    # Find nvidia-ml manually
    find_library(NVIDIA_ML NAMES nvidia-ml PATHS ${NVIDIA_ML_PATH})
    message(STATUS "NVIDIA ML lib: ${NVIDIA_ML}")
    # Inject the file during build and remove it after build (install)
    file(CREATE_LINK ${NVIDIA_ML} ${CMAKE_CURRENT_BINARY_DIR}/libnvidia-ml.so.1 SYMBOLIC)
    link_directories(${CMAKE_CURRENT_BINARY_DIR})
    install(CODE "file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/libnvidia-ml.so.1)")
  endif()

  create_tests_with_backend(custatevec-fp32 "")
  # Given that the fp32 and fp64 difference is largely inherited
  # from a dependency, we omit fp64 tests here and rely on the
  # dependency to to validate that functionality.
  # create_tests_with_backend(custatevec-fp64 "")

  add_executable(test_mqpu ${CUDAQ_GTEST_MAIN} mqpu/mqpu_tester.cpp)
  # Need to force the link to nvqir-qpp here if gcc.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_mqpu PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
  endif()
  target_link_libraries(test_mqpu
    PRIVATE
    cudaq
    cudaq-builder
    cudaq-platform-mqpu
    cudaq-mlir-runtime
    nvqir-custatevec-fp64
    gtest_main)
  gtest_discover_tests(test_mqpu PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")

  # Test CUDAQ_OBSERVE_FROM_SAMPLING=ON mode.
  # (term-by-term expectation value calculation by applying change-of-basis gates then reverting them)
  add_executable(test_custatevec_observe_from_sampling
    integration/builder_tester.cpp
    integration/deuteron_variational_tester.cpp
    integration/gradient_tester.cpp
    integration/nlopt_tester.cpp
  )
  target_include_directories(test_custatevec_observe_from_sampling PRIVATE .)
  target_compile_definitions(test_custatevec_observe_from_sampling
    PRIVATE -DNVQIR_BACKEND_NAME=custatevec_fp32 -DCUDAQ_SIMULATION_SCALAR_FP32)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_custatevec_observe_from_sampling PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
  endif()
  target_link_libraries(test_custatevec_observe_from_sampling
    PRIVATE
    cudaq
    cudaq-builder
    cudaq-platform-default
    cudaq-mlir-runtime
    nvqir-custatevec-fp32
    gtest_main)
  # Run this test with "CUDAQ_OBSERVE_FROM_SAMPLING=1"
  gtest_discover_tests(test_custatevec_observe_from_sampling TEST_SUFFIX _DirectObserve PROPERTIES ENVIRONMENT "CUDAQ_OBSERVE_FROM_SAMPLING=1" PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")

  if (MPI_CXX_FOUND)
    # Count the number of GPUs
    find_program(NVIDIA_SMI "nvidia-smi")
    if(NVIDIA_SMI)
      execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
      # Only build this test if we have more than 1 GPU
      if (${NGPUS} GREATER_EQUAL 2)
        add_executable(test_mpi ${CUDAQ_GTEST_MAIN} mqpu/mpi_mqpu_tester.cpp)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
          target_link_options(test_mpi PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
        endif()
        target_link_libraries(test_mpi
            PRIVATE
            cudaq
            cudaq-platform-mqpu
            cudaq-mlir-runtime
            nvqir-custatevec-fp32
            gtest_main)
          configure_file("mqpu/run_mpi.sh.in"
                    "${CMAKE_BINARY_DIR}/unittests/nvqpp/run_mpi.sh" @ONLY)
        add_test(NAME MPITest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 bash ${CMAKE_BINARY_DIR}/unittests/nvqpp/run_mpi.sh)
        set_tests_properties(MPITest PROPERTIES LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
      endif()
    endif()

    add_executable(test_gpu_get_state ${CUDAQ_GTEST_MAIN} gpu/get_state_tester.cu)
    # Need to force the link to nvqir-qpp here if gcc.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
      target_link_options(test_gpu_get_state PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
    endif()
    target_compile_options(test_gpu_get_state PRIVATE -Wno-suggest-override -Wno-attributes)
    target_include_directories(test_gpu_get_state PRIVATE .)
    target_link_libraries(test_gpu_get_state
      PRIVATE
      cudaq
      cudaq-builder
      cudaq-platform-mqpu
      cudaq-mlir-runtime
      nvqir-custatevec-fp64
      gtest_main)
      gtest_discover_tests(test_gpu_get_state PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
  endif()
endif()

if(TARGET nvqir-tensornet)
  message(STATUS "Building cutensornet backend tests.")
  create_tests_with_backend(tensornet "")
  create_tests_with_backend(tensornet-mps "")
  create_tests_with_backend(tensornet-fp32 "")
  create_tests_with_backend(tensornet-mps-fp32 "")
  if (MPI_CXX_FOUND)
    # Count the number of GPUs
    find_program(NVIDIA_SMI "nvidia-smi")
    if(NVIDIA_SMI)
      execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
      # Only build this test if we have more than 1 GPUs
      if (${NGPUS} GREATER_EQUAL 2)
        message(STATUS "Building cutensornet MPI tests.")
        add_executable(test_tensornet_mpi mpi/tensornet_mpi_tester.cpp)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
          target_link_options(test_tensornet_mpi PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
        endif()
        target_link_libraries(test_tensornet_mpi
            PRIVATE
            cudaq
            cudaq-platform-default
            cudaq-mlir-runtime
            nvqir-tensornet
            gtest)
        add_test(NAME TensornetMPITest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 ${CMAKE_BINARY_DIR}/unittests/nvqpp/test_tensornet_mpi)
        set_tests_properties(TensornetMPITest PROPERTIES LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
      endif() # NGPUS
    endif() # NVIDIA_SMI
  endif() # MPI_CXX_FOUND

  # Test CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=ON mode (on a few test cases that have cudaq::observe)
  add_executable(test_tensornet_observe_path_reuse
    integration/builder_tester.cpp
    integration/deuteron_variational_tester.cpp
    integration/observe_result_tester.cpp
    integration/noise_tester.cpp # This test contains noisy observe test cases.
  )
  target_include_directories(test_tensornet_observe_path_reuse PRIVATE .)
  target_compile_definitions(test_tensornet_observe_path_reuse
                             PRIVATE -DNVQIR_BACKEND_NAME=tensornet)
  target_compile_definitions(test_tensornet_observe_path_reuse PRIVATE -DCUDAQ_BACKEND_TENSORNET)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_tensornet_observe_path_reuse PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
  endif()
  target_link_libraries(test_tensornet_observe_path_reuse
    PRIVATE
    cudaq
    cudaq-builder
    cudaq-platform-default
    cudaq-mlir-runtime
    nvqir-tensornet
    gtest_main)
  # Run this test with "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=TRUE"
  gtest_discover_tests(test_tensornet_observe_path_reuse TEST_SUFFIX _PathReuse PROPERTIES ENVIRONMENT "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=ON" PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
endif()

# build the test qudit execution manager
add_subdirectory(qudit)
add_executable(test_qudit ${CUDAQ_GTEST_MAIN} qudit/SimpleQuditTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_qudit PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
endif()
target_link_libraries(test_qudit
  PRIVATE
  cudaq
  cudaq-platform-default
  cudaq-mlir-runtime
  cudaq-em-qudit
  nvqir-qpp
  gtest_main)
gtest_discover_tests(test_qudit PROPERTIES PROCESSORS ${CUDAQ_TEST_OMP_SLOTS} DISCOVERY_TIMEOUT 120)

# build the test photonics execution manager
add_executable(test_photonics ${CUDAQ_GTEST_MAIN} photonics/PhotonicsTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_photonics PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
endif()
target_link_libraries(test_photonics
  PRIVATE
  cudaq
  cudaq-platform-default
  cudaq-mlir-runtime
  cudaq-em-photonics
  nvqir-qpp
  gtest_main)
gtest_discover_tests(test_photonics PROPERTIES PROCESSORS ${CUDAQ_TEST_OMP_SLOTS} DISCOVERY_TIMEOUT 120)

add_executable(test_ptsbe ${CUDAQ_GTEST_MAIN}
  ptsbe/KrausSelectionTester.cpp
  ptsbe/KrausTrajectoryTester.cpp
  ptsbe/ShotAllocationTester.cpp
  ptsbe/PTSBEInterfaceTester.cpp
  ptsbe/TraceConversionTester.cpp
  ptsbe/MergeTasksWithTrajectoryTester.cpp
  ptsbe/ExecutePTSBETester.cpp
  ptsbe/PTSSamplingStrategyTester.cpp
  ptsbe/NoiseExtractorTester.cpp
  ptsbe/TrajectoryDeduplicationTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_ptsbe PRIVATE -Wl,--no-as-needed)
endif()
target_include_directories(test_ptsbe PRIVATE .)
target_link_libraries(test_ptsbe
  PRIVATE
  cudaq
  cudaq-platform-default
  cudaq-mlir-runtime
  nvqir
  nvqir-qpp
  fmt::fmt-header-only
  gtest_main)
gtest_discover_tests(test_ptsbe PROPERTIES PROCESSORS ${CUDAQ_TEST_OMP_SLOTS} DISCOVERY_TIMEOUT 120)

# Create an executable for MPI UnitTests
# (only if MPI was found, i.e., the builtin plugin is available)
if (MPI_CXX_FOUND)
  set(CUDAQ_MPI_TEST_SOURCES
    mpi/mpi_tester.cpp
  )
  add_executable(test_mpi_plugin ${CUDAQ_MPI_TEST_SOURCES})
  set(NUM_PROCS 4)
  target_compile_definitions(test_mpi_plugin PRIVATE -DNUM_PROCS=${NUM_PROCS})
  target_link_libraries(test_mpi_plugin
    PRIVATE
    cudaq
    cudaq-platform-default
    cudaq-mlir-runtime
    nvqir-qpp
    gtest
  )
  target_link_options(test_mpi_plugin PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
  # Check if `--allow-run-as-root` is supported (OpenMPI)
  # Note: MPICH doesn't need `--allow-run-as-root`.
  execute_process(COMMAND ${MPIEXEC} --allow-run-as-root -np ${NUM_PROCS} hostname ERROR_VARIABLE CHECK_ALLOW_RUN_AS_ROOT_RESULTS ERROR_STRIP_TRAILING_WHITESPACE OUTPUT_QUIET)
  if ("${CHECK_ALLOW_RUN_AS_ROOT_RESULTS}" STREQUAL "")
    set(MPI_EXEC_CMD_ARGS "--allow-run-as-root")
  endif()

  add_test(NAME MPIApiTest COMMAND ${MPIEXEC} ${MPI_EXEC_CMD_ARGS} -np ${NUM_PROCS} ${CMAKE_BINARY_DIR}/unittests/nvqpp/test_mpi_plugin)
  math(EXPR MPI_API_TEST_SLOTS "${NUM_PROCS} * ${CUDAQ_TEST_OMP_SLOTS}")
  set_tests_properties(MPIApiTest PROPERTIES PROCESSORS ${MPI_API_TEST_SLOTS})
endif()

if (CUDA_FOUND AND TARGET nvqir-dynamics)

  # Multi-QPU evolve_async test
  add_executable(test_evolve_async mqpu/dynamics_async_tester.cpp)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_evolve_async PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
  endif()
  target_compile_definitions(test_evolve_async PRIVATE -DCUDAQ_ANALOG_TARGET)
  target_link_libraries(test_evolve_async
   PRIVATE
   cudaq-operator
   cudaq
   cudaq-platform-mqpu
   cudaq-mlir-runtime
   nvqir-dynamics
   cuDensityMat::cuDensityMat
   CUDA::cudart
   CUDA::cublas
   cuTensor::cuTensor
   CUDA::cusparse
   cuTensorNet::cuTensorNet
   gtest_main
   )
  # Count the number of GPUs
  find_program(NVIDIA_SMI "nvidia-smi")
  if(NVIDIA_SMI)
    execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
    # Only add this test if we have more than 1 GPU
    if (${NGPUS} GREATER_EQUAL 2)
      gtest_discover_tests(test_evolve_async PROPERTIES LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")

      # Multi-GPU batching test: this test needs MPI as well
      if (MPI_CXX_FOUND)
        add_executable(test_dynamics_mpi_batching mqpu/dynamics_mpi_batching_tester.cpp)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
          target_link_options(test_dynamics_mpi_batching PRIVATE -Wl,--no-as-needed)
        endif()
        target_compile_definitions(test_dynamics_mpi_batching PRIVATE -DCUDAQ_ANALOG_TARGET)
        target_link_libraries(test_dynamics_mpi_batching
          PRIVATE
          cudaq-operator
          cudaq
          cudaq-platform-mqpu
          cudaq-mlir-runtime
          nvqir-dynamics
          cuDensityMat::cuDensityMat
          CUDA::cudart
          CUDA::cublas
          cuTensor::cuTensor
          CUDA::cusparse
          cuTensorNet::cuTensorNet
          gtest_main
          )

        add_test(NAME DynamicsMPIBatchingTest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 ${CMAKE_BINARY_DIR}/unittests/nvqpp/test_dynamics_mpi_batching)
        set_tests_properties(DynamicsMPIBatchingTest PROPERTIES LABELS "gpu_required;mgpus_required" RESOURCE_LOCK "gpu")
      endif()
    endif()
  endif()
endif()

add_subdirectory(backends)
add_subdirectory(plugins)

if (CUDAQ_ENABLE_PYTHON)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "find_package(Python) not run?")
  endif()

  execute_process(COMMAND ${Python_EXECUTABLE} -c "import openfermionpyscf"
    OUTPUT_VARIABLE PYSCF_output
    ERROR_VARIABLE  PYSCF_error
    RESULT_VARIABLE PYSCF_result)

  if(NOT ${PYSCF_result} EQUAL 0)
    message(STATUS "OpenFermion PySCF not available for chemistry tests.")
    return()
  endif()

  if(NOT TARGET cudaq-pyscf)
    message(STATUS "cudaq-pyscf plugin not enabled; skipping chemistry tests.")
    return()
  endif()

  message(STATUS "OpenFermion PySCF found, enabling chemistry tests.")
  add_executable(test_domains ${CUDAQ_GTEST_MAIN} domains/ChemistryTester.cpp)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_domains PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
  endif()
  target_compile_definitions(test_domains PRIVATE -DNVQIR_BACKEND_NAME=qpp -DCUDAQ_SIMULATION_SCALAR_FP64)
  target_include_directories(test_domains PRIVATE . ..)
  # Embedded Python links to system framework, not venv, so needs explicit sitelib.
  set_property(TARGET test_domains PROPERTY ENVIRONMENT
    "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${Python_SITELIB}")
  target_link_libraries(test_domains
    PRIVATE
    cudaq
    cudaq-platform-default
    cudaq-mlir-runtime
    nvqir nvqir-qpp
    cudaq-operator
    cudaq-chemistry
    cudaq-pyscf
    gtest_main)
  gtest_discover_tests(test_domains
    TEST_SUFFIX _Sampling PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${Python_SITELIB}" PROCESSORS ${CUDAQ_TEST_OMP_SLOTS} DISCOVERY_TIMEOUT 120)
endif()
