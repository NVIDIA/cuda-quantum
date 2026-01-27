# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set (CMAKE_CXX_FLAGS
     "${CMAKE_CXX_FLAGS} -Wno-attributes -Wno-ctad-maybe-unsupported")
# clear any flags for static linking
SET(CMAKE_EXE_LINKER_FLAGS "")
SET(CMAKE_SHARED_LINKER_FLAGS "")

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
  integration/measure_reset_tester.cpp
  integration/mps_to_host_tester.cpp
  qir/NVQIRTester.cpp
  qis/QubitQISTester.cpp
  integration/kernels_tester.cpp
  common/MeasureCountsTester.cpp
  common/NoiseModelTester.cpp
  integration/tracer_tester.cpp
  integration/gate_library_tester.cpp
)

# Make it so we can get function symbols
set (CMAKE_ENABLE_EXPORTS TRUE)

## This Macro allows us to create a test_runtime executable for
## the sources in CUDAQ_RUNTIME_TEST_SOURCE for a specific backend simulator
macro (create_tests_with_backend NVQIR_BACKEND EXTRA_BACKEND_TESTER)
  set(TEST_EXE_NAME "test_runtime_${NVQIR_BACKEND}")
  string(REPLACE "-" "_" NVQIR_BACKEND_OUT ${NVQIR_BACKEND})
  add_executable(${TEST_EXE_NAME} main.cpp ${CUDAQ_RUNTIME_TEST_SOURCES} ${EXTRA_BACKEND_TESTER})
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DNVQIR_BACKEND_NAME=${NVQIR_BACKEND_OUT})
  target_compile_definitions(${TEST_EXE_NAME} PRIVATE __MATH_LONG_DOUBLE_CONSTANTS)
  target_include_directories(${TEST_EXE_NAME} PRIVATE .)

  # On GCC, the default is --as-needed for linking, and therefore the
  # nvqir-simulation plugin may not get picked up. This works as is on clang
  # But here we add it purposefully to avoid that bug.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(${TEST_EXE_NAME} PRIVATE -Wl,--no-as-needed)
  endif()
  target_link_libraries(${TEST_EXE_NAME}
    PUBLIC
    nvqir-${NVQIR_BACKEND}
    nvqir
    cudaq
    fmt::fmt-header-only
    cudaq-platform-default
    cudaq-builder
    gtest_main)
  set(TEST_LABELS "")
  if (${NVQIR_BACKEND} STREQUAL "qpp")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_SIMULATION_SCALAR_FP64)
  endif()
  if (${NVQIR_BACKEND} STREQUAL "dm")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_BACKEND_DM -DCUDAQ_SIMULATION_SCALAR_FP64)
  endif()
  if (${NVQIR_BACKEND} STREQUAL "stim")
    target_compile_definitions(${TEST_EXE_NAME} PRIVATE -DCUDAQ_BACKEND_STIM -DCUDAQ_SIMULATION_SCALAR_FP64)
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
    gtest_discover_tests(${TEST_EXE_NAME})
  else()
    gtest_discover_tests(${TEST_EXE_NAME} PROPERTIES LABELS "${TEST_LABELS}")
  endif()
endmacro()

# We will always have the QPP backend, create a tester for it
create_tests_with_backend(qpp backends/QPPTester.cpp)
create_tests_with_backend(dm backends/QPPDMTester.cpp)
create_tests_with_backend(stim "")

if (CUSTATEVEC_ROOT AND CUDA_FOUND)
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

  add_executable(test_mqpu main.cpp mqpu/mqpu_tester.cpp)
  # Need to force the link to nvqir-qpp here if gcc.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_mqpu PRIVATE -Wl,--no-as-needed)
  endif()
  target_link_libraries(test_mqpu
    PRIVATE
    cudaq
    cudaq-builder
    cudaq-platform-mqpu
    nvqir-custatevec-fp64
    gtest_main)
  gtest_discover_tests(test_mqpu PROPERTIES LABELS "gpu_required")

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
                             PRIVATE -DNVQIR_BACKEND_NAME=custatevec_fp32)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_custatevec_observe_from_sampling PRIVATE -Wl,--no-as-needed)
  endif()
  target_link_libraries(test_custatevec_observe_from_sampling
    PRIVATE
    cudaq
    cudaq-builder
    cudaq-platform-default
    nvqir-custatevec-fp32
    gtest_main)
  # Run this test with "CUDAQ_OBSERVE_FROM_SAMPLING=1"
  gtest_discover_tests(test_custatevec_observe_from_sampling TEST_SUFFIX _DirectObserve PROPERTIES ENVIRONMENT "CUDAQ_OBSERVE_FROM_SAMPLING=1" PROPERTIES LABELS "gpu_required")

  if (MPI_CXX_FOUND)
    # Count the number of GPUs
    find_program(NVIDIA_SMI "nvidia-smi")
    if(NVIDIA_SMI)
      execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
      # Only build this test if we have more than 1 GPU
      if (${NGPUS} GREATER_EQUAL 2)
        add_executable(test_mpi main.cpp mqpu/mpi_mqpu_tester.cpp)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
          target_link_options(test_mpi PRIVATE -Wl,--no-as-needed)
        endif()
        target_link_libraries(test_mpi
            PRIVATE
            cudaq
            cudaq-platform-mqpu
            nvqir-custatevec-fp32
            gtest_main)
          configure_file("mqpu/run_mpi.sh.in"
                    "${CMAKE_BINARY_DIR}/unittests/run_mpi.sh" @ONLY)
        add_test(NAME MPITest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 bash ${CMAKE_BINARY_DIR}/unittests/run_mpi.sh)
        set_tests_properties(MPITest PROPERTIES LABELS "gpu_required;mgpus_required")
      endif()
    endif()

    add_executable(test_gpu_get_state main.cpp gpu/get_state_tester.cu)
    # Need to force the link to nvqir-qpp here if gcc.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
      target_link_options(test_gpu_get_state PRIVATE -Wl,--no-as-needed)
    endif()
    target_compile_options(test_gpu_get_state PRIVATE -Wno-suggest-override -Wno-attributes)
    target_include_directories(test_gpu_get_state PRIVATE .)
    target_link_libraries(test_gpu_get_state
      PRIVATE
      cudaq
      cudaq-builder
      cudaq-platform-mqpu
      nvqir-custatevec-fp64
      gtest_main)
      gtest_discover_tests(test_gpu_get_state PROPERTIES LABELS "gpu_required")
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
          target_link_options(test_tensornet_mpi PRIVATE -Wl,--no-as-needed)
        endif()
        target_link_libraries(test_tensornet_mpi
            PRIVATE
            cudaq
            fmt::fmt-header-only
            cudaq-platform-default
            nvqir-tensornet
            gtest)
        add_test(NAME TensornetMPITest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 ${CMAKE_BINARY_DIR}/unittests/test_tensornet_mpi)
        set_tests_properties(TensornetMPITest PROPERTIES LABELS "gpu_required;mgpus_required")
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
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_tensornet_observe_path_reuse PRIVATE -Wl,--no-as-needed)
  endif()
  target_link_libraries(test_tensornet_observe_path_reuse
    PRIVATE
    cudaq
    cudaq-builder
    cudaq-platform-default
    nvqir-tensornet
    gtest_main)
  # Run this test with "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=TRUE"
  gtest_discover_tests(test_tensornet_observe_path_reuse TEST_SUFFIX _PathReuse PROPERTIES ENVIRONMENT "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=ON" PROPERTIES LABELS "gpu_required")
endif()

# Create an executable for SpinOp UnitTests
set(CUDAQ_SPIN_TEST_SOURCES 
   spin_op/SpinOpTester.cpp
   operators/utils.cpp
)
add_executable(test_spin main.cpp ${CUDAQ_SPIN_TEST_SOURCES})
target_link_libraries(test_spin
  PRIVATE 
  cudaq
  cudaq-operator
  gtest_main
  cudaq-platform-default)
target_include_directories(test_spin PRIVATE operators)
gtest_discover_tests(test_spin)

# Create an executable for operators UnitTests
set(CUDAQ_OPERATOR_TEST_SOURCES
   operators/utils.cpp
   operators/scalar_op.cpp
   operators/matrix_op.cpp
   operators/spin_op.cpp
   operators/boson_op.cpp
   operators/fermion_op.cpp
   operators/conversions.cpp
   operators/product_op.cpp
   operators/sum_op.cpp
   operators/rydberg_hamiltonian.cpp
   operators/manipulation.cpp
)
add_executable(test_operators main.cpp ${CUDAQ_OPERATOR_TEST_SOURCES})
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_operators PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(test_operators
  PRIVATE
  cudaq-operator
  cudaq
  gtest_main
  cudaq-platform-default)
gtest_discover_tests(test_operators)

if (CUDA_FOUND)
  find_package(CUDAToolkit REQUIRED)

  find_library(CUTENSOR_LIB
    NAMES   cutensor libcutensor.so.2
    HINTS
        ${CUTENSOR_ROOT}/lib64
        ${CUTENSOR_ROOT}/lib
        ${CUTENSOR_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
        ${CUTENSOR_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}
  )

  find_library(CUTENSORNET_LIB
    NAMES   cutensornet libcutensornet.so.2
    HINTS   
        ${CUTENSORNET_ROOT}/lib64
        ${CUTENSORNET_ROOT}/lib
        ${CUTENSORNET_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
        ${CUTENSORNET_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}
        ${CUDENSITYMAT_ROOT}/lib/
)

  # Create an executable for dynamics UnitTests
  set(CUDAQ_DYNAMICS_TEST_SOURCES
    dynamics/test_RungeKuttaIntegrator.cpp
    dynamics/test_CuDensityMatState.cpp
    dynamics/test_CuDensityMatTimeStepper.cpp
    dynamics/test_CuDensityMatExpectation.cpp
    dynamics/test_EvolveSingle.cpp
    dynamics/test_EvolveApi.cpp
    dynamics/test_EvolveBatchedImpl.cpp
    dynamics/test_BatchedEvolveApi.cpp
    dynamics/test_BatchingDetection.cpp
  )
  add_executable(test_dynamics main.cpp ${CUDAQ_DYNAMICS_TEST_SOURCES})
  target_compile_definitions(test_dynamics PRIVATE -DCUDAQ_ANALOG_TARGET)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
   target_link_options(test_dynamics PRIVATE -Wl,--no-as-needed)
  endif()
  target_link_libraries(test_dynamics
   PRIVATE
   cudaq-operator
   cudaq
   nvqir-dynamics
   ${CUDENSITYMAT_ROOT}/lib/libcudensitymat.so.0
   CUDA::cudart
   CUDA::cublas
   ${CUTENSOR_LIB}
   CUDA::cusparse
   ${CUTENSORNET_LIB}
   gtest_main
   fmt::fmt-header-only)
  target_include_directories(test_dynamics PRIVATE ${CMAKE_SOURCE_DIR}/runtime/nvqir/cudensitymat)
  gtest_discover_tests(test_dynamics PROPERTIES LABELS "gpu_required")

  # Multi-QPU evolve_async test
  add_executable(test_evolve_async mqpu/dynamics_async_tester.cpp)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
   target_link_options(test_evolve_async PRIVATE -Wl,--no-as-needed)
  endif()
  target_compile_definitions(test_evolve_async PRIVATE -DCUDAQ_ANALOG_TARGET)
  target_link_libraries(test_evolve_async
   PRIVATE
   cudaq-operator
   cudaq
   cudaq-platform-mqpu
   nvqir-dynamics
   ${CUDENSITYMAT_ROOT}/lib/libcudensitymat.so.0
   CUDA::cudart
   CUDA::cublas
   ${CUTENSOR_LIB}
   CUDA::cusparse
   ${CUTENSORNET_LIB}
   gtest_main
   fmt::fmt-header-only)
  # Count the number of GPUs
  find_program(NVIDIA_SMI "nvidia-smi")
  if(NVIDIA_SMI)
    execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
    # Only add this test if we have more than 1 GPU
    if (${NGPUS} GREATER_EQUAL 2)
      gtest_discover_tests(test_evolve_async PROPERTIES LABELS "gpu_required;mgpus_required")

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
          nvqir-dynamics
          ${CUDENSITYMAT_ROOT}/lib/libcudensitymat.so.0
          CUDA::cudart
          CUDA::cublas
          ${CUTENSOR_LIB}
          CUDA::cusparse
          ${CUTENSORNET_LIB}
          gtest_main
          fmt::fmt-header-only)

        add_test(NAME DynamicsMPIBatchingTest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 ${CMAKE_BINARY_DIR}/unittests/test_dynamics_mpi_batching)
        set_tests_properties(DynamicsMPIBatchingTest PROPERTIES LABELS "gpu_required;mgpus_required")
      endif()
    endif()
  endif()
endif()

add_subdirectory(plugin)

# build the test qudit execution manager
add_subdirectory(qudit)
add_executable(test_qudit main.cpp qudit/SimpleQuditTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_qudit PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(test_qudit
  PRIVATE
  cudaq
  cudaq-platform-default
  cudaq-em-qudit
  gtest_main)
gtest_discover_tests(test_qudit)

# build the test photonics execution manager
add_executable(test_photonics main.cpp photonics/PhotonicsTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_photonics PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(test_photonics
  PRIVATE
  cudaq
  cudaq-platform-default
  cudaq-em-photonics
  nvqir-qpp
  gtest_main)
gtest_discover_tests(test_photonics)

add_executable(test_utils main.cpp utils/UtilsTester.cpp utils/Matrix.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_utils PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(test_utils
  PRIVATE
  cudaq
  cudaq-platform-default
  cudaq-em-photonics
  nvqir
  nvqir-qpp fmt::fmt-header-only
  gtest_main)
gtest_discover_tests(test_utils)

# Test for thread-local execution context storage
add_executable(test_exec_ctx_thread main.cpp common/ExecutionContextThreadTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_exec_ctx_thread PRIVATE -Wl,--no-as-needed)
endif()
target_include_directories(test_exec_ctx_thread PRIVATE .)
target_link_libraries(test_exec_ctx_thread
  PRIVATE
  cudaq
  cudaq-platform-default
  fmt::fmt-header-only
  gtest_main)
gtest_discover_tests(test_exec_ctx_thread)

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
    nvqir-qpp
    gtest
  )
  target_link_options(test_mpi_plugin PRIVATE -Wl,--no-as-needed)
  # Check if `--allow-run-as-root` is supported (OpenMPI)
  # Note: MPICH doesn't need `--allow-run-as-root`.
  execute_process(COMMAND ${MPIEXEC} --allow-run-as-root -np ${NUM_PROCS} hostname ERROR_VARIABLE CHECK_ALLOW_RUN_AS_ROOT_RESULTS ERROR_STRIP_TRAILING_WHITESPACE OUTPUT_QUIET)
  if ("${CHECK_ALLOW_RUN_AS_ROOT_RESULTS}" STREQUAL "")
    set(MPI_EXEC_CMD_ARGS "--allow-run-as-root")
  endif()

  add_test(NAME MPIApiTest COMMAND ${MPIEXEC} ${MPI_EXEC_CMD_ARGS} -np ${NUM_PROCS} ${CMAKE_BINARY_DIR}/unittests/test_mpi_plugin)
endif()

add_subdirectory(backends)
add_subdirectory(Optimizer)
add_subdirectory(output_record)
add_subdirectory(target_config)

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

  message(STATUS "OpenFermion PySCF found, enabling chemistry tests.")
  add_executable(test_domains main.cpp domains/ChemistryTester.cpp)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_domains PRIVATE -Wl,--no-as-needed)
  endif()
  target_compile_definitions(test_domains PRIVATE -DNVQIR_BACKEND_NAME=qpp -DCUDAQ_SIMULATION_SCALAR_FP64)
  target_include_directories(test_domains PRIVATE .)
  set_property(TARGET test_domains PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
  target_link_libraries(test_domains
    PRIVATE
    cudaq
    cudaq-platform-default
    nvqir nvqir-qpp
    cudaq-operator
    cudaq-chemistry
    cudaq-pyscf
    gtest_main)
  gtest_discover_tests(test_domains
    TEST_SUFFIX _Sampling PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
endif()
