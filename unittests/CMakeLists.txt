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

# These unit test executables instantiate `__qpu__` kernel functor structs and
# call `cudaq::sample(myKernel)` directly from C++ test code, with no `nvq++`
# / `cudaq-quake` step. The AST bridge never runs, so measurement and gate
# calls reach the inline bodies in `runtime/cudaq/qis/qubit_qis.h` on the
# host.
#
# The user-facing `--library-mode` flag has been removed, but the macro it set
# survives as an internal implementation detail.
add_compile_definitions(CUDAQ_LIBRARY_MODE)

# ctest's PROCESSORS property tells the scheduler how many CPU slots each test
# occupies. Without it, `ctest -j N` launches N OpenMP-parallel tests at once
# (2N threads competing for N cores). This value is only applied to tests that
# link OpenMP backends (qpp, dm). Non-OpenMP tests keep the default of 1.
# CI passes -DCUDAQ_TEST_OMP_SLOTS=2 alongside OMP_NUM_THREADS=2 at runtime.
set(CUDAQ_TEST_OMP_SLOTS 1 CACHE STRING
  "CPU slots per OpenMP-parallel test (should match OMP_NUM_THREADS).")

# For CUDAQTestUtils.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (CUDA_FOUND AND CUDAQ_ENABLE_REALTIME)
  add_executable(test_device_call_dispatch
    device_call/DeviceCallDispatchTester.cpp
    device_call/DeviceCallDispatchTester.cu)
  set_target_properties(test_device_call_dispatch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    ENABLE_EXPORTS ON)
  target_include_directories(test_device_call_dispatch
    PRIVATE
      .)
  target_link_libraries(test_device_call_dispatch
    PRIVATE
      cudaq-device-call-runtime
      cudaq::cudaq-realtime-dispatch
      CUDA::cudart_static
      gtest_main)

  gtest_discover_tests(test_device_call_dispatch
    PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu"
    DISCOVERY_TIMEOUT 120)

  # Pure-HOST_CALL host dispatch must work without a visible CUDA device. The
  # binary hides every device from itself (CUDA_VISIBLE_DEVICES="" in its own
  # main()), so this deliberately carries no gpu_required label: it runs on
  # GPU-less CI runners and exercises the same no-device path everywhere.
  # Plain CXX link: no .cu sources, and the shared cudaq-device-call-runtime
  # carries the dispatcher, so no CUDA device-link step is needed.
  add_executable(test_host_dispatch_no_gpu
    device_call/HostDispatchNoGpuTester.cpp)
  set_target_properties(test_host_dispatch_no_gpu PROPERTIES
    ENABLE_EXPORTS ON)
  target_link_libraries(test_host_dispatch_no_gpu
    PRIVATE
      cudaq-device-call-runtime
      CUDA::cudart_static
      gtest)

  gtest_discover_tests(test_host_dispatch_no_gpu
    DISCOVERY_TIMEOUT 120)

  # cpu_roce DeviceCallChannel test + its service-side daemon. Only when the
  # CPU RoCE transport (libibverbs) is available; the fixture itself GTEST_SKIPs
  # unless the loopback test topology is provided via environment variables.
  if (TARGET cudaq::cudaq-realtime-cpu-roce-transport)
    add_executable(cpu_roce_test_daemon
      device_call/cpu_roce_test_daemon.cpp)
    target_include_directories(cpu_roce_test_daemon
      PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    # host-dispatch.a contains relocatable device code (host_dispatcher.cu); the
    # final link must run the CUDA device-link step even though this target has
    # no .cu sources. Same "bridge as .cpp with CUDA device linking" pattern as
    # the realtime hololink/hsb bridge tools.
    set_target_properties(cpu_roce_test_daemon PROPERTIES
      LINKER_LANGUAGE CUDA
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(cpu_roce_test_daemon
      PRIVATE
        cudaq::cudaq-realtime-cpu-roce-transport
        cudaq::cudaq-realtime-host-dispatch
        CUDA::cudart_static
        Threads::Threads)

    # The cpu_roce GoogleTest fixture builds as its OWN executable rather than
    # being linked into test_device_call_dispatch, so it can be compiled
    # independently of DeviceCallDispatchTester.cu. This lets the build-only CI
    # job compile-check just this harness (target test_cpu_roce_device_call)
    # without pulling in the shared-memory device_call test, which #4565 keeps
    # out of CI by design. CUDA device-link pattern matches cpu_roce_test_daemon
    # (links the realtime dispatch static lib's relocatable device code).
    add_executable(test_cpu_roce_device_call
      device_call/CpuRoceChannelTester.cpp)
    target_include_directories(test_cpu_roce_device_call
      PRIVATE
        .)
    set_target_properties(test_cpu_roce_device_call PROPERTIES
      LINKER_LANGUAGE CUDA
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(test_cpu_roce_device_call
      PRIVATE
        cudaq-device-call-runtime
        cudaq::cudaq-realtime-dispatch
        CUDA::cudart_static
        gtest_main)
    target_compile_definitions(test_cpu_roce_device_call
      PRIVATE CPU_ROCE_DAEMON_PATH="$<TARGET_FILE:cpu_roce_test_daemon>")
    add_dependencies(test_cpu_roce_device_call cpu_roce_test_daemon)
    # PRE_TEST discovery so the build-only CI job never executes the binary at
    # build time (no RDMA NIC there); ctest still enumerates it at test time.
    gtest_discover_tests(test_cpu_roce_device_call
      PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu"
      DISCOVERY_TIMEOUT 120
      DISCOVERY_MODE PRE_TEST)
  endif()
endif()

set(CUDAQ_GTEST_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

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
gtest_discover_tests(test_spin DISCOVERY_TIMEOUT 120)

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
target_link_libraries(test_operators
  PRIVATE
  cudaq-operator
  cudaq
  gtest_main
  cudaq-platform-default)
gtest_discover_tests(test_operators DISCOVERY_TIMEOUT 120)

if (CUDA_FOUND AND TARGET nvqir-dynamics)

  # Create an executable for dynamics UnitTests
  set(CUDAQ_DYNAMICS_TEST_SOURCES
    dynamics/test_RungeKuttaIntegrator.cpp
    dynamics/test_CrankNicolsonIntegrator.cpp
    dynamics/test_MagnusIntegrator.cpp
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
  target_link_libraries(test_dynamics
   PRIVATE
   cudaq-operator
   cudaq
   cudaq-em-default
   cudaq-logger
   nvqir-dynamics
   cuDensityMat::cuDensityMat
   CUDA::cudart
   CUDA::cublas
   cuTensor::cuTensor
   CUDA::cusparse
   cuTensorNet::cuTensorNet
   gtest_main
   )
  target_include_directories(test_dynamics PRIVATE ${CMAKE_SOURCE_DIR}/runtime/nvqir/cudensitymat)
  gtest_discover_tests(test_dynamics PROPERTIES LABELS "gpu_required" RESOURCE_LOCK "gpu")
endif()

add_executable(test_utils main.cpp utils/UtilsTester.cpp utils/Matrix.cpp)
target_link_libraries(test_utils
  PRIVATE
  cudaq
  cudaq-platform-default
  cudaq-em-photonics
  nvqir
  nvqir-qpp 
  gtest_main)
gtest_discover_tests(test_utils PROPERTIES PROCESSORS ${CUDAQ_TEST_OMP_SLOTS} DISCOVERY_TIMEOUT 120)

add_executable(test_nvqir_verify main.cpp qir/NVQIRVerify.cpp)
target_link_libraries(test_nvqir_verify
  PRIVATE
    cudaq
    MLIRIR
    MLIRMathDialect
    MLIRParser
    MLIRSupport
    MLIROptLib
    cudaq-qir-verifier
    cudaq-platform-default
    gtest_main)
gtest_discover_tests(test_nvqir_verify DISCOVERY_TIMEOUT 120)

# Test for thread-local execution context storage
add_executable(test_exec_ctx_thread main.cpp common/ExecutionContextThreadTester.cpp)
target_include_directories(test_exec_ctx_thread PRIVATE .)
target_link_libraries(test_exec_ctx_thread
  PRIVATE
  cudaq
  cudaq-platform-default
  gtest_main)
gtest_discover_tests(test_exec_ctx_thread DISCOVERY_TIMEOUT 120)

add_subdirectory(nvqpp)
add_subdirectory(output_record)
add_subdirectory(target_config)
add_subdirectory(logger)
add_subdirectory(backends)

