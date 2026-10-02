# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

option(CUDAQ_BUILD_TESTS "Build the CUDA-Q pulse test suites" ON)
if(NOT CUDAQ_BUILD_TESTS)
  return()
endif()

enable_testing()

if(LLVM_RUNTIME_OUTPUT_INTDIR)
  set(CUDAQ_PULSE_TOOLS_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}")
else()
  set(CUDAQ_PULSE_TOOLS_DIR
      "${CUDAQ_PULSE_BINARY_DIR}/core/mlir/tools/cudaq-pulse-opt")
endif()
configure_file(
  ${CUDAQ_PULSE_SOURCE_DIR}/test/lit.site.cfg.py.in
  ${CUDAQ_PULSE_BINARY_DIR}/test/lit.site.cfg.py
  @ONLY)

# Locate the lit launcher from the configured interpreter's own environment.
# lit ships only as a console script (there is no runnable `lit.__main__`), and
# the cudaq-devel wheel's bundled `llvm-lit` carries a shebang pointing at its
# build interpreter, which is absent here. Restrict the search to the
# interpreter's bin directory and run the launcher through that interpreter, so
# the launcher and its shebang always match the installed wheels.
get_filename_component(_cudaq_pulse_python_bindir "${Python_EXECUTABLE}" DIRECTORY)
find_program(CUDAQ_PULSE_LIT_EXECUTABLE NAMES lit llvm-lit
  HINTS "${_cudaq_pulse_python_bindir}"
  NO_DEFAULT_PATH
  REQUIRED)
set(CUDAQ_PULSE_LIT_PYTHONPATH "" CACHE PATH
    "Optional Python module path for the LLVM lit launcher")

add_custom_target(check-pulse-mlir
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${CUDAQ_PULSE_LIT_PYTHONPATH}"
          ${Python_EXECUTABLE} ${CUDAQ_PULSE_LIT_EXECUTABLE} -sv
          ${CUDAQ_PULSE_BINARY_DIR}/test
  DEPENDS cudaq-pulse-opt
  COMMENT "Running CUDA-Q pulse MLIR tests"
  USES_TERMINAL)

add_custom_target(check-pulse-python
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${CUDAQ_PULSE_PYTHONPATH}"
          ${Python_EXECUTABLE} -m pytest
          ${CUDAQ_PULSE_SOURCE_DIR}/tests -m "not gpu" -q
  DEPENDS _cudaq_pulse_native
  COMMENT "Running CUDA-Q pulse Python unit tests"
  USES_TERMINAL)

add_custom_target(check-pulse
  DEPENDS check-pulse-mlir check-pulse-python)

add_test(NAME PulsePythonUnitTests
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${CUDAQ_PULSE_PYTHONPATH}"
          ${Python_EXECUTABLE} -m pytest
          ${CUDAQ_PULSE_SOURCE_DIR}/tests -m "not gpu" -q)
set_tests_properties(PulsePythonUnitTests PROPERTIES LABELS "pulse;unit")

add_test(NAME PulseMLIRTests
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${CUDAQ_PULSE_LIT_PYTHONPATH}"
          ${Python_EXECUTABLE} ${CUDAQ_PULSE_LIT_EXECUTABLE} -sv
          ${CUDAQ_PULSE_BINARY_DIR}/test)
set_tests_properties(PulseMLIRTests PROPERTIES LABELS "pulse;mlir")

if(TARGET cudm_runtime)
  # This private CTest driver is not installed. Without --gpu it verifies SDK
  # linkage/version discovery; with --gpu it creates and destroys the basic
  # cuDensityMat descriptors used by the preview runtime.
  add_executable(cudaq-pulse-cudm-smoke
    ${CUDAQ_PULSE_SOURCE_DIR}/tests/runtime/cudm_runtime_smoke.cpp)
  target_link_libraries(cudaq-pulse-cudm-smoke PRIVATE
    cudm_runtime
    CUDA::cudart)
  set_target_properties(cudaq-pulse-cudm-smoke PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:cudm_runtime>")

  add_test(NAME PulseCuDensityMatLinkSmoke
    COMMAND cudaq-pulse-cudm-smoke)
  set_tests_properties(PulseCuDensityMatLinkSmoke PROPERTIES
    LABELS "pulse;gpu-build")

  set(_cudaq_pulse_gpu_available FALSE)
  find_program(CUDAQ_PULSE_NVIDIA_SMI_EXECUTABLE NAMES nvidia-smi)
  if(CUDAQ_PULSE_NVIDIA_SMI_EXECUTABLE AND TARGET CUDA::cudart AND
     NOT CMAKE_CROSSCOMPILING)
    execute_process(
      COMMAND ${CUDAQ_PULSE_NVIDIA_SMI_EXECUTABLE} -L
      RESULT_VARIABLE _cudaq_pulse_nvidia_smi_result
      OUTPUT_VARIABLE _cudaq_pulse_nvidia_smi_output
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_cudaq_pulse_nvidia_smi_result EQUAL 0 AND
       _cudaq_pulse_nvidia_smi_output MATCHES "GPU [0-9]+:")
      try_run(
        _cudaq_pulse_cuda_probe_result
        _cudaq_pulse_cuda_probe_compiled
        ${CMAKE_CURRENT_BINARY_DIR}/cuda-device-probe
        SOURCES
          ${CUDAQ_PULSE_SOURCE_DIR}/tests/runtime/cuda_device_probe.cpp
        LINK_LIBRARIES CUDA::cudart)
      if(_cudaq_pulse_cuda_probe_compiled AND
         _cudaq_pulse_cuda_probe_result EQUAL 0)
        set(_cudaq_pulse_gpu_available TRUE)
      endif()
    endif()
  endif()

  if(_cudaq_pulse_gpu_available)
    message(STATUS "CUDA-Q pulse numerical GPU tests enabled")

    add_test(NAME PulseCuDensityMatGpuSmoke
      COMMAND cudaq-pulse-cudm-smoke --gpu)
    set_tests_properties(PulseCuDensityMatGpuSmoke PROPERTIES
      LABELS "pulse;gpu"
      SKIP_RETURN_CODE 77)

    add_test(NAME PulsePythonGpuTests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CUDAQ_PULSE_PYTHONPATH}"
              "CUDAQ_PULSE_BUILD_DIR=${CMAKE_BINARY_DIR}"
              "CUDAQ_PULSE_LLVM_BIN=${LLVM_TOOLS_BINARY_DIR}"
              ${Python_EXECUTABLE} -m pytest
              ${CUDAQ_PULSE_SOURCE_DIR}/tests/runtime -m gpu -q)
    set_tests_properties(PulsePythonGpuTests PROPERTIES LABELS "pulse;gpu")

    add_custom_target(check-pulse-gpu
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
              -L gpu
      DEPENDS cudaq-pulse-cudm-smoke _cudaq_pulse_native
      COMMENT "Running CUDA-Q pulse cuDensityMat and numerical GPU tests"
      USES_TERMINAL)
  else()
    message(STATUS
      "CUDA-Q pulse numerical GPU tests disabled (no usable NVIDIA GPU/CUDA runtime)")
    add_custom_target(check-pulse-gpu
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
              -R PulseCuDensityMatLinkSmoke
      DEPENDS cudaq-pulse-cudm-smoke
      COMMENT "GPU unavailable; running CUDA-Q pulse SDK linkage test only"
      USES_TERMINAL)
  endif()
  add_dependencies(check-pulse check-pulse-gpu)
else()
  add_custom_target(check-pulse-gpu
    COMMAND ${CMAKE_COMMAND} -E echo
            "CUDA-Q pulse GPU tests disabled: CUDA/cuDensityMat not available"
    COMMENT "CUDA-Q pulse GPU tests are disabled")
endif()
