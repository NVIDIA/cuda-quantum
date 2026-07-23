# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(plugin_root ${CMAKE_CURRENT_BINARY_DIR})
set(plugin_lib_dir ${plugin_root}/lib)
set(plugin_target_dir ${plugin_root}/targets)
set(plugin_python_package cudaq_example_mock_observe_qpu)
set(plugin_python_project cudaq-example-mock-observe-qpu)
set(plugin_python_entry_point mock_observe_qpu)
set(plugin_python_description
    "CUDA-Q reference external custom QPU plugin for server-side observe().")
set(plugin_python_install_name mock-observe-qpu)

file(MAKE_DIRECTORY ${plugin_lib_dir} ${plugin_target_dir})
configure_file(targets/mock_observe_qpu.yml.in
               ${plugin_target_dir}/mock_observe_qpu.yml
               @ONLY)
configure_file(pyproject.toml.in
               ${plugin_root}/pyproject.toml
               @ONLY)
configure_file(python/__init__.py.in
               ${plugin_root}/__init__.py
               @ONLY)
configure_file(python/__main__.py.in
               ${plugin_root}/__main__.py
               @ONLY)

# Single shared library registers both the custom QPU and ServerHelper.
# Auto-loaded as libcudaq-<platform-qpu>-qpu.so when the target is activated.
add_library(cudaq-example-qpu-mock_observe_qpu EXCLUDE_FROM_ALL SHARED
  MockObserveQPU.cpp)
set_target_properties(cudaq-example-qpu-mock_observe_qpu
  PROPERTIES
    OUTPUT_NAME cudaq-mock_observe_qpu-qpu
    LIBRARY_OUTPUT_DIRECTORY ${plugin_lib_dir}
    RUNTIME_OUTPUT_DIRECTORY ${plugin_lib_dir}
    ARCHIVE_OUTPUT_DIRECTORY ${plugin_lib_dir})
target_include_directories(cudaq-example-qpu-mock_observe_qpu
  PRIVATE ${PROJECT_SOURCE_DIR}/runtime)
target_link_libraries(cudaq-example-qpu-mock_observe_qpu
  PRIVATE
    cudaq
    cudaq-common
    cudaq-logger
    cudaq-operator
    cudaq-mlir-runtime
    cudaq-platform-default
    nvqir)

if(NOT PYTHON_EXECUTABLE AND Python3_EXECUTABLE)
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

if(LLVM_EXTERNAL_LIT)
  set(cudaq_mock_observe_qpu_lit ${LLVM_EXTERNAL_LIT})
else()
  set(cudaq_mock_observe_qpu_lit ${LLVM_TOOLS_BINARY_DIR}/llvm-lit)
endif()

set(CUDAQ_PLUGIN_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tests/lit.site.cfg.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/tests/lit.site.cfg.py
               @ONLY)

set(cudaq_mock_observe_qpu_check_depends cudaq-example-qpu-mock_observe_qpu)
foreach(dep nvq++ cudaq-install-plugin FileCheck CopyPythonFiles
            CUDAQuantumPythonModules)
  if(TARGET ${dep})
    list(APPEND cudaq_mock_observe_qpu_check_depends ${dep})
  endif()
endforeach()

add_custom_target(check-mock-observe-qpu
  COMMAND ${CMAKE_COMMAND} -E env
          "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
          "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{DYLD_LIBRARY_PATH}"
          ${cudaq_mock_observe_qpu_lit} ${LLVM_LIT_ARGS}
          ${CMAKE_CURRENT_BINARY_DIR}/tests
  DEPENDS ${cudaq_mock_observe_qpu_check_depends}
  USES_TERMINAL)
