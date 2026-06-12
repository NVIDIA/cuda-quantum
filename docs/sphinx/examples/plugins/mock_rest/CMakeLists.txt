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
set(plugin_python_package cudaq_example_mock_rest)
set(plugin_python_project cudaq-example-mock-rest)
set(plugin_python_entry_point mock_rest)
set(plugin_python_description
    "CUDA-Q reference REST-style external QPU plugin.")
set(plugin_python_install_name mock-rest)

file(MAKE_DIRECTORY ${plugin_lib_dir} ${plugin_target_dir})
configure_file(targets/mock_rest.yml.in
               ${plugin_target_dir}/mock_rest.yml
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

add_library(cudaq-example-serverhelper-mock_rest EXCLUDE_FROM_ALL SHARED
  MockRestServerHelper.cpp)
set_target_properties(cudaq-example-serverhelper-mock_rest
  PROPERTIES
    OUTPUT_NAME cudaq-serverhelper-mock_rest
    LIBRARY_OUTPUT_DIRECTORY ${plugin_lib_dir}
    RUNTIME_OUTPUT_DIRECTORY ${plugin_lib_dir}
    ARCHIVE_OUTPUT_DIRECTORY ${plugin_lib_dir})
target_include_directories(cudaq-example-serverhelper-mock_rest
  PRIVATE ${PROJECT_SOURCE_DIR}/runtime)
target_link_libraries(cudaq-example-serverhelper-mock_rest
  PRIVATE cudaq-common cudaq-logger)

if(NOT PYTHON_EXECUTABLE AND Python3_EXECUTABLE)
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

if(LLVM_EXTERNAL_LIT)
  set(cudaq_mock_rest_lit ${LLVM_EXTERNAL_LIT})
else()
  set(cudaq_mock_rest_lit ${LLVM_TOOLS_BINARY_DIR}/llvm-lit)
endif()

set(CUDAQ_PLUGIN_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../tests/lit.site.cfg.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/tests/lit.site.cfg.py
               @ONLY)

set(cudaq_mock_rest_check_depends cudaq-example-serverhelper-mock_rest)
foreach(dep nvq++ cudaq-install-plugin FileCheck CopyPythonFiles
            CUDAQuantumPythonModules)
  if(TARGET ${dep})
    list(APPEND cudaq_mock_rest_check_depends ${dep})
  endif()
endforeach()

add_custom_target(check-mock-rest
  COMMAND ${CMAKE_COMMAND} -E env
          "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
          "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{DYLD_LIBRARY_PATH}"
          ${cudaq_mock_rest_lit} ${LLVM_LIT_ARGS}
          ${CMAKE_CURRENT_BINARY_DIR}/tests
  DEPENDS ${cudaq_mock_rest_check_depends}
  USES_TERMINAL)
