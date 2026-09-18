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

# Target configs are compiled target plugin libraries, not raw YAML text -
# see docs/sphinx/using/extending/packaging.rst. The source-of-truth YAML
# still exists (mock_rest.yml.in) but is only ever consumed at plugin *build*
# time, by cudaq-target-db-gen --plugin, which parses/validates it and
# generates a small translation unit exporting the compiled
# cudaq::config::TargetConfig. That's compiled into the plugin library below
# (mock_rest.so on Linux, mock_rest.dylib on macOS); the raw YAML itself is
# never installed or read at run/compile time.
set(plugin_target_yml ${CMAKE_CURRENT_BINARY_DIR}/mock_rest.yml)
configure_file(targets/mock_rest.yml.in
               ${plugin_target_yml}
               @ONLY)
set(plugin_target_gen_cpp ${CMAKE_CURRENT_BINARY_DIR}/mock_rest_target.gen.cpp)
add_custom_command(
  OUTPUT ${plugin_target_gen_cpp}
  COMMAND cudaq-target-db-gen --plugin -o ${plugin_target_gen_cpp}
          mock_rest=${plugin_target_yml}
  DEPENDS cudaq-target-db-gen ${plugin_target_yml}
  COMMENT "Generating mock_rest target plugin library source"
  VERBATIM)
add_library(cudaq-example-target-mock_rest EXCLUDE_FROM_ALL MODULE ${plugin_target_gen_cpp})
set_target_properties(cudaq-example-target-mock_rest
  PROPERTIES
    OUTPUT_NAME mock_rest
    PREFIX ""
    SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX}
    LIBRARY_OUTPUT_DIRECTORY ${plugin_target_dir})
target_link_libraries(cudaq-example-target-mock_rest PRIVATE CUDAQTargetConfig)

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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tests/lit.site.cfg.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/tests/lit.site.cfg.py
               @ONLY)

set(cudaq_mock_rest_check_depends
  cudaq-example-serverhelper-mock_rest
  cudaq-example-target-mock_rest)
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
