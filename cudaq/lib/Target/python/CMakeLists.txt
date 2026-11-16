# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Python 3 REQUIRED COMPONENTS Interpreter)

set(_target_py "${CMAKE_BINARY_DIR}/python/cudaq/core/_target.py")
# Symlink the pure Python dataclasses to the build directory.
add_custom_command(
  OUTPUT "${_target_py}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/_target.py"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/python/cudaq/core"
  COMMAND "${CMAKE_COMMAND}" -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/_target.py" "${_target_py}")
add_custom_target(CUDAQTargetPythonStaging ALL DEPENDS "${_target_py}")
# Install the pure Python dataclasses next to the other `cudaq.core` modules.
install(FILES _target.py
  DESTINATION cudaq/core COMPONENT CUDAQCorePythonModules)

# `cudaq.core._target` used to be a compiled extension module: make sure to
# remove them from both the build directory and the install prefix.
# TODO: this can probably be removed once no old build or install is left.
file(GLOB _stale_target_libs
  "${CMAKE_BINARY_DIR}/python/cudaq/core/_target*.so"
  "${CMAKE_BINARY_DIR}/python/cudaq/core/_target*.pyd")
if(_stale_target_libs)
  file(REMOVE ${_stale_target_libs})
endif()
install(CODE [[
  file(GLOB _stale_target_libs
    "${CMAKE_INSTALL_PREFIX}/cudaq/core/_target*.so"
    "${CMAKE_INSTALL_PREFIX}/cudaq/core/_target*.pyd")
  if(_stale_target_libs)
    file(REMOVE ${_stale_target_libs})
  endif()
]] COMPONENT CUDAQCorePythonModules)

# The nanobind casters between those dataclasses and the C++ types.
add_library(CUDAQTargetPython INTERFACE)
target_sources(CUDAQTargetPython INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/CompileTargetCaster.cpp")
target_include_directories(CUDAQTargetPython INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_include_directories(CUDAQTargetPython SYSTEM INTERFACE
  $<TARGET_PROPERTY:reflectcpp,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(CUDAQTargetPython INTERFACE CUDAQTargetCatalog)

# The casters import `cudaq.core._target` at runtime.
add_dependencies(CUDAQTargetPython CUDAQTargetPythonStaging)

if(CUDAQ_BUILD_TESTS)
  if(CUDAQ_TARGET_STANDALONE_BUILD)
    # If standalone, shadow any installed `cudaq` on purpose to resolve `cudaq.core._target` properly
    file(WRITE "${CMAKE_BINARY_DIR}/python/cudaq/__init__.py" "")
  endif()
  add_test(NAME ctest-target-python
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
      "${Python_EXECUTABLE}" -m pytest -p no:cacheprovider
      "${CMAKE_CURRENT_SOURCE_DIR}/test")
  set_tests_properties(ctest-target-python PROPERTIES
    LABELS target-python TIMEOUT 60)
endif()
