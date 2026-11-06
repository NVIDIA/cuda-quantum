# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Python 3 REQUIRED COMPONENTS Interpreter)

# The compile target types are pure Python dataclasses; the nanobind casters
# that convert them are compiled into the main Python extension.
set(_target_py "${CMAKE_BINARY_DIR}/python/cudaq/core/_target.py")
add_custom_command(
  OUTPUT "${_target_py}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/_target.py"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/python/cudaq/core"
  COMMAND "${CMAKE_COMMAND}" -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/_target.py" "${_target_py}")
add_custom_target(CUDAQTargetPython ALL DEPENDS "${_target_py}")
install(FILES _target.py
  DESTINATION cudaq/core COMPONENT CUDAQCorePythonModules)

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
