# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(CUDAQTargetPython NB_STATIC NB_DOMAIN cudaq
  py_target.cpp py_profiles.cpp py_compile_target.cpp)
target_link_libraries(CUDAQTargetPython PRIVATE CUDAQTargetConfig)
target_compile_options(CUDAQTargetPython PRIVATE -fexceptions -frtti -Wno-cast-qual)
set_target_properties(CUDAQTargetPython PROPERTIES
  OUTPUT_NAME _target
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq")
set_target_properties(nanobind-static PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
install(TARGETS CUDAQTargetPython
  LIBRARY DESTINATION cudaq COMPONENT CUDAQCorePythonModules)

if(CUDAQ_BUILD_TESTS)
  if(CUDAQ_TARGET_STANDALONE_BUILD)
    # If standalone, shadow any installed `cudaq` on purpose to resolve `cudaq._target` properly
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
