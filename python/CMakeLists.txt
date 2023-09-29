# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
add_subdirectory(utils)

set(LIBRARY_NAME _pycudaq)
if (NOT SKBUILD)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/../../lib:$ORIGIN/../../lib/plugins")
else()
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/../cuda_quantum.libs")
endif()

find_package(Python COMPONENTS Interpreter)

if (NOT Python_FOUND)
  message(STATUS "Python interpreter not found. Skipping Python API build.")
  return()
endif()

if (${Python_VERSION} VERSION_LESS 3.0.0)
  message(STATUS "Found Python version ${Python_VERSION}. Version must be greater than 3.0.0, skipping Python API build.")
  return()
endif()

message(STATUS "Found Python version ${Python_VERSION}. Building CUDA Quantum Python API with ${Python_INCLUDE_DIRS}")

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_BUILD_TYPE "Debug")
endif()
pybind11_add_module(_pycudaq 
                     cudaq/_cudaq.cpp 
                     runtime/cudaq/target/py_runtime_target.cpp 
                     runtime/cudaq/target/py_testing_utils.cpp
                     runtime/cudaq/builder/py_kernel_builder.cpp
                     runtime/cudaq/builder/py_QuakeValue.cpp
                     runtime/cudaq/algorithms/py_observe.cpp
                     runtime/cudaq/algorithms/py_sample.cpp
                     runtime/cudaq/algorithms/py_optimizer.cpp
                     runtime/cudaq/algorithms/py_state.cpp
                     runtime/cudaq/algorithms/py_vqe.cpp
                     runtime/cudaq/spin/py_matrix.cpp
                     runtime/cudaq/spin/py_spin_op.cpp
                     runtime/cudaq/kernels/py_chemistry.cpp
                     runtime/common/py_ObserveResult.cpp
                     runtime/common/py_SampleResult.cpp                     
                     runtime/common/py_NoiseModel.cpp)

target_compile_options(_pycudaq PRIVATE -Wno-attributes)                     
target_include_directories(_pycudaq PRIVATE .)
target_link_libraries(_pycudaq 
    PRIVATE 
      nvqir
      cudaq 
      cudaq-builder 
      cudaq-em-default 
      cudaq-platform-default 
      cudaq-py-utils)

if (NOT SKBUILD)
  install(DIRECTORY cudaq DESTINATION .)
  install(TARGETS _pycudaq DESTINATION cudaq)
  # Also move these to the build directory for testing.
  file (COPY cudaq DESTINATION ${CMAKE_BINARY_DIR}/python/)
  add_custom_command(TARGET _pycudaq POST_BUILD 
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_pycudaq> 
              ${CMAKE_BINARY_DIR}/python/cudaq)
else()
  install(TARGETS _pycudaq LIBRARY DESTINATION cudaq)
endif()

if (NOT CUDAQ_DISABLE_CPP_FRONTEND)
  add_subdirectory(runtime/cudaq/domains/plugins)
endif()

# Do we have pytest and numpy? 
execute_process(COMMAND ${Python_EXECUTABLE} -m pytest --version
  OUTPUT_VARIABLE PYTEST_output
  ERROR_VARIABLE  PYTEST_error
  RESULT_VARIABLE PYTEST_result)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy"
  OUTPUT_VARIABLE NUMPY_output
  ERROR_VARIABLE NUMPY_error
  RESULT_VARIABLE NUMPY_result)

if(CUDAQ_BUILD_TESTS)
  if(${PYTEST_result} EQUAL 0 AND ${NUMPY_result} EQUAL 0)
    message(STATUS "Pytest and Numpy found, building Python tests.")
    add_subdirectory(tests)
  else()
    message(FATAL_ERROR "CUDA Quantum Python Warning - CUDAQ_BUILD_TESTS=TRUE but can't find numpy or pytest modules required for testing.")
  endif()
endif()
