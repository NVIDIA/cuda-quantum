# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(LIBRARY_NAME _pycudaq)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

find_package(Python COMPONENTS Interpreter)

if (NOT Python_FOUND)
  message(STATUS "Python interpreter not found. Skipping Python API build.")
  return()
endif()

if (${Python_VERSION} VERSION_LESS 3.0.0)
  message(STATUS "Found Python version ${Python_VERSION}. Version must be greater than 3.0.0, skipping Python API build.")
  return()
endif()

add_subdirectory(src)

message(STATUS "Found Python version ${Python_VERSION}. Building CUDA Quantum Python API with ${Python_INCLUDE_DIRS}")

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

# set(pybind11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tpls/pybind11/tools/)
find_package(pybind11 REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../tpls/pybind11/)
pybind11_add_module(_pycudaq 
                     utils/LinkedLibraryHolder.cpp 
                     src/_cudaq.cpp 
                     src/runtime/cudaq/builder/py_kernel_builder.cpp
                     src/runtime/cudaq/builder/py_QuakeValue.cpp
                     src/runtime/cudaq/algorithms/py_observe.cpp
                     src/runtime/cudaq/algorithms/py_sample.cpp
                     src/runtime/cudaq/algorithms/py_optimizer.cpp
                     src/runtime/cudaq/algorithms/py_state.cpp
                     src/runtime/cudaq/algorithms/py_vqe.cpp
                     src/runtime/cudaq/spin/py_matrix.cpp
                     src/runtime/cudaq/spin/py_spin_op.cpp
                     src/runtime/cudaq/kernels/py_chemistry.cpp
                     src/runtime/common/py_ObserveResult.cpp
                     src/runtime/common/py_SampleResult.cpp                     
                     src/runtime/common/py_NoiseModel.cpp)

target_compile_options(_pycudaq PRIVATE -Wno-attributes)                     
target_include_directories(_pycudaq PRIVATE .)
target_link_libraries(_pycudaq 
    PRIVATE 
      nvqir
      cudaq 
      cudaq-builder 
      cudaq-em-qir 
      cudaq-platform-default 
      fmt::fmt-header-only)

install(TARGETS _pycudaq DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES cudaq.py DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES domains/chemistry.py DESTINATION ${CMAKE_INSTALL_PREFIX}/domains/)

if(APPLE)
  set_target_properties(
    _pycudaq PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
else()
  set_target_properties(_pycudaq PROPERTIES INSTALL_RPATH
                                          "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
endif()

# Copy to the build dir for testing
file(COPY cudaq.py DESTINATION ${CMAKE_BINARY_DIR}/python)
file(COPY domains/chemistry.py DESTINATION ${CMAKE_BINARY_DIR}/python/domains)

add_subdirectory(domains/plugins)

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
    message(WARNING "CUDA Quantum Python Warning - CUDAQ_BUILD_TESTS=TRUE but can't find numpy or pytest modules required for testing.")
  endif()
endif()
