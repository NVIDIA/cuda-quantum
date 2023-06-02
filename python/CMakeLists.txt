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

message(STATUS "Found Python version ${Python_VERSION}. Building CUDA Quantum Python API with ${Python_INCLUDE_DIRS}")

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_BUILD_TYPE "Debug")
endif()
pybind11_add_module(_pycudaq 
                     _cudaq.cpp 
                     utils/LinkedLibraryHolder.cpp 
                     utils/TestingUtils.cpp
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
      cudaq-common
      cudaq-builder 
      cudaq-em-qir 
      cudaq-platform-default 
      fmt::fmt-header-only)

install(TARGETS _pycudaq DESTINATION ${CMAKE_INSTALL_PREFIX})
if (SKBUILD)
  install(FILES __init__.py DESTINATION ${CMAKE_INSTALL_PREFIX})
else()
  install(FILES cudaq.py DESTINATION ${CMAKE_INSTALL_PREFIX})
  file(COPY cudaq.py DESTINATION ${CMAKE_BINARY_DIR}/python)
endif()

install(FILES domains/chemistry.py DESTINATION ${CMAKE_INSTALL_PREFIX}/domains/)
file(COPY domains/chemistry.py DESTINATION ${CMAKE_BINARY_DIR}/python/domains)

if(APPLE)
  set_target_properties(
     _pycudaq PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
else()
  set_target_properties(_pycudaq PROPERTIES INSTALL_RPATH
                                          "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
endif()

if (NOT CUDAQ_DISABLE_CPP_FRONTEND)
  add_subdirectory(domains/plugins)
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
    message(WARNING "CUDA Quantum Python Warning - CUDAQ_BUILD_TESTS=TRUE but can't find numpy or pytest modules required for testing.")
  endif()
endif()
