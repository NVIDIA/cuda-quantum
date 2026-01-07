# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(utils)

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

# [RFC]:
# Check how to solve this better than just disable the warning for the whole directory.
# If this is better addressed after updating to a newer LLVM version, track as an issue on GitHub.
add_compile_options(-Wno-ambiguous-reversed-operator)

if (CMAKE_COMPILER_IS_GNUCXX)
  # Deals with warnings in gcc12 - may not be needed for gcc 13.
  # To enable this also for clang, make sure to install the 
  # corresponding llvm-*-dev package (no change needed for the LLVM build).
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto=auto")
endif()

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME 1)

add_subdirectory(extension)

file(GLOB_RECURSE PYTHON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

if (CUDA_FOUND)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

set(METADATA_FILE "${CMAKE_BINARY_DIR}/python/cudaq/_metadata.py" )
add_custom_target(
  CopyPythonFiles ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/python
  COMMAND ${CMAKE_COMMAND} 
  -DMETADATA_FILE="${METADATA_FILE}" 
  -DCUDA_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR} 
  -P ${CMAKE_CURRENT_SOURCE_DIR}/metadata.cmake
  DEPENDS ${PYTHON_SOURCES}
  BYPRODUCTS "${METADATA_FILE}"
)

add_dependencies(CUDAQuantumPythonModules CopyPythonFiles)

add_subdirectory(runtime/cudaq/domains/plugins)

if (NOT SKBUILD)
  install(DIRECTORY cudaq DESTINATION .)
endif()
install(FILES "${METADATA_FILE}" DESTINATION cudaq)

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

add_subdirectory(runtime/interop)

if (CUDENSITYMAT_ROOT AND CUDA_FOUND)
  add_subdirectory(runtime/cudaq/dynamics)
endif()
