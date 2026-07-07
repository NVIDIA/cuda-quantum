# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(utils)

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

if (CUDA_FOUND)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(LLVM_ENABLE_ASSERTIONS)
  set(CUDAQ_ASSERTIONS_ENABLED "1")
else()
  set(CUDAQ_ASSERTIONS_ENABLED "")
endif()

set(METADATA_FILE "${CMAKE_BINARY_DIR}/python/cudaq/_metadata.py" )
add_custom_command(
  OUTPUT "${METADATA_FILE}"
  COMMAND ${CMAKE_COMMAND}
  -DMETADATA_FILE="${METADATA_FILE}"
  -DCUDA_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
  -DASSERTIONS_ENABLED=${CUDAQ_ASSERTIONS_ENABLED}
  -P ${CMAKE_CURRENT_SOURCE_DIR}/metadata.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metadata.cmake
)

# cudaq/mlir is excluded: the MLIR machinery in extension/ stages it, and
# a second producer for the same outputs races it.
file(GLOB_RECURSE _cudaq_package_sources CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/cudaq"
  "${CMAKE_CURRENT_SOURCE_DIR}/cudaq/*.py")
list(FILTER _cudaq_package_sources EXCLUDE REGEX "^mlir/")
cudaq_stage_python_sources(CUDAQPythonPackageStaging
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cudaq"
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq"
  SOURCES ${_cudaq_package_sources})

# The pycudaq-mlir lit suite and the interop pytest run from the build tree.
# mlir/generated holds artifacts lit writes into the source tree at config
# load; they are discovered from there, not staged.
file(GLOB_RECURSE _cudaq_test_sources CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*")
list(FILTER _cudaq_test_sources EXCLUDE REGEX
  "^mlir/generated/|__pycache__|\\.pyc$|\\.DS_Store$")
cudaq_stage_python_sources(CUDAQPythonTestStaging
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/tests"
  SOURCES ${_cudaq_test_sources})

add_custom_target(CUDAQPythonStaging ALL DEPENDS "${METADATA_FILE}")
add_dependencies(CUDAQPythonStaging
  CUDAQPythonPackageStaging CUDAQPythonTestStaging)
add_dependencies(CUDAQuantumPythonModules CUDAQPythonStaging)

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

if (cuDensityMat_FOUND)
  add_subdirectory(runtime/cudaq/dynamics)
endif()
