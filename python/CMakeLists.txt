# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(utils)

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

include(AddMLIRPython)

# [RFC]:
# Check how to solve this better than just disable the warning for the whole directory.
# If this is better addressed after updating to a newer LLVM version, track as an issue on GitHub.
add_compile_options(-Wno-ambiguous-reversed-operator)

# Specifies that all MLIR packages are co-located under the cudaq
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=cudaq.mlir.")

if (CMAKE_COMPILER_IS_GNUCXX)
  # Deals with warnings in gcc12 - may not be needed for gcc 13.
  # To enable this also for clang, make sure to install the 
  # corresponding llvm-*-dev package (no change needed for the LLVM build).
  add_definitions(-flto=auto)
endif()

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME 1)

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(CUDAQuantumPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cudaq/mlir"
  TD_FILE dialects/QuakeOps.td
  SOURCES
    dialects/quake.py
  DIALECT_NAME quake)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cudaq/mlir"
  TD_FILE dialects/CCOps.td
  SOURCES
    dialects/cc.py
  DIALECT_NAME cc)

declare_mlir_python_extension(CUDAQuantumPythonSources.Extension
  MODULE_NAME _quakeDialects
  ADD_TO_PARENT CUDAQuantumPythonSources
  SOURCES
   CUDAQuantumExtension.cpp
   runtime/common/py_ExecutionContext.cpp
   runtime/common/py_NoiseModel.cpp
   runtime/common/py_ObserveResult.cpp
   runtime/common/py_SampleResult.cpp
   runtime/cudaq/algorithms/py_draw.cpp
   runtime/cudaq/algorithms/py_observe_async.cpp
   runtime/cudaq/algorithms/py_optimizer.cpp
   runtime/cudaq/algorithms/py_sample_async.cpp
   runtime/cudaq/algorithms/py_state.cpp
   runtime/cudaq/algorithms/py_vqe.cpp
   runtime/cudaq/platform/JITExecutionCache.cpp
   runtime/cudaq/platform/py_alt_launch_kernel.cpp
   runtime/cudaq/qis/py_execution_manager.cpp
   runtime/cudaq/qis/py_qubit_qis.cpp
   runtime/cudaq/spin/py_matrix.cpp
   runtime/cudaq/spin/py_spin_op.cpp
   runtime/cudaq/target/py_runtime_target.cpp
   runtime/cudaq/target/py_testing_utils.cpp
   runtime/mlir/py_register_dialects.cpp
   runtime/utils/PyRemoteRESTQPU.cpp
   runtime/utils/PyRemoteSimulatorQPU.cpp
   runtime/utils/PyRestRemoteClient.cpp
   utils/LinkedLibraryHolder.cpp
   ../runtime/cudaq/platform/common/QuantumExecutionQueue.cpp
   ../runtime/cudaq/platform/default/rest_server/RemoteRuntimeClient.cpp
   ../runtime/cudaq/platform/orca/OrcaQPU.cpp

  EMBED_CAPI_LINK_LIBS
   CUDAQuantumMLIRCAPI
   MLIRCAPIExecutionEngine
  PRIVATE_LINK_LIBS
   OptCodeGen
   OptTransforms
   MLIRPass
)

target_include_directories(CUDAQuantumPythonSources.Extension INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_SOURCE_DIR}/python 
    ${CMAKE_SOURCE_DIR}/python/utils
    ${CMAKE_SOURCE_DIR}/runtime
)
target_link_libraries(CUDAQuantumPythonSources.Extension INTERFACE 
    cudaq 
    cudaq-common 
    cudaq-em-default
    fmt::fmt-header-only
    unzip_util
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(CUDAQuantumPythonCAPI
  INSTALL_COMPONENT CUDAQuantumPythonModules
  INSTALL_DESTINATION cudaq/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python/cudaq/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../.."
  DECLARED_SOURCES
    CUDAQuantumPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(CUDAQuantumPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python/cudaq/mlir"
  INSTALL_PREFIX "cudaq/mlir"
  DECLARED_SOURCES
    CUDAQuantumPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    CUDAQuantumPythonCAPI
  )

file (COPY cudaq DESTINATION ${CMAKE_BINARY_DIR}/python)

## The Python bindings module for Quake dialect depends on CUDAQ libraries 
## which it can't locate since they are in "../../lib" and the 'rpath' is set
## to '$ORIGIN' by default.
set(_origin_prefix "\$ORIGIN")
if(APPLE)
  set(_origin_prefix "@loader_path")
endif()

add_subdirectory(runtime/cudaq/domains/plugins)

if (NOT SKBUILD)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/../../lib:$ORIGIN/../../lib/plugins")
  set_property(TARGET CUDAQuantumPythonModules.extension._quakeDialects.dso 
    APPEND PROPERTY INSTALL_RPATH "${_origin_prefix}/../../../lib:${_origin_prefix}/../../../lib/plugins")
else()
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/../cuda_quantum.libs")
  set_property(TARGET CUDAQuantumPythonModules.extension._quakeDialects.dso 
    APPEND PROPERTY INSTALL_RPATH "${_origin_prefix}/../../../lib:${_origin_prefix}/../../../cuda_quantum.libs")
endif()

if (NOT SKBUILD)
  install(DIRECTORY cudaq DESTINATION .)
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
