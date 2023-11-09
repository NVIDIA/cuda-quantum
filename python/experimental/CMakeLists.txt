# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_cudaq`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_cudaq.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(CUDAQuantumPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_cudaq"
  TD_FILE dialects/QuakeOps.td
  SOURCES
    dialects/quake.py
  DIALECT_NAME quake)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_cudaq"
  TD_FILE dialects/CCOps.td
  SOURCES
    dialects/cc.py
  DIALECT_NAME cc)

declare_mlir_python_extension(CUDAQuantumPythonSources.Extension
  MODULE_NAME _quakeDialects
  ADD_TO_PARENT CUDAQuantumPythonSources
  SOURCES
   CUDAQuantumExtension.cpp
   runtime//mlir/py_register_dialects.cpp
   runtime/utils/PyRemoteRESTQPU.cpp
   runtime/cudaq/qis/py_qubit_qis.cpp
   runtime/cudaq/kernels/py_common_kernels.cpp
   runtime/cudaq/platform/py_alt_launch_kernel.cpp
   runtime/cudaq/qis/py_execution_manager.cpp 
   runtime/common/py_ExecutionContext.cpp
   runtime/cudaq/algorithms/py_sample_async.cpp
   runtime/cudaq/algorithms/py_observe_async.cpp
   runtime/cudaq/platform/JITExecutionCache.cpp
   ../utils/LinkedLibraryHolder.cpp
   ../runtime/cudaq/target/py_runtime_target.cpp 
   ../runtime/common/py_SampleResult.cpp
   ../runtime/common/py_ObserveResult.cpp
   ../runtime/common/py_NoiseModel.cpp
   ../runtime/cudaq/spin/py_matrix.cpp
   ../runtime/cudaq/spin/py_spin_op.cpp 
   ../runtime/cudaq/algorithms/py_optimizer.cpp 
   ../../runtime/cudaq/platform/common/QuantumExecutionQueue.cpp

  EMBED_CAPI_LINK_LIBS
   CUDAQuantumMLIRCAPI
   MLIRCAPIExecutionEngine
  PRIVATE_LINK_LIBS
   OptCodeGen
   OptTransforms
)

target_include_directories(CUDAQuantumPythonSources.Extension INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_SOURCE_DIR}/python 
    ${CMAKE_SOURCE_DIR}/python/utils)
target_link_libraries(CUDAQuantumPythonSources.Extension INTERFACE 
    cudaq 
    cudaq-common 
    cudaq-em-default
    fmt::fmt-header-only
    )

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(CUDAQuantumPythonCAPI
  INSTALL_COMPONENT CUDAQuantumPythonModules
  INSTALL_DESTINATION python_packages/cudaq/mlir_cudaq/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/cudaq/mlir_cudaq/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
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
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/cudaq/mlir_cudaq"
  INSTALL_PREFIX "python_packages/cudaq/mlir_cudaq"
  DECLARED_SOURCES
    CUDAQuantumPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    CUDAQuantumPythonCAPI
  )

file (COPY cudaq DESTINATION ${MLIR_BINARY_DIR}/python_packages/cudaq)

if (CUDAQ_BUILD_TESTS)
 add_subdirectory(tests)
endif()