# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

include(HandleLLVMOptions)
include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the cudaq
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=cudaq.mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(CUDAQuantumPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cudaq/mlir"
  TD_FILE dialects/QuakeOps.td
  SOURCES
    dialects/quake.py
  DIALECT_NAME quake)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cudaq/mlir"
  TD_FILE dialects/CCOps.td
  SOURCES
    dialects/cc.py
  DIALECT_NAME cc)

declare_mlir_python_extension(CUDAQuantumPythonSources.Extension
  MODULE_NAME _quakeDialects
  ADD_TO_PARENT CUDAQuantumPythonSources
  SOURCES
    CUDAQuantumExtension.cpp
    ../runtime/common/py_ExecutionContext.cpp
    ../runtime/common/py_NoiseModel.cpp
    ../runtime/common/py_ObserveResult.cpp
    ../runtime/common/py_SampleResult.cpp
    ../runtime/cudaq/algorithms/py_draw.cpp
    ../runtime/cudaq/algorithms/py_observe_async.cpp
    ../runtime/cudaq/algorithms/py_optimizer.cpp
    ../runtime/cudaq/algorithms/py_sample_async.cpp
    ../runtime/cudaq/algorithms/py_state.cpp
    ../runtime/cudaq/algorithms/py_translate.cpp
    ../runtime/cudaq/algorithms/py_utils.cpp
    ../runtime/cudaq/algorithms/py_vqe.cpp
    ../runtime/cudaq/platform/JITExecutionCache.cpp
    ../runtime/cudaq/platform/py_alt_launch_kernel.cpp
    ../runtime/cudaq/qis/py_execution_manager.cpp
    ../runtime/cudaq/qis/py_qubit_qis.cpp
    ../runtime/cudaq/spin/py_matrix.cpp
    ../runtime/cudaq/spin/py_spin_op.cpp
    ../runtime/cudaq/target/py_runtime_target.cpp
    ../runtime/cudaq/target/py_testing_utils.cpp
    ../runtime/mlir/py_register_dialects.cpp
    ../runtime/utils/PyRemoteRESTQPU.cpp
    ../runtime/utils/PyRemoteSimulatorQPU.cpp
    ../runtime/utils/PyRestRemoteClient.cpp
    ../utils/LinkedLibraryHolder.cpp
    ../../runtime/cudaq/platform/common/QuantumExecutionQueue.cpp
    ../../runtime/cudaq/platform/default/rest_server/RemoteRuntimeClient.cpp
    ../../runtime/cudaq/platform/orca/OrcaQPU.cpp

  EMBED_CAPI_LINK_LIBS
   CUDAQuantumMLIRCAPI
   MLIRCAPIExecutionEngine
  PRIVATE_LINK_LIBS
   OptCodeGen
   OptTransforms
   MLIRPass
   CUDAQTargetConfigUtil
)

target_include_directories(CUDAQuantumPythonSources.Extension INTERFACE 
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
# Instantiation of Python module
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

## The Python bindings module for Quake dialect depends on CUDAQ libraries 
## which it can't locate since they are in "../../lib" and the 'rpath' is set
## to '$ORIGIN' by default.
set(_origin_prefix "\$ORIGIN")
if(APPLE)
  set(_origin_prefix "@loader_path")
endif()

if (NOT SKBUILD)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/../../lib:$ORIGIN/../../lib/plugins")
  set_property(TARGET CUDAQuantumPythonModules.extension._quakeDialects.dso 
    APPEND PROPERTY INSTALL_RPATH "${_origin_prefix}/../../../lib:${_origin_prefix}/../../../lib/plugins")
else()
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/../cuda_quantum.libs")
  set_property(TARGET CUDAQuantumPythonModules.extension._quakeDialects.dso 
    APPEND PROPERTY INSTALL_RPATH "${_origin_prefix}/../../../lib:${_origin_prefix}/../../../cuda_quantum.libs")
endif()

