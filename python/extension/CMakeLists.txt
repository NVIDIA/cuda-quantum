# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
    # ../runtime/mlir/py_register_dialects.cpp

  EMBED_CAPI_LINK_LIBS
   CUDAQuantumMLIRCAPI
  # PRIVATE_LINK_LIBS
  #  OptCodeGen
  #  OptTransforms
  #  MLIRPass
)

target_include_directories(CUDAQuantumPythonSources.Extension INTERFACE 
    ${CMAKE_SOURCE_DIR}/python 
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

# ## The Python bindings module for Quake dialect depends on CUDAQ libraries 
# ## which it can't locate since they are in "../../lib" and the 'rpath' is set
# ## to '$ORIGIN' by default.
# ## macOS uses @loader_path instead of $ORIGIN for RPATH.
# if(APPLE)
#   set(_origin_prefix "@loader_path")
# else()
#   set(_origin_prefix "$ORIGIN")
# endif()

# if (NOT SKBUILD)
#   list(APPEND CMAKE_INSTALL_RPATH "${_origin_prefix}/../../lib" "${_origin_prefix}/../../lib/plugins")
#   set_property(TARGET CUDAQuantumPythonModules.extension._quakeDialects.dso
#     APPEND PROPERTY INSTALL_RPATH "${_origin_prefix}/../../../lib" "${_origin_prefix}/../../../lib/plugins")
# else()
#   list(APPEND CMAKE_INSTALL_RPATH "${_origin_prefix}/../cuda_quantum.libs")
#   set_property(TARGET CUDAQuantumPythonModules.extension._quakeDialects.dso
#     APPEND PROPERTY INSTALL_RPATH "${_origin_prefix}/../../../lib" "${_origin_prefix}/../../../cuda_quantum.libs")
# endif()
