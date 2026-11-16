# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build the Python extension using the CUDA-Q bindings helpers: 
#  1. Define the Python extension modules using standard MLIR CMake macros
#  2. Use `add_cudaq_python_common_capi_library` to define the shared library
#     containing the C-APIs that the extension modules will use. Instead of
#     embedding MLIR as upstream does, this will create a shallow library
#     linking in the libcudaqMLIR shared library.
#  3. Use `add_cudaq_python_modules` to define the Python extension modules.
#     This helper sets rpaths etc to ensure all MLIR symbols resolve properly 
#     to the libcudaqMLIR shared library.

mlir_configure_python_dev_packages()

# Co-locate the extension under the package, mirroring the cudaq.mlir layout.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=cudaq_mlir_extension.mlir.")

declare_mlir_python_sources(TrivialMLIRPythonSources)

declare_mlir_python_extension(TrivialMLIRPythonSources.Extension
  MODULE_NAME _mlirExtension
  ADD_TO_PARENT TrivialMLIRPythonSources
  SOURCES
  TrivialExtension.cpp
  PRIVATE_LINK_LIBS
  trivial-mlir
  EMBED_CAPI_LINK_LIBS
  MLIRCAPIIR)

set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "cudaq_mlir_extension")

set(_declared_sources TrivialMLIRPythonSources)

# Build the common CAPI shared library (installed next to libcudaqMLIR in lib/).
add_cudaq_python_common_capi_library(TrivialMLIRPythonCAPI
  INSTALL_COMPONENT TrivialMLIRPythonModules
  INSTALL_DESTINATION lib
  DECLARED_SOURCES ${_declared_sources})

# Build the Python extension modules against that common CAPI library.
add_cudaq_python_modules(TrivialMLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/cudaq_mlir_extension/mlir"
  INSTALL_PREFIX "cudaq_mlir_extension/mlir"
  DECLARED_SOURCES ${_declared_sources}
  COMMON_CAPI_LINK_LIBS
  TrivialMLIRPythonCAPI)
