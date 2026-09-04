# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The MLIR Python bindings for cudaq/'s own dialects (Quake, CC, QEC), built
# here as part of the cudaq/ subproject -- mirroring how LLVM/MLIR itself
# builds its Python bindings as part of the mlir/ subproject, rather than as
# an unrelated downstream consumer project. Gated by
# CUDAQ_ENABLE_PYTHON_BINDINGS (see the top-level CMakeLists.txt for the
# default derivation and the "python project requires these bindings"
# diagnostic).
#
# The python/ subproject's much larger `_quakeDialects` extension module
# (the full Python runtime surface: algorithms, operators, platform, qis,
# ...) is NOT built here: it is genuinely python/-runtime-coupled, not a
# cudaq/-core dialect binding, and stays in python/extension/. That module
# links against the single common CAPI aggregate library built here
# (cudaqMLIRCAPI) rather than building a second, duplicate aggregate --
# loading two independently-linked copies of the same MLIR/CAPI machinery
# into one Python process risks duplicate dialect/TypeID registration.

include(HandleLLVMOptions)
include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the cudaq top level
# package (the API has been embedded in a relocatable way). Matches the
# prefix used by python/extension/CMakeLists.txt so both halves assemble into
# one coherent installed "cudaq.mlir" package tree.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=cudaq.mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(CUDAQCorePythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQCorePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/QuakeOps.td
  SOURCES
    dialects/quake.py
  DIALECT_NAME quake)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQCorePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/CCOps.td
  SOURCES
    dialects/cc.py
  DIALECT_NAME cc)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQCorePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/QECOps.td
  SOURCES
    dialects/qec.py
  DIALECT_NAME qec)

# Define the register_dialects hook to populate the default DialectRegistry
# for every Context.
declare_mlir_python_extension(CUDAQCorePythonSources.SiteInitialize
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT CUDAQCorePythonSources
  SOURCES
    SiteInitialize.cpp
  PRIVATE_LINK_LIBS
    cudaqMLIR
)

################################################################################
# Instantiation of Python module
################################################################################

# See python/extension/CMakeLists.txt for why this is set explicitly.
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "cudaq")

set(CUDAQ_CORE_PYTHON_DECLARED_SOURCES
  CUDAQCorePythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.ExecutionEngine
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.complex
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.Dialects.math)

# Build the common CAPI shared library. cudaqDialectsCAPI/MLIRCAPITransforms
# are embedded explicitly (EMBED_LIBS) rather than relying on the automatic
# collection from a declared extension's EMBED_CAPI_LINK_LIBS property: the
# python/ Extension module that also needs them is declared later (in a
# sibling subdirectory processed after this one), so it cannot contribute to
# this aggregation pass. It instead links against this already-built
# cudaqMLIRCAPI target directly (see python/extension/CMakeLists.txt).
add_cudaq_python_common_capi_library(cudaqMLIRCAPI
  DECLARED_SOURCES ${CUDAQ_CORE_PYTHON_DECLARED_SOURCES}
  EMBED_LIBS
    cudaqDialectsCAPI
    MLIRCAPITransforms
)

# Namespaced alias so consumers (in this same configure, e.g.
# python/extension/CMakeLists.txt) can uniformly say cudaq::cudaqMLIRCAPI
# whether it is the in-tree target (this ALIAS) or an imported one from a
# separately-configured python/ build's find_package(CUDAQ) (via the
# NAMESPACE on the install(EXPORT ...) below) -- same pattern as
# cudaq::cudaqMLIR in cudaq/lib/Optimizer/cudaqMLIR-shlib.cmake.
add_library(cudaq::cudaqMLIRCAPI ALIAS cudaqMLIRCAPI)

install(TARGETS cudaqMLIRCAPI
  EXPORT cudaq-python-targets
  COMPONENT CUDAQuantumPythonModules
  DESTINATION lib)

add_cudaq_python_modules(CUDAQCorePythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python/cudaq/mlir"
  INSTALL_PREFIX "cudaq/mlir"
  DECLARED_SOURCES ${CUDAQ_CORE_PYTHON_DECLARED_SOURCES}
  COMMON_CAPI_LINK_LIBS
    cudaqMLIRCAPI
)

# Suppress warnings-as-errors for upstream MLIR Python extension sources
# that have minor GCC warnings (address-of-function, parentheses) in LLVM 22.
foreach(_cudaq_py_ext_target
  CUDAQCorePythonModules.extension._mlir.dso
  CUDAQCorePythonModules.extension.MLIRPythonSupport-cudaq.so)
  if(TARGET ${_cudaq_py_ext_target})
    target_compile_options(${_cudaq_py_ext_target} PRIVATE
      -Wno-error=address -Wno-error=parentheses)
  endif()
endforeach()

# Export a marker so a separately-configured python/ build (against an
# installed CUDAQ, not built together with cudaq/ in this configure) can
# detect that these bindings are present. See CUDAQPythonBindingsConfig.cmake.
# COMPONENT Development (not CUDAQuantumPythonModules): this target-definition
# file ships in the devel SDK, mirroring CUDAQPythonInteropTargets.cmake --
# the underlying cudaqMLIRCAPI binary above still ships in the runtime
# component; a from-source consumer needs both installed to the same prefix
# to actually link against it, but detecting *whether bindings were built*
# only needs the devel-wheel-side marker to be present.
install(EXPORT cudaq-python-targets
  FILE CUDAQPythonBindingsTargets.cmake
  NAMESPACE cudaq::
  DESTINATION lib/cmake/cudaq
  COMPONENT Development)
