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

function(add_mlir_python_extension libname extname)
  cmake_parse_arguments(ARG
    ""
    "INSTALL_COMPONENT;INSTALL_DIR;OUTPUT_DIRECTORY"
    "SOURCES;LINK_LIBS"
    ${ARGN})

  # Use nanobind for CUDA-Q's own extension (_quakeDialects) and pybind11
  # for upstream MLIR extensions (AsyncPasses, RegisterEverything, etc.).
  if(libname MATCHES "_quakeDialects")
    nanobind_add_module(${libname} NB_STATIC ${ARG_SOURCES})
    target_compile_options(${libname} PRIVATE -frtti -fexceptions -Wno-cast-qual)
  else()
    pybind11_add_module(${libname} MODULE ${ARG_SOURCES})
    target_compile_options(${libname} PRIVATE -frtti -fexceptions)
  endif()

  set_target_properties(${libname} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${ARG_OUTPUT_DIRECTORY}
    OUTPUT_NAME "${extname}"
    NO_SONAME ON
  )

  target_link_libraries(${libname} PRIVATE ${ARG_LINK_LIBS})
  target_link_options(${libname} PRIVATE
    $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
  )

  if(ARG_INSTALL_DIR)
    install(TARGETS ${libname}
      COMPONENT ${ARG_INSTALL_COMPONENT}
      LIBRARY DESTINATION "${ARG_INSTALL_DIR}"
      RUNTIME DESTINATION "${ARG_INSTALL_DIR}"
    )
  endif()
endfunction()

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
  DIALECT_NAME quake)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cudaq/mlir"
  TD_FILE dialects/CCOps.td
  DIALECT_NAME cc)

declare_mlir_python_extension(CUDAQuantumPythonSources.Extension
  MODULE_NAME _quakeDialects
  ADD_TO_PARENT CUDAQuantumPythonSources
  SOURCES
    CUDAQuantumExtension.cpp
    ../runtime/common/py_ExecutionContext.cpp
    ../runtime/common/py_NoiseModel.cpp
    ../runtime/common/py_EvolveResult.cpp
    ../runtime/common/py_ObserveResult.cpp
    ../runtime/common/py_SampleResult.cpp
    ../runtime/common/py_Resources.cpp
    ../runtime/common/py_CustomOpRegistry.cpp
    ../runtime/common/py_AnalogHamiltonian.cpp
    ../runtime/cudaq/algorithms/py_draw.cpp
    ../runtime/cudaq/algorithms/py_evolve.cpp
    ../runtime/cudaq/algorithms/py_observe_async.cpp
    ../runtime/cudaq/algorithms/py_optimizer.cpp
    ../runtime/cudaq/algorithms/py_sample_async.cpp
    ../runtime/cudaq/algorithms/py_sample_ptsbe.cpp
    ../runtime/cudaq/algorithms/py_resource_count.cpp
    ../runtime/cudaq/algorithms/py_run.cpp
    ../../runtime/cudaq/algorithms/run.cpp      # Common Python and C++ implementation of run
    ../runtime/cudaq/algorithms/py_state.cpp
    ../runtime/cudaq/algorithms/py_translate.cpp
    ../runtime/cudaq/algorithms/py_unitary.cpp
    ../runtime/cudaq/algorithms/py_utils.cpp
    ../runtime/cudaq/platform/PythonSignalCheck.cpp
    ../runtime/cudaq/platform/py_alt_launch_kernel.cpp
    ../runtime/cudaq/qis/py_execution_manager.cpp
    ../runtime/cudaq/qis/py_pauli_word.cpp
    ../runtime/cudaq/operators/py_helpers.cpp
    ../runtime/cudaq/operators/py_matrix.cpp
    ../runtime/cudaq/operators/py_scalar_op.cpp
    ../runtime/cudaq/operators/py_spin_op.cpp
    ../runtime/cudaq/operators/py_fermion_op.cpp
    ../runtime/cudaq/operators/py_boson_op.cpp
    ../runtime/cudaq/operators/py_matrix_op.cpp
    ../runtime/cudaq/operators/py_super_op.cpp
    ../runtime/cudaq/operators/py_handlers.cpp
    ../runtime/cudaq/target/py_runtime_target.cpp
    ../runtime/cudaq/target/py_testing_utils.cpp
    ../runtime/mlir/py_register_dialects.cpp
    ../utils/LinkedLibraryHolder.cpp
    ../../runtime/common/CodeGenConfig.cpp
    ../../runtime/cudaq/platform/orca/OrcaExecutor.cpp
    ../../runtime/cudaq/platform/orca/OrcaQPU.cpp
    ../../runtime/cudaq/platform/orca/OrcaRemoteRESTQPU.cpp
    ../../runtime/cudaq/platform/orca/OrcaServerHelper.cpp
    ../../runtime/cudaq/platform/pasqal/PasqalRemoteRESTQPU.cpp
    ../../runtime/cudaq/platform/pasqal/PasqalServerHelper.cpp
    ../../runtime/cudaq/platform/quera/QuEraRemoteRESTQPU.cpp
    ../../runtime/cudaq/platform/fermioniq/FermioniqQPU.cpp
    ../../runtime/cudaq/platform/default/rest/RemoteRESTQPU.cpp
    ../../runtime/cudaq/platform/default/python/QPU.cpp
    ../../runtime/internal/compiler/ArgumentConversion.cpp
    ../../runtime/internal/compiler/CompiledModuleHelper.cpp
    ../../runtime/internal/compiler/LayoutInfo.cpp
    ../../runtime/internal/compiler/RuntimeMLIR.cpp
    ../../runtime/internal/compiler/RuntimePyMLIR.cpp
    ../../runtime/internal/compiler/JIT.cpp
    ../../runtime/internal/compiler/Compiler.cpp
    ../../runtime/internal/compiler/TracePassInstrumentation.cpp

  EMBED_CAPI_LINK_LIBS
   CUDAQuantumMLIRCAPI
   MLIRCAPIExecutionEngine
  PRIVATE_LINK_LIBS
   OptCodeGen
   OptTransforms
   MLIRPass
   CUDAQTargetConfigUtil
   cudaq-python-interop
   cudaq-platform-default
   cudaq-qir-verifier
   cudaq-mlir-runtime-headers
)

# MLIR/LLVM is built without RTTI (LLVM_ENABLE_RTTI=OFF). This file subclasses
# mlir::PassInstrumentation, so it must match LLVM's RTTI setting.
set_source_files_properties(
    ../runtime/cudaq/platform/PythonSignalCheck.cpp
    ../../runtime/internal/compiler/TracePassInstrumentation.cpp
    PROPERTIES COMPILE_FLAGS "-fno-rtti -fno-exceptions"
)


target_include_directories(CUDAQuantumPythonSources.Extension INTERFACE
    ${CMAKE_SOURCE_DIR}/python
    ${CMAKE_SOURCE_DIR}/python/utils
    ${CMAKE_SOURCE_DIR}/runtime
)
target_link_libraries(CUDAQuantumPythonSources.Extension INTERFACE 
    cudaq 
    cudaq-logger 
    cudaq-common 
    cudaq-em-default
    cudaq-em-photonics
    fmt::fmt-header-only
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

if(TARGET nanobind-static)
  target_compile_options(nanobind-static PRIVATE -Wno-cast-qual -Wno-covered-switch-default)
endif()

## The Python bindings module for Quake dialect depends on CUDAQ libraries 
## which it can't locate since they are in "../../lib" and the 'rpath' is set
## to '$ORIGIN' by default.
## macOS uses @loader_path instead of $ORIGIN for RPATH.
if(APPLE)
  set(_origin_prefix "@loader_path")
else()
  set(_origin_prefix "$ORIGIN")
endif()

if (NOT SKBUILD)
  list(APPEND CMAKE_INSTALL_RPATH "${_origin_prefix}/../../lib" "${_origin_prefix}/../../lib/plugins")
  set_property(TARGET CUDAQuantumPythonModules.extension._quakeDialects.dso
    APPEND PROPERTY INSTALL_RPATH "${_origin_prefix}/../../../lib" "${_origin_prefix}/../../../lib/plugins")
else()
  list(APPEND CMAKE_INSTALL_RPATH "${_origin_prefix}/../cuda_quantum.libs")
  set_property(TARGET CUDAQuantumPythonModules.extension._quakeDialects.dso
    APPEND PROPERTY INSTALL_RPATH "${_origin_prefix}/../../../lib" "${_origin_prefix}/../../../cuda_quantum.libs")
endif()
