# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)
include(AddMLIRPython)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wdeprecated-literal-operator"
  CUDAQ_HAS_WDEPRECATED_LITERAL_OPERATOR)
if(NOT CUDAQ_HAS_WDEPRECATED_LITERAL_OPERATOR)
  add_compile_options(-Wno-unknown-warning-option)
endif()

# Specifies that all MLIR packages are co-located under the cudaq
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=cudaq.mlir.")
# Mark QPU sources compiled into the Python extension so they use the
# cross-DSO registry hook (cudaq_add_qpu_node) instead of the local
# CUDAQ_REGISTER_TYPE which would register into the wrong DSO.
add_compile_definitions("CUDAQ_PYTHON_EXTENSION")

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

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cudaq/mlir"
  TD_FILE dialects/QECOps.td
  SOURCES
    dialects/qec.py
  DIALECT_NAME qec)

# Define the register_dialects hook to populate the default DialectRegistry
# for every Context.
declare_mlir_python_extension(CUDAQuantumPythonSources.SiteInitialize
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT CUDAQuantumPythonSources
  SOURCES
    SiteInitialize.cpp
  PRIVATE_LINK_LIBS
    cudaqMLIR
)

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
    ../runtime/cudaq/algorithms/py_observe.cpp
    ../runtime/cudaq/algorithms/py_observe_async.cpp
    ../runtime/cudaq/algorithms/py_optimizer.cpp
    ../runtime/cudaq/algorithms/py_sample.cpp
    ../runtime/cudaq/algorithms/py_sample_async.cpp
    ../runtime/cudaq/algorithms/py_sample_ptsbe.cpp
    ../runtime/cudaq/algorithms/py_resource_count.cpp
    ../runtime/cudaq/analysis/py_dem.cpp
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
    ../runtime/cudaq/trace/py_trace.cpp
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
    PythonMLIRHooks.cpp

  PRIVATE_LINK_LIBS
    cudaqMLIR
    cudaq-mlir-runtime
    CUDAQTargetConfig
    cudaq-python-interop
    cudaq-platform-default

  EMBED_CAPI_LINK_LIBS
    cudaqDialectsCAPI
    MLIRCAPITransforms
)

# MLIR/LLVM is built without RTTI (LLVM_ENABLE_RTTI=OFF). This file subclasses
# mlir::PassInstrumentation, so it must match LLVM's RTTI setting.
set_source_files_properties(
    ../runtime/cudaq/platform/PythonSignalCheck.cpp
    PROPERTIES COMPILE_FLAGS "-fno-rtti -fno-exceptions"
)


target_include_directories(CUDAQuantumPythonSources.Extension INTERFACE
    ${CMAKE_SOURCE_DIR}/python
    ${CMAKE_SOURCE_DIR}/python/utils
    ${CMAKE_SOURCE_DIR}/runtime
    ${CMAKE_SOURCE_DIR}/runtime/internal/compiler/include
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
# Instantiation of Python module
################################################################################

# This variable is unused in cudaq but if it is not set, we hit a bug in  
# add_mlir_python_modules whereby it is defined twice on the compilation line:
# -DMLIR_BINDINGS_PYTHON_NB_DOMAIN "" -DMLIR_BINDINGS_PYTHON_NB_DOMAIN mlir
# which results in a compilation error. 
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "cudaq")

set(CUDAQ_PYTHON_DECLARED_SOURCES
  CUDAQuantumPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.ExecutionEngine
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.complex
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.Dialects.math)

# Build the common CAPI shared library.
add_cudaq_python_common_capi_library(cudaqMLIRCAPI
  DECLARED_SOURCES ${CUDAQ_PYTHON_DECLARED_SOURCES})

install(TARGETS cudaqMLIRCAPI
  EXPORT cudaq-targets
  COMPONENT CUDAQuantumPythonModules
  DESTINATION lib)

add_cudaq_python_modules(CUDAQuantumPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python/cudaq/mlir"
  INSTALL_PREFIX "cudaq/mlir"
  DECLARED_SOURCES ${CUDAQ_PYTHON_DECLARED_SOURCES}
  COMMON_CAPI_LINK_LIBS
    cudaqMLIRCAPI
)

# Suppress warnings-as-errors for upstream MLIR Python extension sources
# that have minor GCC warnings (address-of-function, parentheses) in LLVM 22.
foreach(_cudaq_py_ext_target
  CUDAQuantumPythonModules.extension._mlir.dso
  CUDAQuantumPythonModules.extension.MLIRPythonSupport-cudaq.so)
  if(TARGET ${_cudaq_py_ext_target})
    target_compile_options(${_cudaq_py_ext_target} PRIVATE
      -Wno-error=address -Wno-error=parentheses)
  endif()
endforeach()
