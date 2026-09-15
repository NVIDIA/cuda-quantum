# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# python/'s _quakeDialects extension links against cudaq::cudaqMLIRCAPI,
# built by cudaq/python/CMakeLists.txt when CUDAQ_ENABLE_PYTHON_BINDINGS is
# ON. In the normal build that target already exists (cudaq/ is processed
# before python/ in the same configure). If it doesn't -- e.g. python/ is
# being configured independently of cudaq/ (CUDAQ_ENABLE_CORE=OFF) -- look
# for an installed CUDAQ and fail with a clear diagnostic if it doesn't
# provide the bindings either, rather than a cryptic "target not found"
# further down.
if (NOT TARGET cudaq::cudaqMLIRCAPI)
  find_package(CUDAQ REQUIRED)
  if (NOT CUDAQ_ENABLE_PYTHON_BINDINGS)
    message(FATAL_ERROR
      "python/ requires cudaq/'s own MLIR Python bindings (Quake, CC, QEC "
      "dialects, built under cudaq/python/ when CUDAQ_ENABLE_PYTHON_BINDINGS "
      "is ON), but they were not found. Either cudaq/ is not being built as "
      "part of this configure with them enabled (CUDAQ_ENABLE_CORE=OFF, or "
      "CUDAQ_ENABLE_PYTHON_BINDINGS was OFF when cudaq/ was configured), or "
      "the installed CUDAQ found here (${CUDAQ_DIR}) was built without them. "
      "Rebuild cudaq/ -- in this configure, or as a separate install -- with "
      "-DCUDAQ_ENABLE_PYTHON_BINDINGS=ON.")
  endif()
endif()

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

# The Quake/CC/QEC dialect bindings and the _site_initialize_0 extension are
# declared in cudaq/python/CMakeLists.txt (part of the cudaq/ subproject, not
# python/). This module -- the full Python runtime surface (algorithms,
# operators, platform, qis, ...) -- is genuinely python/-runtime-coupled, so
# it stays here. It links against the already-built cudaqMLIRCAPI aggregate
# (built by cudaq/python/) rather than embedding a second copy of the same
# CAPI libraries: see cudaq/python/CMakeLists.txt for why.

declare_mlir_python_sources(CUDAQuantumPythonSources)

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
    ../runtime/cudaq/algorithms/py_state.cpp
    ../runtime/cudaq/algorithms/py_translate.cpp
    ../runtime/cudaq/algorithms/py_unitary.cpp
    ../runtime/cudaq/algorithms/py_utils.cpp
    ../runtime/cudaq/platform/CompiledModuleCache.cpp
    ../runtime/cudaq/platform/ProgramFingerprint.cpp
    ../runtime/cudaq/platform/PyRuntimeEndpoint.cpp
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
    ../runtime/cudaq/target/py_compile_target.cpp
    ../runtime/cudaq/target/py_runtime_target.cpp
    ../runtime/cudaq/target/py_testing_utils.cpp
    ../runtime/cudaq/trace/py_trace.cpp
    ../runtime/mlir/py_register_dialects.cpp
    ../utils/LinkedLibraryHolder.cpp
    ../../runtime/cudaq/platform/default/python/QPU.cpp
    PythonMLIRHooks.cpp

  PRIVATE_LINK_LIBS
    cudaqMLIR
    cudaq-mlir-runtime
    CUDAQTargetConfig
    cudaq-python-interop
    cudaq-platform-default
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
# The `cudaq.orca` submodule binds cudaq::orca::sample directly, so it can only
# be built when the ORCA backend is. Link the library rather than recompiling
# its sources here: a private copy would shadow libcudaq-orca-qpu in-process
# and would ignore the build configuration entirely.
if (CUDAQ_ENABLE_ORCA_BACKEND)
  target_link_libraries(CUDAQuantumPythonSources.Extension INTERFACE cudaq-orca-qpu)
  target_compile_definitions(CUDAQuantumPythonSources.Extension INTERFACE CUDAQ_ORCA_BACKEND_ENABLED)
endif()

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

# Only this module's own sources: the Core MLIR / Quake / CC / QEC sources
# are already assembled (and, for the standalone-python-against-installed-
# cudaq case, already installed) by cudaq/python/CMakeLists.txt. This module
# adds its files into the same "cudaq/mlir" install prefix alongside that
# (already-built-or-installed) tree, the same pattern upstream MLIR uses for
# a downstream project's own dialect bindings layered onto MLIR's own
# self-contained python package.
add_cudaq_python_modules(CUDAQuantumPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python/cudaq/mlir"
  INSTALL_PREFIX "cudaq/mlir"
  DECLARED_SOURCES CUDAQuantumPythonSources
  COMMON_CAPI_LINK_LIBS
    cudaq::cudaqMLIRCAPI
)
