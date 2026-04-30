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
    ../runtime/utils/PyRemoteSimulatorQPU.cpp
    ../runtime/utils/PyRestRemoteClient.cpp
    ../utils/LinkedLibraryHolder.cpp
    ../../runtime/common/CodeGenConfig.cpp
    ../../runtime/cudaq/platform/default/rest_server/RemoteRuntimeClient.cpp
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
if (CUDAQ_ENABLE_REST)
  target_link_libraries(CUDAQuantumPythonSources.Extension INTERFACE unzip_util)
endif()

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
    # Include full MLIRPythonSources so dialect extensions' EMBED_CAPI_LINK_LIBS
    # (e.g. obj.MLIRCAPILLVM for the LLVM dialect) are embedded into the common
    # CAPI lib. Otherwise _mlirDialectsLLVM.so fails with undefined symbol
    # mlirTypeIsALLVMStructType at runtime.
    MLIRPythonSources
)

if(APPLE)
  target_link_options(CUDAQuantumPythonCAPI PRIVATE
    "LINKER:-flat_namespace"
    "LINKER:-undefined,dynamic_lookup")
endif()

################################################################################
# Instantiation of Python module
################################################################################

# This variable is unused in cudaq but if it is not set, we hit a bug in  
# add_mlir_python_modules whereby it is defined twice on the compilation line:
# -DMLIR_BINDINGS_PYTHON_NB_DOMAIN "" -DMLIR_BINDINGS_PYTHON_NB_DOMAIN mlir
# which results in a compilation error. 
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "cudaq")

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

# Upstream MLIR's add_mlir_python_extension sets `-Wl,--exclude-libs,ALL` on
# every extension, which hides the symbols pulled in from the static MLIR
# archives from the extension's dynamic symbol table. For upstream extensions
# that only use CAPI functions this is fine, but CUDA-Q's _quakeDialects.so
# calls MLIR C++ APIs directly (e.g. StringAttr::get in py_register_dialects
# and CUDAQuantumExtension). Those calls reference template statics like
# `mlir::detail::TypeIDResolver<T>::id` which are GNU UNIQUE symbols with
# default visibility. When `--exclude-libs,ALL` hides them, each DSO ends up
# with its own private copy, and the TypeID used inside _quakeDialects.so no
# longer matches CAPI's — tripping "storage uniquer isn't initialized" and
# "different dialects for the same namespace" errors at runtime.
#
# Strip that option so the UNIQUE statics stay in the dynamic symbol table
# and the runtime linker unifies them with libCUDAQuantumPythonCAPI.so's
# copy at load time.
if(TARGET CUDAQuantumPythonModules.extension._quakeDialects.dso)
  # 1) Strip --exclude-libs,ALL so the MLIR template statics (e.g.
  #    mlir::detail::TypeIDResolver<T>::id) that come in via the MLIR static
  #    archives stay in the dynamic symbol table. These are STB_GLOBAL
  #    (default visibility) COMDAT symbols; with --exclude-libs,ALL they are
  #    demoted to local and each DSO ends up with its own private copy.
  get_target_property(_qd_link_options
    CUDAQuantumPythonModules.extension._quakeDialects.dso LINK_OPTIONS)
  if(_qd_link_options)
    list(REMOVE_ITEM _qd_link_options
      "$<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>"
      "LINKER:--exclude-libs,ALL"
      "LINKER:-twolevel_namespace")
    set_target_properties(CUDAQuantumPythonModules.extension._quakeDialects.dso
      PROPERTIES LINK_OPTIONS "${_qd_link_options}")
  endif()

  # 2) Prepend libCUDAQuantumPythonCAPI.so to the link line so ld's archive
  #    extraction finds MLIR symbols in the (shared) CAPI before scanning the
  #    static archives. When CAPI already defines `mlir::StringAttr::get`,
  #    `mlir::detail::TypeIDResolver<T>::id`, etc., the matching .o files in
  #    libMLIRIR.a are not pulled in, so _quakeDialects.so has no private
  #    copies and its references resolve to CAPI at runtime — keeping the
  #    TypeID addresses consistent with the ones CAPI used when constructing
  #    the MLIRContext.
  target_link_options(CUDAQuantumPythonModules.extension._quakeDialects.dso
    BEFORE PRIVATE
      "$<TARGET_FILE:CUDAQuantumPythonCAPI>")

  if(APPLE)
    target_link_options(CUDAQuantumPythonModules.extension._quakeDialects.dso
      PRIVATE
        "LINKER:-flat_namespace"
        "LINKER:-undefined,dynamic_lookup")
  endif()
endif()

if(TARGET cudaq-mlir-runtime AND TARGET CUDAQuantumPythonCAPI)
  get_target_property(_mr_link_options cudaq-mlir-runtime LINK_OPTIONS)
  if(_mr_link_options)
    list(REMOVE_ITEM _mr_link_options
      "$<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>"
      "LINKER:--exclude-libs,ALL"
      "LINKER:-twolevel_namespace")
    set_target_properties(cudaq-mlir-runtime
      PROPERTIES LINK_OPTIONS "${_mr_link_options}")
  endif()
  target_link_options(cudaq-mlir-runtime BEFORE PRIVATE
    "$<TARGET_FILE:CUDAQuantumPythonCAPI>")
  target_link_libraries(cudaq-mlir-runtime INTERFACE
    $<BUILD_INTERFACE:CUDAQuantumPythonCAPI>)
  if(APPLE)
    target_link_options(cudaq-mlir-runtime PRIVATE
      "LINKER:-undefined,dynamic_lookup")
  endif()
  set_property(TARGET cudaq-mlir-runtime APPEND PROPERTY
    BUILD_RPATH "$<TARGET_FILE_DIR:CUDAQuantumPythonCAPI>")
  add_dependencies(cudaq-mlir-runtime CUDAQuantumPythonCAPI)
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

## Retain all linked libraries (e.g. libcudaq) so that static initializers
## (ModuleLauncher registry and PythonLauncher registration) run and resolve
## in the same process. Without --no-as-needed the linker may drop libcudaq
## and the launcher is never registered.
if(CUDAQ_FORCE_LINK_FLAG)
  target_link_options(CUDAQuantumPythonCAPI PRIVATE
    ${CUDAQ_FORCE_LINK_FLAG})
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
