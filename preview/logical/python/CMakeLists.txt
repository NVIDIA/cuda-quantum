# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[=======================================================================[.rst:
CUDA-Q Logical Python bindings
-------------------

The CUDA-Q Logical Python package has two layers:

1. ``cudaq.logical.*`` -- an embedded copy of MLIR's standalone Python
   package (relocated under our package via
   ``MLIR_PYTHON_PACKAGE_PREFIX``), plus the CUDA-Q Logical dialect/compiler extensions
   assembled with CUDA-Q's shared-MLIR helpers. No symlinks to the SDK
   install are created.

2. ``cudaq.logical.*`` -- the user-facing pure-Python model, compiler, target,
   and analysis API. Copied to the build tree by a regular
   ``configure_file`` glob, since these are hand-written modules
   unrelated to MLIR's machinery.

Run ``ninja QLXPythonModules`` to assemble both halves.
#]=======================================================================]

include(AddMLIRPython)
include(MLIRDetectPythonEnv)

mlir_configure_python_dev_packages()

# Silence two classes of warnings that flood the MLIR Python-binding builds:
#   - [-Wattributes]: MLIR's NanobindUtils / NanobindAdaptors headers declare
#     several structs / lambdas with visibility greater than the type of one of
#     their fields (an upstream issue we can't fix). GCC emits hundreds of
#     these per TU.
#   - [-Wpragmas]: upstream MLIR adds Clang-specific -Wno-* flags (e.g.
#     -Wno-zero-length-array, -Wno-nested-anon-types) that GCC doesn't
#     recognize; each unrelated warning then trails a "unrecognized
#     command-line option" note from GCC under -Wpragmas.
# Scope: only this subdirectory, so the rest of the project keeps strict
# warning defaults.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-attributes -Wno-pragmas)
endif()

# --------------------------------------------------------------------------- #
# Package layout
# --------------------------------------------------------------------------- #

# Embed MLIR's standalone Python package directly under the cudaq.logical.*
# namespace -- matches upstream MLIR's mlir/python/mlir/ layout where
# `mlir.ir`, `mlir.dialects.arith`, `mlir.passmanager` etc. all sit at
# the top of the mlir package. For CUDA-Q Logical that's cudaq.logical.ir,
# cudaq.logical.dialects.qlx, cudaq.logical.passmanager, plus the canonical
# pure-Python modules as sibling submodules.
set(MLIR_PYTHON_PACKAGE_PREFIX     "cudaq.logical")
if(SKBUILD)
  # scikit-build-core installs relative to the wheel's platlib root, where
  # the canonical package is cudaq/logical/. Keep the historical standalone
  # CMake layout for non-wheel installs.
  set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "cudaq/logical")
else()
  set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/cudaq/logical")
endif()
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "cudaq_logical" CACHE STRING "" FORCE)

# Every MLIR-side Python C extension (theirs and ours) is compiled with
# the same MLIR_PYTHON_PACKAGE_PREFIX so all generated symbols agree.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

set(_PYTHON_BUILD_PREFIX        "${PROJECT_BINARY_DIR}/python/cudaq/logical")
set(QLX_PYTHON_BINDINGS_DIR     "${PROJECT_SOURCE_DIR}/ir/lib/Bindings/Python")

# Assemble the staged package from the current reviewed sources on every
# configure. Preserve the separately built CUDA-Q ingress plug-in: its staging
# copy is attached to the linker target and does not rerun merely because the
# copied file disappeared.
set(_QLX_STAGED_QUAKE_PLUGIN_BACKUP
  "${CMAKE_CURRENT_BINARY_DIR}/.qlx-quake-stage")
if(EXISTS "${_PYTHON_BUILD_PREFIX}/_quake")
  file(REMOVE_RECURSE "${_QLX_STAGED_QUAKE_PLUGIN_BACKUP}")
  file(RENAME "${_PYTHON_BUILD_PREFIX}/_quake"
    "${_QLX_STAGED_QUAKE_PLUGIN_BACKUP}")
endif()
file(REMOVE_RECURSE "${_PYTHON_BUILD_PREFIX}")
# The pre-move staging dir would otherwise stay on PYTHONPATH beside
# cudaq/logical/ and load a second copy of the same nanobind domain.
file(REMOVE_RECURSE "${PROJECT_BINARY_DIR}/python/cudaq_logical")
if(EXISTS "${_QLX_STAGED_QUAKE_PLUGIN_BACKUP}")
  file(MAKE_DIRECTORY "${_PYTHON_BUILD_PREFIX}")
  file(RENAME "${_QLX_STAGED_QUAKE_PLUGIN_BACKUP}"
    "${_PYTHON_BUILD_PREFIX}/_quake")
endif()

# mlir-tblgen needs to find our CUDA-Q Logical TD includes when expanding the
# python-binding dialect shims.
include_directories(${PROJECT_SOURCE_DIR}/ir/include)

# --------------------------------------------------------------------------- #
# Sources umbrella
# --------------------------------------------------------------------------- #

declare_mlir_python_sources(QLXPythonSources)

# --------------------------------------------------------------------------- #
# Dialects: TableGen-generated bindings + thin Python wrappers
# --------------------------------------------------------------------------- #

declare_mlir_python_sources(QLXPythonSources.Dialects
  ADD_TO_PARENT QLXPythonSources)

# MLIR's Python binding generator does not implement the C++ generator's
# operation filters. Generate into a private intermediate, then structurally
# retain only the native product registry before the underscore module is
# staged or installed.
function(declare_qlx_product_dialect_python_bindings)
  cmake_parse_arguments(ARG
    "GEN_ENUM_BINDINGS"
    "ROOT_DIR;ADD_TO_PARENT;TD_FILE;DIALECT_NAME;GEN_ENUM_BINDINGS_TD_FILE"
    "SOURCES;DEPENDS;OP_CLASSES"
    ${ARGN})

  set(_dialect_target "${ARG_ADD_TO_PARENT}.${ARG_DIALECT_NAME}")
  declare_mlir_python_sources(${_dialect_target}
    ROOT_DIR "${ARG_ROOT_DIR}"
    ADD_TO_PARENT "${ARG_ADD_TO_PARENT}"
    SOURCES "${ARG_SOURCES}")

  set(_tablegen_target "${_dialect_target}.tablegen")
  set(_td_file "${ARG_ROOT_DIR}/${ARG_TD_FILE}")
  get_filename_component(_relative_td_directory "${ARG_TD_FILE}" DIRECTORY)
  file(MAKE_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/${_relative_td_directory}")
  set(_full_filename
    "${_relative_td_directory}/_${ARG_DIALECT_NAME}_ops_full.py")
  set(_product_filename
    "${_relative_td_directory}/_${ARG_DIALECT_NAME}_ops_gen.py")
  set(LLVM_TARGET_DEFINITIONS "${_td_file}")
  mlir_tablegen("${_full_filename}"
    -gen-python-op-bindings -bind-dialect=${ARG_DIALECT_NAME}
    DEPENDS ${ARG_DEPENDS})
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_product_filename}"
    COMMAND "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/filter_generated_ops.py"
      "${CMAKE_CURRENT_BINARY_DIR}/${_full_filename}"
      "${CMAKE_CURRENT_BINARY_DIR}/${_product_filename}"
      ${ARG_OP_CLASSES}
    DEPENDS
      "${CMAKE_CURRENT_BINARY_DIR}/${_full_filename}"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/filter_generated_ops.py"
    COMMAND_EXPAND_LISTS
    VERBATIM
    COMMENT "Filtering ${ARG_DIALECT_NAME} product Python operations")
  list(APPEND TABLEGEN_OUTPUT
    "${CMAKE_CURRENT_BINARY_DIR}/${_product_filename}")
  add_public_tablegen_target(${_tablegen_target})

  set(_generated_sources "${_product_filename}")
  if(ARG_GEN_ENUM_BINDINGS OR ARG_GEN_ENUM_BINDINGS_TD_FILE)
    if(ARG_GEN_ENUM_BINDINGS_TD_FILE)
      set(_enum_td_file
        "${ARG_ROOT_DIR}/${ARG_GEN_ENUM_BINDINGS_TD_FILE}")
      set(LLVM_TARGET_DEFINITIONS "${_enum_td_file}")
    endif()
    set(_enum_filename
      "${_relative_td_directory}/_${ARG_DIALECT_NAME}_enum_gen.py")
    mlir_tablegen("${_enum_filename}" -gen-python-enum-bindings
      -bind-dialect=${ARG_DIALECT_NAME})
    list(APPEND _generated_sources "${_enum_filename}")
  endif()

  declare_mlir_python_sources("${_dialect_target}.ops_gen"
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    ADD_TO_PARENT "${_dialect_target}"
    SOURCES ${_generated_sources})
endfunction()

set(_QLX_PRODUCT_OP_CLASSES
  EstimateResultOp DeviceOp LogicalToQECBindingOp QECLoweringOp
  LoweringRecipeOp TargetManifestOp ExperimentOp ReturnOp ProgramOp
  ObjectiveBodyOp ActionOp InstrumentDeclOp ApplyOp InstrumentOp PrepareOp
  MeasureOp CallOp IdleOp DiscardOp SelectionOp RepeatOp YieldOp
  WhileConditionOp WhileOp IfOp XorOp)

declare_qlx_product_dialect_python_bindings(
  ADD_TO_PARENT QLXPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cudaq/logical"
  TD_FILE dialects/QLXOps.td
  SOURCES dialects/qlx.py
  DIALECT_NAME qlx
  OP_CLASSES ${_QLX_PRODUCT_OP_CLASSES}
  GEN_ENUM_BINDINGS
  GEN_ENUM_BINDINGS_TD_FILE dialects/QLXAttrs.td
  DEPENDS MLIRQLXIncGen)

set(_FABRIC_PRODUCT_OP_CLASSES
  DeviceOp RegionOp InterconnectOp CodeOp CodeProfileOp EncodingOp PatchGraphOp
  EncodingEpochSchemaOp EncodingEpochOp EpochTransitionOp EncodingHierarchyOp
  EncodingProjectionOp PatchTransformOp TransformBeginOp TransformEndOp
  ObjectiveOp GadgetSpecOp ReturnOp CircuitOp GadgetOp CallOp RelocateOp
  EstablishSupportOp RetryOp ProtocolReturnOp ProtocolOp EncodingUnpackOp
  MapChildrenOp EncodingPackOp AllocOp PrepZOp PrepXOp DeallocOp HOp SOp SdgOp
  XOp ZOp TOp TdgOp ResetOp InitBasisOp PermuteOp MzOp MeasureBasisOp MppOp
  ReadSyndromeAncillasOp AssembleSyndromeOp MeasureGaugesOp CXOp CZOp
  TransversalCXOp MergeOp SplitOp MultiMeasureOp MeasureProductOp RotateProductOp
  ResourceRotateProductOp ResourceRequestOp SelectionOp ProduceResourceOp
  UnpackResourceOp PackResourceOp InjectOp DiscardResourceOp TransportOp MoveOp
  SendOp RecvOp BarrierOp IdleOp YieldOp WhileConditionOp WhileOp IfOp RepeatOp
  XorOp AllZeroOp ParityOp AllFalseOp)

declare_qlx_product_dialect_python_bindings(
  ADD_TO_PARENT QLXPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cudaq/logical"
  TD_FILE dialects/FabricOps.td
  SOURCES dialects/fabric.py
  DIALECT_NAME fabric
  OP_CLASSES ${_FABRIC_PRODUCT_OP_CLASSES}
  GEN_ENUM_BINDINGS
  GEN_ENUM_BINDINGS_TD_FILE dialects/FabricAttrs.td
  DEPENDS MLIRFabricIncGen)

# --------------------------------------------------------------------------- #
# C++ extensions
# --------------------------------------------------------------------------- #

# _site_initialize_0: MLIR loads every _site_initialize_<i> module under
# _mlir_libs/ at Context construction; ours auto-registers every dialect
# a typical CUDA-Q Logical program touches (CUDA-Q Logical + LVM + Fabric and standard
# func/arith/cf/llvm) plus eagerly registers the retained
# CUDA-Q Logical-side passes at module import time.
declare_mlir_python_extension(QLXPythonSources.Init
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT QLXPythonSources
  ROOT_DIR "${QLX_PYTHON_BINDINGS_DIR}"
  SOURCES _site_initialize_0.cpp
  EMBED_CAPI_LINK_LIBS
    # CUDA-Q Logical-side CAPI.
    QLXCAPIDialectQLX
    QLXCAPIDialectLVM
    QLXCAPIDialectFabric
    QLXCAPIPasses
    # Standard MLIR dialect CAPI: needed so our site-init can register
    # func/arith/cf/llvm on every Context without having to pull in
    # MLIRPythonExtension.RegisterEverything (which would drag in every
    # upstream MLIR dialect). These are thin C wrapper objects; in CUDA-Q SDK
    # mode their C++ implementation still resolves from libcudaqMLIR.
    MLIRCAPIArith
    MLIRCAPIControlFlow
    MLIRCAPIFunc
    MLIRCAPILLVM
    MLIRCAPIIR
    # MLIRCAPITransforms provides Rewrite.cpp symbols (e.g.
    # mlirPDLResultListPushBackType) referenced by _mlir.so via
    # Rewrite.cpp. Without it the common CAPI lib doesn't export
    # those symbols and _mlir.so fails to load with undefined symbol.
    MLIRCAPITransforms
)

# Single dialect extension: _qlx_ext.  Built from _qlx_ext.cpp (main
# NB_MODULE) plus DialectQLX.cpp / DialectFabric.cpp which each
# contribute a populate* function into a submodule.  Mirrors the upstream
# MLIR MainModule.cpp + IR*.cpp layout.
declare_mlir_python_extension(QLXPythonSources.Ext
  MODULE_NAME _qlx_ext
  ADD_TO_PARENT QLXPythonSources
  ROOT_DIR "${QLX_PYTHON_BINDINGS_DIR}"
  SOURCES
    _qlx_ext.cpp
    DialectQLX.cpp
    DialectFabric.cpp
  EMBED_CAPI_LINK_LIBS
    QLXCAPIDialectQLX
    QLXCAPIDialectFabric
    MLIRCAPIIR
)

# Textual / module-level helpers (verify / lower / translate). Link deps that
# aren't part of MLIR's export set are wired below on the .dso target instead
# of via PRIVATE_LINK_LIBS (avoids "QLXPythonSources.Runtime requires target X
# that is not in any export set" errors).
declare_mlir_python_extension(QLXPythonSources.Runtime
  MODULE_NAME _qlxRuntime
  ADD_TO_PARENT QLXPythonSources
  ROOT_DIR "${QLX_PYTHON_BINDINGS_DIR}"
  SOURCES QLXRuntime.cpp
  EMBED_CAPI_LINK_LIBS
    QLXCAPIDialectQLX
    QLXCAPIDialectFabric
    QLXCAPITranslate
    QLXCAPIPasses
    MLIRCAPIIR
)

# --------------------------------------------------------------------------- #
# Common CAPI shared library + module assembly
# --------------------------------------------------------------------------- #

set(_SOURCE_COMPONENTS
  QLXPythonSources
  MLIRPythonSources.Core
  # Python-side dialect modules (for the pythonic MLIR API surface like
  # `cudaq.logical.dialects.func.FuncOp`).  Dialect *registration* on
  # new Contexts is handled by _site_initialize_0.cpp via the mlir-c
  # dialect handles, not by pulling in MLIRPythonExtension.RegisterEverything
  # (which would drag in every upstream dialect).
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.llvm
  # MLIR's Core Python sources install dialects/ext.py, whose public helper
  # implementation imports dialects/irdl.py.  Keep that upstream dependency
  # closure intact so every installed Python module is importable.
  MLIRPythonSources.Dialects.irdl
)

add_cudaq_python_common_capi_library(QLXPythonCAPI
  INSTALL_COMPONENT QLXPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${_PYTHON_BUILD_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${_SOURCE_COMPONENTS}
)

# CUDA-Q's helper deliberately exports only CAPI symbols, but its current
# export list is hard-coded to the mlir* and cudaq* prefixes. CUDA-Q Logical's embedded
# CAPI objects use qlx*, so extend the generated list until the upstream
# helper accepts downstream export prefixes.
get_target_property(_qlx_capi_link_options QLXPythonCAPI LINK_OPTIONS)
if(NOT _qlx_capi_link_options
    OR _qlx_capi_link_options STREQUAL "_qlx_capi_link_options-NOTFOUND")
  set(_qlx_capi_link_options)
endif()
if(APPLE)
  list(FILTER _qlx_capi_link_options EXCLUDE REGEX
    "exported_symbols_list")
  set(_qlx_capi_exports
    "${CMAKE_CURRENT_BINARY_DIR}/QLXPythonCAPI-qlx-exported.txt")
  file(WRITE "${_qlx_capi_exports}"
    "_mlir*\n_cudaq*\n_qlx*\n_lvm*\n_fabric*\n")
  set(_qlx_capi_export_option
    "LINKER:-exported_symbols_list,${_qlx_capi_exports}")
else()
  list(FILTER _qlx_capi_link_options EXCLUDE REGEX "version-script")
  set(_qlx_capi_exports
    "${CMAKE_CURRENT_BINARY_DIR}/QLXPythonCAPI-qlx.map")
  file(WRITE "${_qlx_capi_exports}"
    "{\n  global:\n    mlir*;\n    cudaq*;\n    qlx*;\n    lvm*;\n"
    "    fabric*;\n  local:\n    *;\n};\n")
  set(_qlx_capi_export_option
    "LINKER:--version-script=${_qlx_capi_exports}")
endif()
set_property(TARGET QLXPythonCAPI PROPERTY
  LINK_OPTIONS "${_qlx_capi_link_options}")
set_property(TARGET QLXPythonCAPI APPEND PROPERTY
  LINK_DEPENDS "${_qlx_capi_exports}")
target_link_options(QLXPythonCAPI PRIVATE "${_qlx_capi_export_option}")

add_cudaq_python_modules(QLXPythonModules
  ROOT_PREFIX "${_PYTHON_BUILD_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_SOURCE_COMPONENTS}
  COMMON_CAPI_LINK_LIBS QLXPythonCAPI
)

# --------------------------------------------------------------------------- #
# Post-assembly: link in non-exported deps and wire compile defs.
#
# add_mlir_python_extension creates targets named
# <Modules>.extension.<Module>.dso.  Anything we attach here goes on the
# real build target, not the upstream INTERFACE target -- so it doesn't
# cascade through MLIR's export set.
# --------------------------------------------------------------------------- #

set(_qlx_runtime_target "QLXPythonModules.extension._qlxRuntime.dso")
if(TARGET ${_qlx_runtime_target})
  target_link_libraries(${_qlx_runtime_target} PRIVATE
    MLIRQLXTransforms
    MLIRFabricTransforms
    MLIRPluginsLib
  )
endif()

# The Python Quake conversion APIs load qlx-quake-plugin at runtime. Make the
# Python package depend on it (so it is always built) and install it into the
# package's _quake/ directory. Without both, an installed tree can report
# has_quake_import=true yet fail conversion with "plugin not found". (The
# plugin's POST_BUILD stages a copy into the build-tree package for dev/CI use.)
if(TARGET qlx-quake-plugin)
  add_dependencies(QLXPythonModules qlx-quake-plugin)
  install(TARGETS qlx-quake-plugin
    LIBRARY
      DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_quake"
      COMPONENT QLXPythonModules
    RUNTIME
      DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_quake"
      COMPONENT QLXPythonModules)
endif()

# QLXPythonCAPI and qlx-quake-plugin are not covered by
# add_cudaq_python_modules' SKBUILD RPATH helper. They live at
# cudaq/logical/_mlir_libs and cudaq/logical/_quake -- three levels below
# platlib, the same depth as upstream cudaq/mlir/_mlir_libs -- so they need
# $ORIGIN/../../../lib to reach the cudaq wheel's libcudaqMLIR.
# Extension DSOs go through add_cudaq_python_modules, which already appends
# that path; do not override their RPATH here (that would drop MLIR's $ORIGIN).
if(SKBUILD)
  if(APPLE)
    set(_qlx_cudaq_runtime_rpaths
      "@loader_path"
      "@loader_path/../../../lib"
      "@loader_path/../../../cuda_quantum.libs")
  else()
    set(_qlx_cudaq_runtime_rpaths
      "$ORIGIN"
      "$ORIGIN/../../../lib"
      "$ORIGIN/../../../cuda_quantum.libs")
  endif()

  set(_qlx_cudaq_runtime_targets QLXPythonCAPI)
  if(TARGET qlx-quake-plugin)
    list(APPEND _qlx_cudaq_runtime_targets qlx-quake-plugin)
  endif()
  foreach(_qlx_cudaq_runtime_target IN LISTS _qlx_cudaq_runtime_targets)
    set_property(TARGET ${_qlx_cudaq_runtime_target} PROPERTY
      INSTALL_RPATH "${_qlx_cudaq_runtime_rpaths}")
  endforeach()
endif()

# --------------------------------------------------------------------------- #
# Canonical Python user surface.
#
# These hand-written modules provide the programming model, immutable model
# values, compiler, analyses, and target recipes over the generated MLIR
# bindings staged above.
# --------------------------------------------------------------------------- #

file(GLOB_RECURSE _py_sources
  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cudaq/logical/*.py")
foreach(_src ${_py_sources})
  file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${_src})
  configure_file(${_src} ${CMAKE_CURRENT_BINARY_DIR}/${_rel} COPYONLY)
endforeach()

# Build-tree only: extend cudaq.__path__ so `import cudaq.logical` resolves
# from this prefix without a cudaq/__init__.py in our tree. Never installed.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_cudaq_logical_devpath.py"
"import cudaq, pathlib
_p = str(pathlib.Path(__file__).resolve().parent / \"cudaq\")
if _p not in cudaq.__path__:
    cudaq.__path__.append(_p)
")

# --------------------------------------------------------------------------- #
# Tests (CTest registrations)
# --------------------------------------------------------------------------- #

set(_qlx_runtime_pytest_env "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
# The Python suite is a first-class CTest conformance test rather than an
# unrecorded direct pytest invocation; pytest reports the live collected count,
# so documentation never needs to pin a number that immediately goes stale.
add_test(NAME qlx-python-tests
  COMMAND ${Python3_EXECUTABLE} -m pytest --rootdir=${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/cudaq/logical/
    -v
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
set_tests_properties(qlx-python-tests PROPERTIES
  ENVIRONMENT "${_qlx_runtime_pytest_env}"
  DEPENDS QLXPythonModules
  LABELS "qlx;conformance"
  TIMEOUT 1200
)
