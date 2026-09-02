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

1. ``cudaq.mlir.*`` -- dialect wrappers (``qlx`` / ``fabric`` / ``lvm``) and
   C++ extensions (``_qlx_ext``, ``_qlxRuntime``, ``_site_initialize_1``,
   ``libQLXPythonCAPI``) registered into CUDA-Q's existing MLIR Python
   bindings. Core ``ir`` / ``passmanager`` / ``_mlir.so`` come from the
   CUDA-Q runtime wheel; this package does not rebuild them.

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

# Register into CUDA-Q's existing cudaq.mlir Python bindings (same nanobind
# domain, same _mlir.so). Dialect wrappers and our extension DSOs land next
# to CUDA-Q's ir.py / passmanager.py / _mlir_libs/.
set(MLIR_PYTHON_PACKAGE_PREFIX     "cudaq.mlir")
set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "cudaq/mlir")
set(QLX_PACKAGE_INSTALL_PREFIX "cudaq/logical")
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "cudaq" CACHE STRING "" FORCE)

# Every MLIR-side Python C extension is compiled with the same prefix so
# generated capsule names agree with CUDA-Q's _mlir.so.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

set(_MLIR_PYTHON_BUILD_PREFIX   "${PROJECT_BINARY_DIR}/python/cudaq/mlir")
set(QLX_PYTHON_BINDINGS_DIR     "${PROJECT_SOURCE_DIR}/ir/lib/Bindings/Python")

# Wipe only the MLIR staging root. cudaq/logical/_quake lives outside it, so
# the plugin copy does not need to be backed up across configure.
file(REMOVE_RECURSE "${_MLIR_PYTHON_BUILD_PREFIX}")
# Stale private-MLIR copies from before this layout would otherwise stay on
# PYTHONPATH and load a second copy of the same nanobind domain.
file(REMOVE_RECURSE "${PROJECT_BINARY_DIR}/python/cudaq_logical")
file(REMOVE_RECURSE "${PROJECT_BINARY_DIR}/python/cudaq/logical/_mlir_libs")
file(REMOVE
  "${PROJECT_BINARY_DIR}/python/cudaq/logical/ir.py"
  "${PROJECT_BINARY_DIR}/python/cudaq/logical/passmanager.py"
  "${PROJECT_BINARY_DIR}/python/cudaq/logical/execution_engine.py"
  "${PROJECT_BINARY_DIR}/python/cudaq/logical/rewrite.py")
file(REMOVE_RECURSE
  "${PROJECT_BINARY_DIR}/python/cudaq/logical/extras"
  "${PROJECT_BINARY_DIR}/python/cudaq/logical/dialects")

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
  ADD_TO_PARENT QLXPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  SOURCES dialects/lvm.py)

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
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
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
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
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

# _site_initialize_1: CUDA-Q owns _site_initialize_0. MLIR's loader counts
# up from 0 and stops at the first gap, so a third distribution taking
# index 2 would only load while cudaq-logical is installed.
# Dialect handles and the QLX pass set are registered here at import time.
declare_mlir_python_extension(QLXPythonSources.Init
  MODULE_NAME _site_initialize_1
  ADD_TO_PARENT QLXPythonSources
  ROOT_DIR "${QLX_PYTHON_BINDINGS_DIR}"
  SOURCES _site_initialize_1.cpp
  EMBED_CAPI_LINK_LIBS
    QLXCAPIDialectQLX
    QLXCAPIDialectLVM
    QLXCAPIDialectFabric
    QLXCAPIPasses
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
)

# --------------------------------------------------------------------------- #
# Common CAPI shared library + module assembly
# --------------------------------------------------------------------------- #

set(_SOURCE_COMPONENTS QLXPythonSources)

# add_cudaq_python_common_capi_library checks undefined mlir::/llvm:: symbols
# against libcudaqMLIR only. After dropping the embedded MLIRCAPI* objects,
# CAPI and nanobind-domain symbols resolve from the runtime wheel, so disable
# the helper's check and re-run it with those providers.
set(_qlx_saved_symbol_check "${CUDAQ_CHECK_MLIR_SYMBOL_CLOSURE}")
set(CUDAQ_CHECK_MLIR_SYMBOL_CLOSURE OFF)
add_cudaq_python_common_capi_library(QLXPythonCAPI
  INSTALL_COMPONENT QLXPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${_MLIR_PYTHON_BUILD_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${_SOURCE_COMPONENTS}
)
set(CUDAQ_CHECK_MLIR_SYMBOL_CLOSURE "${_qlx_saved_symbol_check}")

# QLXCAPI* libraries declare a PUBLIC dep on MLIRCAPIIR. Exclude the
# upstream CAPI aggregates so their objects are not copied into this DSO;
# the implementations live in libcudaqMLIRCAPI.
set_property(TARGET QLXPythonCAPI APPEND PROPERTY
  MLIR_AGGREGATE_EXCLUDE_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPIControlFlow
    MLIRCAPIFunc
    MLIRCAPILLVM
    MLIRCAPITransforms)

target_link_libraries(QLXPythonCAPI PRIVATE
  cudaq::cudaqMLIRCAPI
  cudaq::MLIRPythonSupport)
cudaq_check_mlir_symbol_closure(QLXPythonCAPI
  PROVIDERS cudaq::cudaqMLIRCAPI cudaq::MLIRPythonSupport)

add_cudaq_python_modules(QLXPythonModules
  ROOT_PREFIX "${_MLIR_PYTHON_BUILD_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_SOURCE_COMPONENTS}
  COMMON_CAPI_LINK_LIBS
    QLXPythonCAPI
    cudaq::cudaqMLIRCAPI
    cudaq::MLIRPythonSupport
)

# MLIRPythonSupport and nanobind live in the runtime wheel's
# cudaq/mlir/_mlir_libs, not in lib/. The build-tree output directory is
# also named _mlir_libs, so CMake warns about an RPATH cycle; the installed
# wheel path must still be on BUILD_RPATH for the DSOs to load from the
# build tree.
set(_qlx_cudaq_py_libs "${QLX_CUDAQ_PREFIX}/cudaq/mlir/_mlir_libs")
set_property(TARGET QLXPythonCAPI APPEND PROPERTY
  BUILD_RPATH "${_qlx_cudaq_py_libs}")
get_property(_qlx_py_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(FILTER _qlx_py_targets INCLUDE REGEX "^QLXPythonModules\\.extension\\.")
foreach(_qlx_py_dso IN LISTS _qlx_py_targets)
  set_property(TARGET ${_qlx_py_dso} APPEND PROPERTY
    BUILD_RPATH "${_qlx_cudaq_py_libs}")
endforeach()

# MLIR rebuilds libnanobind-cudaq.so for NB_DOMAIN=cudaq and would otherwise
# stage/install it next to CUDA-Q's copy. Keep one image: the runtime wheel's.
if(TARGET nanobind-cudaq)
  set_target_properties(nanobind-cudaq PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  # Undo the install() issued by build_nanobind_lib into cudaq/mlir/_mlir_libs.
  install(CODE "
    foreach(_nb_name IN ITEMS
        libnanobind-cudaq.so libnanobind-cudaq.dylib nanobind-cudaq.dll)
      set(_nb_path \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs/\${_nb_name}\")
      if(EXISTS \"\${_nb_path}\")
        file(REMOVE \"\${_nb_path}\")
      endif()
    endforeach()
  " COMPONENT QLXPythonModules)
endif()

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
      DESTINATION "${QLX_PACKAGE_INSTALL_PREFIX}/_quake"
      COMPONENT QLXPythonModules
    RUNTIME
      DESTINATION "${QLX_PACKAGE_INSTALL_PREFIX}/_quake"
      COMPONENT QLXPythonModules)
endif()

# QLXPythonCAPI and qlx-quake-plugin are not covered by
# add_cudaq_python_modules' SKBUILD RPATH helper. They live at
# cudaq/mlir/_mlir_libs and cudaq/logical/_quake -- three levels below
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
# Dialect wrappers live under python/mlir/ and are staged by MLIR's assembly
# into cudaq/mlir/dialects/. Keep them out of this glob so two producers
# cannot race for the same staged files.
list(FILTER _py_sources EXCLUDE REGEX "/dialects/")
foreach(_src ${_py_sources})
  file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${_src})
  configure_file(${_src} ${CMAKE_CURRENT_BINARY_DIR}/${_rel} COPYONLY)
endforeach()

# Hand-written cudaq.logical modules (compiler/, architecture/, ...) are staged
# above for the build tree. MLIR machinery installs bindings separately; mirror
# core CUDA-Q's ``install(DIRECTORY cudaq ...)`` for source-prefix installs.
if(NOT SKBUILD)
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cudaq/logical"
    DESTINATION cudaq
    COMPONENT QLXPythonModules
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "dialects/*.td" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE)
endif()

# Build-tree only: extend cudaq.__path__, cudaq.mlir.__path__ and
# cudaq.mlir._mlir_libs.__path__ so `import cudaq.logical` and our MLIR
# extensions resolve from this prefix. Never installed.
# cudaq/__init__.py imports _quakeDialects, which runs _site_initialize()
# before this shim can patch cudaq.mlir, so `_site_initialize_1` is missed
# on that first pass. After appending our `_mlir_libs` path, load it and
# call `register_dialects` on the existing global registry — do not re-run
# `_site_initialize()`, which would replace `ir.Context`.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_cudaq_logical_devpath.py"
"import importlib, pathlib
import cudaq
_build_cudaq = pathlib.Path(__file__).resolve().parent / \"cudaq\"
_p = str(_build_cudaq)
if _p not in list(cudaq.__path__):
    cudaq.__path__.append(_p)
import cudaq.mlir as _mlir
_mlir_p = str(_build_cudaq / \"mlir\")
if _mlir_p not in list(_mlir.__path__):
    _mlir.__path__.append(_mlir_p)
import cudaq.mlir._mlir_libs as _libs
_libs_p = str(_build_cudaq / \"mlir\" / \"_mlir_libs\")
if _libs_p not in list(_libs.__path__):
    _libs.__path__.append(_libs_p)
    # Do not re-run _site_initialize(): that replaces ir.Context and
    # double-wraps the CUDA-Q subclass. Register our dialects into the
    # already-built global registry instead.
    _init1 = importlib.import_module(
        \"cudaq.mlir._mlir_libs._site_initialize_1\")
    _init1.register_dialects(_libs.get_dialect_registry())
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
