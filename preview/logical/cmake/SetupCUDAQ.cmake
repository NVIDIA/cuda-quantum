# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[=======================================================================[.rst:
SetupCUDAQ
----------

Locate the CUDA-Q development installation that supplies CUDA-Q Logical's LLVM, MLIR and
CUDA-Q CMake packages, and its shared MLIR runtime ``cudaq::MLIR``.

CUDA-Q Logical never discovers or builds LLVM/MLIR on its own. Every MLIR symbol has to
resolve from the single ``libcudaqMLIR`` image CUDA-Q ships, otherwise the
dialect registry, pass registry, and TypeIDs are duplicated between CUDA-Q Logical's
Python extensions and CUDA-Q's. Taking LLVM/MLIR from the same prefix as
CUDA-Q is what keeps that invariant true by construction.

CUDA-Q Discovery
^^^^^^^^^^^^^^^^

By default, CUDA-Q is taken from the ``cudaq-devel`` wheel installed in the
Python environment resolved by ``QLXPythonEnv``.

Alternatively, CMake will look for an installation of CUDA-Q at
``CUDAQ_INSTALL_PREFIX`` if that variable is. Use ``-DCUDAQ_BUNDLE_MLIR_INSTALL=ON``
when building CUDA-Q to colocate the LLVM/MLIR installation in the same place.
Otherwise, specify the installation prefix of LLVM/MLIR using the usual
``LLVM_DIR`` and/or ``MLIR_DIR`` variables.

After inclusion the rest of the build has:

- ``LLVM_INCLUDE_DIRS`` / ``MLIR_INCLUDE_DIRS`` on the include path
- the ``TableGen``, ``AddLLVM``, ``AddMLIR`` and ``AddCUDAQ`` helpers
- ``cudaq::MLIR`` -- the shared MLIR/LLVM image every CUDA-Q Logical target links
- ``CUDAQ_INCLUDE_DIR`` / ``CUDAQ_LIBRARY_DIR``
- the LLVM test utilities as ``LIT_PATH``, ``FILECHECK_PATH``,
  ``MLIR_TRANSLATE_PATH``, ``QLX_NOT_PATH``, ``QLX_COUNT_PATH``,
  ``CLANGXX_PATH``

#]=======================================================================]

include_guard(GLOBAL)

# --------------------------------------------------------------------------- #
# Resolve the CUDA-Q prefix
# --------------------------------------------------------------------------- #
set(CUDAQ_INSTALL_PREFIX "" CACHE PATH
  "Prefix of a pre-built CUDA-Q installation to build against instead of the \
cudaq-devel wheel installed in the active Python environment")

if(CUDAQ_INSTALL_PREFIX)
  set(_qlx_cudaq_prefix "${CUDAQ_INSTALL_PREFIX}")
  set(_qlx_cudaq_origin "CUDAQ_INSTALL_PREFIX")
  set(_qlx_cudaq_hint
    "Point -DCUDAQ_INSTALL_PREFIX at the CMAKE_INSTALL_PREFIX of a CUDA-Q "
    "build, or leave it empty to auto-detect the cudaq-devel SDK from the active Python environment.")
else()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
    "import sysconfig; print(sysconfig.get_paths()['purelib'], end='')"
    RESULT_VARIABLE _qlx_purelib_rc
    OUTPUT_VARIABLE _qlx_cudaq_prefix
    ERROR_VARIABLE _qlx_purelib_error)
  if(NOT _qlx_purelib_rc EQUAL 0 OR NOT _qlx_cudaq_prefix)
    message(FATAL_ERROR
      "Could not determine site-packages from ${Python3_EXECUTABLE}: "
      "${_qlx_purelib_error}")
  endif()
  set(_qlx_cudaq_origin "cudaq-devel wheel")
  set(_qlx_cudaq_hint
    "Install the CUDA-Q development wheel into ${Python3_EXECUTABLE} "
    "('pip install cudaq-devel'), or pass -DCUDAQ_INSTALL_PREFIX=<prefix> to "
    "build against a CUDA-Q installation built from source.")
endif()

if(NOT EXISTS "${_qlx_cudaq_prefix}/lib/cmake/cudaq/CUDAQConfig.cmake")
  message(FATAL_ERROR
    "No CUDA-Q development installation under ${_qlx_cudaq_prefix} "
    "(expected lib/cmake/cudaq/CUDAQConfig.cmake).\n"
    ${_qlx_cudaq_hint})
endif()

set(QLX_CUDAQ_PREFIX "${_qlx_cudaq_prefix}" CACHE INTERNAL
  "Resolved CUDA-Q prefix supplying CUDA-Q, LLVM and MLIR")

list(PREPEND CMAKE_PREFIX_PATH "${QLX_CUDAQ_PREFIX}")
message(STATUS "CUDA-Q:          ${QLX_CUDAQ_PREFIX} (${_qlx_cudaq_origin})")

# --------------------------------------------------------------------------- #
# CUDA-Q, and the LLVM/MLIR it was built against
# --------------------------------------------------------------------------- #
# CUDA-Q Logical consumes CMake targets and helpers only; it compiles no CUDA-Q kernels,
# so the nvq++ toolchain is not required.
# CUDAQConfig resolves the colocated LLVM/MLIR, puts their helper modules on
# CMAKE_MODULE_PATH, and includes AddCUDAQ -- so this one call is enough.
set(CUDAQ_ENABLE_LANGUAGE OFF)
find_package(CUDAQ REQUIRED CONFIG)
# If LLVM_DIR is set but not MLIR_DIR, look for a colocated MLIR installation.
# TODO: this could go in `/cmake/CUDAQConfig.cmake`
if (NOT MLIR_DIR AND EXISTS "${LLVM_DIR}/../mlir/MLIRConfig.cmake")
  get_filename_component(MLIR_DIR "${LLVM_DIR}/../mlir" ABSOLUTE)
endif()
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${CUDAQ_CMAKE_DIR}" "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddCUDAQ)

if(NOT TARGET cudaq::MLIR)
  message(FATAL_ERROR
    "${QLX_CUDAQ_PREFIX} does not export cudaq::MLIR. It is either a runtime "
    "installation or predates CUDA-Q's shared-MLIR packaging.")
endif()

# CUDA-Q Logical libraries declare the MLIR components they actually use. Bundled
# components resolve from libcudaqMLIR via cudaq::cudaqMLIR's interface link
# options; cudaq_use_static_mlir() still overrides per target when needed.

message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")
message(STATUS "Using MLIR from ${MLIR_DIR}")
message(STATUS "Shared MLIR:     ${CUDAQ_LIBRARY_DIR}")

# --------------------------------------------------------------------------- #
# Locate all required CUDA-Q Python binding libraries
# --------------------------------------------------------------------------- #
set(_qlx_cudaq_mlir_capi
  "${CUDAQ_LIBRARY_DIR}/libcudaqMLIRCAPI${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(_qlx_cudaq_mlir_libs "${QLX_CUDAQ_PREFIX}/cudaq/mlir/_mlir_libs")
find_file(_qlx_cudaq_py_support
  NAMES
    "libMLIRPythonSupport-cudaq${CMAKE_SHARED_LIBRARY_SUFFIX}"
    "MLIRPythonSupport-cudaq${CMAKE_SHARED_LIBRARY_SUFFIX}"
  PATHS "${_qlx_cudaq_mlir_libs}"
  NO_DEFAULT_PATH)
find_file(_qlx_cudaq_nanobind
  NAMES
    "libnanobind-cudaq${CMAKE_SHARED_LIBRARY_SUFFIX}"
    "nanobind-cudaq${CMAKE_SHARED_LIBRARY_SUFFIX}"
  PATHS "${_qlx_cudaq_mlir_libs}"
  NO_DEFAULT_PATH)

if(NOT EXISTS "${_qlx_cudaq_mlir_capi}")
  message(FATAL_ERROR
    "CUDA-Q runtime wheel is missing libcudaqMLIRCAPI "
    "(expected ${_qlx_cudaq_mlir_capi}).\n"
    "Install the matching cudaq / cuda-quantum runtime wheel into the same "
    "environment as cudaq-devel, or point -DQLX_CUDAQ_INSTALL_DIR at a CUDA-Q "
    "prefix that includes the Python bindings.")
endif()
if(NOT _qlx_cudaq_py_support)
  message(FATAL_ERROR
    "CUDA-Q runtime wheel is missing MLIRPythonSupport-cudaq "
    "(expected under ${_qlx_cudaq_mlir_libs}).\n"
    "Install the matching cudaq / cuda-quantum runtime wheel into the same "
    "environment as cudaq-devel.")
endif()

if(NOT TARGET cudaq::cudaqMLIRCAPI)
  add_library(cudaq::cudaqMLIRCAPI SHARED IMPORTED)
  set_target_properties(cudaq::cudaqMLIRCAPI PROPERTIES
    IMPORTED_LOCATION "${_qlx_cudaq_mlir_capi}"
    IMPORTED_NO_SONAME FALSE)
endif()
if(NOT TARGET cudaq::nanobind)
  if(_qlx_cudaq_nanobind)
    add_library(cudaq::nanobind SHARED IMPORTED)
    set_target_properties(cudaq::nanobind PROPERTIES
      IMPORTED_LOCATION "${_qlx_cudaq_nanobind}")
  endif()
endif()
if(NOT TARGET cudaq::MLIRPythonSupport)
  add_library(cudaq::MLIRPythonSupport SHARED IMPORTED)
  set_target_properties(cudaq::MLIRPythonSupport PROPERTIES
    IMPORTED_LOCATION "${_qlx_cudaq_py_support}")
  target_link_libraries(cudaq::MLIRPythonSupport INTERFACE cudaq::cudaqMLIRCAPI)
  if(TARGET cudaq::nanobind)
    target_link_libraries(cudaq::MLIRPythonSupport INTERFACE cudaq::nanobind)
  endif()
endif()
message(STATUS "cudaqMLIRCAPI:   ${_qlx_cudaq_mlir_capi}")
message(STATUS "MLIRPythonSupport: ${_qlx_cudaq_py_support}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

if(NOT SKBUILD)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()
list(APPEND CMAKE_BUILD_RPATH "${CUDAQ_LIBRARY_DIR}")

# --------------------------------------------------------------------------- #
# LLVM helper tools (lit, FileCheck, ...) consumed by test/ and runtime/.
#
# The cudaq-devel wheel ships these next to the CUDA-Q tools, and a source
# CUDA-Q install colocates them under the same prefix; a PATH-visible copy is
# accepted too. Each is optional: test/CMakeLists.txt reports which suites it
# had to disable.
# --------------------------------------------------------------------------- #
# lit deliberately does not take LLVM_TOOLS_BINARY_DIR as a hint: the wrapper
# CUDA-Q installs is generated for its own LLVM build tree (hard-coded
# interpreter, sys.path pointing into the LLVM sources) and does not run
# elsewhere. Use the `lit` the developer installed instead.
find_program(LIT_PATH NAMES lit llvm-lit
  DOC "Path to lit or llvm-lit")
find_program(FILECHECK_PATH FileCheck
  HINTS "${LLVM_TOOLS_BINARY_DIR}" DOC "Path to FileCheck")
find_program(MLIR_TRANSLATE_PATH mlir-translate
  HINTS "${LLVM_TOOLS_BINARY_DIR}" DOC "Path to mlir-translate")
find_program(QLX_NOT_PATH not
  HINTS "${LLVM_TOOLS_BINARY_DIR}" DOC "Path to not")
find_program(QLX_COUNT_PATH count
  HINTS "${LLVM_TOOLS_BINARY_DIR}" DOC "Path to count")
find_program(CLANGXX_PATH clang++
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  DOC "Path to clang++ (used by fabric-execute lit tests)")

foreach(_qlx_tool IN ITEMS
  "lit;${LIT_PATH}"
  "FileCheck;${FILECHECK_PATH}"
  "mlir-translate;${MLIR_TRANSLATE_PATH}"
  "not;${QLX_NOT_PATH}"
  "count;${QLX_COUNT_PATH}"
  "clang++;${CLANGXX_PATH}")
  list(GET _qlx_tool 0 _qlx_tool_name)
  list(GET _qlx_tool 1 _qlx_tool_path)
  if(_qlx_tool_path)
    message(STATUS "${_qlx_tool_name}: ${_qlx_tool_path}")
  endif()
endforeach()
