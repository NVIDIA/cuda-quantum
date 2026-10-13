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

Build modes
^^^^^^^^^^^

*In-tree* (``CUDAQ_ENABLE_LOGICAL=ON`` in the CUDA-Q build, or
``CUDAQ_EXTERNAL_PROJECTS``): CUDA-Q's own targets are used directly. Nothing is
searched for; the enclosing configure is consistent by construction.

*Out-of-tree* (``cmake -S preview/logical``): Logical is built against an
*installation* of CUDA-Q -- the result of ``cmake --install`` on a CUDA-Q build
-- that matches this source tree. The installation is found, in order, from

1. ``-DCUDAQ_INSTALL_PREFIX=<prefix>``,
2. ``CUDAQ_DIR`` / ``CMAKE_PREFIX_PATH`` (the usual ``find_package(CUDAQ)``
   conventions), then
3. as a last resort, the ``cudaq-devel`` wheel in the active Python environment
   (this is how the Logical wheel itself is built).

The build engineer is trusted to supply a CUDA-Q installation that is in
lockstep with these sources and whose LLVM/MLIR is the one CUDA-Q was built
with (colocated in the prefix via ``-DCUDAQ_BUNDLE_MLIR_INSTALL=ON``, or named
by ``LLVM_DIR`` / ``MLIR_DIR``). No cross-checks are made by default. To audit
the setup anyway, configure with ``-DQLX_VERIFY_TOOLCHAIN=ON``: it fails the
configure if the LLVM/MLIR found does not match the one libcudaqMLIR was built
against, or if this project's C++ compiler family differs from the one CUDA-Q
was built with. (The compiler must be the same family -- clang vs GCC -- for
everything: mixing them builds fine and then fails at runtime.)

After inclusion the rest of the build has:

- ``LLVM_INCLUDE_DIRS`` / ``MLIR_INCLUDE_DIRS`` on the include path
- the ``TableGen``, ``AddLLVM``, ``AddMLIR`` and ``AddCUDAQ`` helpers
- ``cudaq::MLIR`` -- the shared MLIR/LLVM image every CUDA-Q Logical target links
- ``CUDAQ_INCLUDE_DIR`` / ``CUDAQ_LIBRARY_DIR``
- the LLVM test utilities as ``LIT_PATH``, ``FILECHECK_PATH``,
  ``MLIR_TRANSLATE_PATH``, ``CUDAQ_LOGICAL_NOT_PATH``,
  ``CUDAQ_LOGICAL_COUNT_PATH``,
  ``CLANGXX_PATH``

#]=======================================================================]

include_guard(GLOBAL)

# --------------------------------------------------------------------------- #
# In-tree: everything below is a target of this very configure
# --------------------------------------------------------------------------- #
if(QLX_IN_TREE)
  foreach(_qlx_required_target
      cudaq::MLIR cudaq::cudaqMLIRCAPI
      CUDAQCorePythonModules.extension.MLIRPythonSupport-cudaq.so
      nanobind-cudaq)
    if(NOT TARGET ${_qlx_required_target})
      message(FATAL_ERROR
        "CUDA-Q Logical needs ${_qlx_required_target} from the enclosing "
        "CUDA-Q build. Enable CUDA-Q's Python bindings "
        "(-DCUDAQ_ENABLE_PYTHON=ON) and the 'cudaq' project.")
    endif()
  endforeach()

  # The staged Python package of the enclosing build.
  set(CUDAQ_LOGICAL_CUDAQ_PREFIX "${MLIR_BINARY_DIR}/python" CACHE INTERNAL "")
  set(CUDAQ_CORE_LIBRARY_DIR "${CUDAQ_LIBRARY_DIR}")
  set(_cudaq_logical_mlir_libs "${CUDAQ_LOGICAL_CUDAQ_PREFIX}/cudaq/mlir/_mlir_libs")

  # Same names the standalone mode imports, backed by the real targets.
  if(NOT TARGET cudaq::nanobind)
    add_library(cudaq::nanobind ALIAS nanobind-cudaq)
  endif()
  if(NOT TARGET cudaq::MLIRPythonSupport)
    add_library(cudaq::MLIRPythonSupport ALIAS
      CUDAQCorePythonModules.extension.MLIRPythonSupport-cudaq.so)
  endif()

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(AddCUDAQ)
  find_package(GMP REQUIRED COMPONENTS Headers)
  find_package(MPFR REQUIRED COMPONENTS Headers)
  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
else()
  # --------------------------------------------------------------------------- #
  # Resolve the CUDA-Q installation
  # --------------------------------------------------------------------------- #
  set(CUDAQ_INSTALL_PREFIX "" CACHE PATH
    "Prefix of the CUDA-Q installation (cmake --install) to build against")
  option(QLX_VERIFY_TOOLCHAIN
    "Fail the configure if LLVM/MLIR does not match what CUDA-Q was built with" OFF)

  set(_cudaq_logical_cudaq_prefix "")
  if(CUDAQ_INSTALL_PREFIX)
    set(_cudaq_logical_cudaq_prefix "${CUDAQ_INSTALL_PREFIX}")
    set(_cudaq_logical_cudaq_origin "CUDAQ_INSTALL_PREFIX")
  else()
    # Standard package conventions: CUDAQ_DIR or CMAKE_PREFIX_PATH.
    if(CUDAQ_DIR)
      get_filename_component(_cudaq_logical_cudaq_prefix
        "${CUDAQ_DIR}/../../.." ABSOLUTE)
      set(_cudaq_logical_cudaq_origin "CUDAQ_DIR")
    else()
      foreach(_cudaq_logical_candidate IN LISTS CMAKE_PREFIX_PATH)
        if(EXISTS "${_cudaq_logical_candidate}/lib/cmake/cudaq/CUDAQConfig.cmake")
          set(_cudaq_logical_cudaq_prefix "${_cudaq_logical_candidate}")
          set(_cudaq_logical_cudaq_origin "CMAKE_PREFIX_PATH")
          break()
        endif()
      endforeach()
    endif()
    if(NOT _cudaq_logical_cudaq_prefix)
      # Last resort: the cudaq-devel wheel in the active Python environment.
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
        "import sysconfig; print(sysconfig.get_paths()['purelib'], end='')"
        RESULT_VARIABLE _cudaq_logical_purelib_rc
        OUTPUT_VARIABLE _cudaq_logical_cudaq_prefix
        ERROR_QUIET)
      set(_cudaq_logical_cudaq_origin "cudaq-devel wheel")
    endif()
  endif()

  if(NOT _cudaq_logical_cudaq_prefix
      OR NOT EXISTS
        "${_cudaq_logical_cudaq_prefix}/lib/cmake/cudaq/CUDAQConfig.cmake")
    message(FATAL_ERROR
      "No CUDA-Q installation found.\n"
      "Pass -DCUDAQ_INSTALL_PREFIX=<prefix> naming a CUDA-Q installation "
      "(the prefix given to 'cmake --install' on a CUDA-Q build), or put it on "
      "CMAKE_PREFIX_PATH. It must contain lib/cmake/cudaq/CUDAQConfig.cmake.")
  endif()

  # Logical is part of CUDA-Q: unless told otherwise, install into the same tree
  # (bin/, cudaq/logical, cudaq/mlir/_mlir_libs) as the installation it was
  # built against. Skipped for wheels, whose prefix is site-packages.
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
      AND NOT _cudaq_logical_cudaq_origin STREQUAL "cudaq-devel wheel")
    set(CMAKE_INSTALL_PREFIX "${_cudaq_logical_cudaq_prefix}" CACHE PATH
      "Install path prefix (defaults to the CUDA-Q installation)" FORCE)
  endif()

  set(CUDAQ_LOGICAL_CUDAQ_PREFIX "${_cudaq_logical_cudaq_prefix}" CACHE INTERNAL
    "Resolved CUDA-Q prefix supplying CUDA-Q, LLVM and MLIR")

  list(PREPEND CMAKE_PREFIX_PATH "${CUDAQ_LOGICAL_CUDAQ_PREFIX}")
  message(STATUS
    "CUDA-Q:          ${CUDAQ_LOGICAL_CUDAQ_PREFIX} (${_cudaq_logical_cudaq_origin})")

  # --------------------------------------------------------------------------- #
  # CUDA-Q, and the LLVM/MLIR it was built against
  # --------------------------------------------------------------------------- #
  # CUDA-Q Logical consumes CMake targets and helpers only; it compiles no CUDA-Q
  # kernels, so the nvq++ toolchain is not required.
  # CUDAQConfig resolves the colocated LLVM/MLIR, puts their helper modules on
  # CMAKE_MODULE_PATH, and includes AddCUDAQ -- so this one call is enough.
  set(CUDAQ_ENABLE_LANGUAGE OFF)
  find_package(CUDAQ REQUIRED CONFIG)

  # QLXSynthesize directly includes CUDA-Q's rotation-synthesis headers, which
  # need the GMP/MPFR development headers.
  find_package(GMP REQUIRED COMPONENTS Headers)
  find_package(MPFR REQUIRED COMPONENTS Headers)

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
      "${CUDAQ_LOGICAL_CUDAQ_PREFIX} does not export cudaq::MLIR. It is either a runtime "
      "installation or predates CUDA-Q's shared-MLIR packaging.")
  endif()

  # CUDA-Q Logical libraries declare the MLIR components they actually use. Bundled
  # components resolve from libcudaqMLIR via cudaq::cudaqMLIR's interface link
  # options; cudaq_use_static_mlir() still overrides per target when needed.

  message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")
  message(STATUS "Using MLIR from ${MLIR_DIR}")

  # Opt-in audit of the toolchain: libcudaqMLIR embeds the "LLVM version X.Y.Z"
  # string of the LLVM it was built with.
  if(QLX_VERIFY_TOOLCHAIN)
    set(_cudaq_logical_llvm_built "")
    find_program(_cudaq_logical_grep grep)
    if(_cudaq_logical_grep AND EXISTS "${CUDAQ_LIBRARY_DIR}/libcudaqMLIR${CMAKE_SHARED_LIBRARY_SUFFIX}")
      execute_process(
        COMMAND "${_cudaq_logical_grep}" -a -o -m1 "LLVM version [0-9][0-9.]*"
          "${CUDAQ_LIBRARY_DIR}/libcudaqMLIR${CMAKE_SHARED_LIBRARY_SUFFIX}"
        OUTPUT_VARIABLE _cudaq_logical_llvm_built
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      string(REGEX REPLACE "^LLVM version " "" _cudaq_logical_llvm_built
        "${_cudaq_logical_llvm_built}")
    endif()
    if(_cudaq_logical_llvm_built)
      if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL _cudaq_logical_llvm_built)
        message(FATAL_ERROR
          "QLX_VERIFY_TOOLCHAIN: found LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR}) "
          "but CUDA-Q at ${CUDAQ_LOGICAL_CUDAQ_PREFIX} was built with LLVM "
          "${_cudaq_logical_llvm_built}.")
      endif()
      message(STATUS
        "QLX_VERIFY_TOOLCHAIN: LLVM ${LLVM_PACKAGE_VERSION} matches the LLVM CUDA-Q was built with")
    else()
      message(WARNING
        "QLX_VERIFY_TOOLCHAIN: could not determine the LLVM version "
        "libcudaqMLIR was built with; skipping the version check.")
    endif()
    # The C++ compiler must match the one CUDA-Q was built with. MLIR derives some
    # TypeIDs (e.g. op traits like SymbolTable) from compiler-specific type-name
    # strings, so objects from a different compiler family disagree with
    # libcudaqMLIR about them; the symptom is baffling verifier failures at
    # runtime, not a build error. The ELF .comment section records the compilers
    # of every object in the library: a clang-built object adds "clang version N"
    # (the GCC: entries a clang-built library also carries come from the system
    # crt objects and say nothing), a GCC-built object adds "GCC: (...) N".
    find_program(_cudaq_logical_readelf readelf)
    set(_cudaq_logical_comment "")
    if(_cudaq_logical_readelf AND
        EXISTS "${CUDAQ_LIBRARY_DIR}/libcudaqMLIR${CMAKE_SHARED_LIBRARY_SUFFIX}")
      execute_process(
        COMMAND "${_cudaq_logical_readelf}" -p .comment
          "${CUDAQ_LIBRARY_DIR}/libcudaqMLIR${CMAKE_SHARED_LIBRARY_SUFFIX}"
        OUTPUT_VARIABLE _cudaq_logical_comment
        ERROR_QUIET)
    endif()
    if(_cudaq_logical_comment)
      if(_cudaq_logical_comment MATCHES "clang version ([0-9]+)")
        set(_cudaq_logical_built_family "Clang")
        set(_cudaq_logical_built_major "${CMAKE_MATCH_1}")
      elseif(_cudaq_logical_comment MATCHES "GCC: \\([^)]*\\) ([0-9]+)")
        set(_cudaq_logical_built_family "GNU")
        set(_cudaq_logical_built_major "${CMAKE_MATCH_1}")
      else()
        set(_cudaq_logical_built_family "")
      endif()
      set(_cudaq_logical_this_family "${CMAKE_CXX_COMPILER_ID}")
      string(REGEX REPLACE "^AppleClang$" "Clang" _cudaq_logical_this_family
        "${_cudaq_logical_this_family}")
      string(REGEX MATCH "^[0-9]+" _cudaq_logical_this_major
        "${CMAKE_CXX_COMPILER_VERSION}")
      if(_cudaq_logical_built_family)
        if(NOT _cudaq_logical_this_family STREQUAL _cudaq_logical_built_family)
          message(FATAL_ERROR
            "QLX_VERIFY_TOOLCHAIN: CUDA-Q at ${CUDAQ_LOGICAL_CUDAQ_PREFIX} was built "
            "with ${_cudaq_logical_built_family} ${_cudaq_logical_built_major}, but "
            "this project is being built with ${_cudaq_logical_this_family} "
            "${_cudaq_logical_this_major} (${CMAKE_CXX_COMPILER}). Build everything "
            "with the same compiler family: configure with CC/CXX (or "
            "-DCMAKE_C_COMPILER/-DCMAKE_CXX_COMPILER) set to the compiler CUDA-Q "
            "was built with. Mixing them compiles fine but fails at runtime.")
        endif()
        if(NOT _cudaq_logical_this_major STREQUAL _cudaq_logical_built_major)
          message(WARNING
            "QLX_VERIFY_TOOLCHAIN: CUDA-Q was built with "
            "${_cudaq_logical_built_family} ${_cudaq_logical_built_major}; this project "
            "uses ${_cudaq_logical_this_family} ${_cudaq_logical_this_major}. "
            "The same family is required; a different major version is usually fine.")
        else()
          message(STATUS
            "QLX_VERIFY_TOOLCHAIN: compiler ${_cudaq_logical_this_family} "
            "${_cudaq_logical_this_major} matches the one CUDA-Q was built with")
        endif()
      endif()
    else()
      message(STATUS
        "QLX_VERIFY_TOOLCHAIN: no ELF .comment section to inspect; skipping the compiler check")
    endif()

    # The LLVM/MLIR found must be the one libcudaqMLIR was built against, i.e.
    # colocated in the CUDA-Q prefix, unless the user pointed elsewhere on purpose.
    if(NOT LLVM_DIR MATCHES "^${CUDAQ_LOGICAL_CUDAQ_PREFIX}/")
      message(STATUS
        "QLX_VERIFY_TOOLCHAIN: LLVM_DIR (${LLVM_DIR}) is outside the CUDA-Q "
        "prefix; version ${LLVM_PACKAGE_VERSION} accepted.")
    endif()
  endif()

  if(NOT CUDAQ_CORE_LIBRARY_DIR)
    set(CUDAQ_CORE_LIBRARY_DIR "${CUDAQ_LIBRARY_DIR}")
  endif()
  message(STATUS "Shared MLIR:     ${CUDAQ_CORE_LIBRARY_DIR}")

  # --------------------------------------------------------------------------- #
  # Locate all required CUDA-Q Python binding libraries
  # --------------------------------------------------------------------------- #
  set(_cudaq_logical_mlir_capi
    "${CUDAQ_CORE_LIBRARY_DIR}/libcudaqMLIRCAPI${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(_cudaq_logical_mlir_libs
    "${CUDAQ_LOGICAL_CUDAQ_PREFIX}/cudaq/mlir/_mlir_libs")
  if(EXISTS "${CUDAQ_LOGICAL_CUDAQ_PREFIX}/cudaq_core")
    set(_cudaq_logical_mlir_libs
      "${CUDAQ_LOGICAL_CUDAQ_PREFIX}/cudaq_core/cudaq/mlir/_mlir_libs")
    # AddCUDAQ supplies the relative wheel RPATHs for split binding providers.
    set(CUDAQ_BUILD_SPLIT_WHEELS ON)
  endif()
  find_file(_cudaq_logical_python_support
    NAMES
      "libMLIRPythonSupport-cudaq${CMAKE_SHARED_LIBRARY_SUFFIX}"
      "MLIRPythonSupport-cudaq${CMAKE_SHARED_LIBRARY_SUFFIX}"
    PATHS "${_cudaq_logical_mlir_libs}"
    NO_DEFAULT_PATH)
  find_file(_cudaq_logical_nanobind
    NAMES
      "libnanobind-cudaq${CMAKE_SHARED_LIBRARY_SUFFIX}"
      "nanobind-cudaq${CMAKE_SHARED_LIBRARY_SUFFIX}"
    PATHS "${_cudaq_logical_mlir_libs}"
    NO_DEFAULT_PATH)

  if(NOT EXISTS "${_cudaq_logical_mlir_capi}")
    message(FATAL_ERROR
      "CUDA-Q installation is missing libcudaqMLIRCAPI "
      "(expected ${_cudaq_logical_mlir_capi}).\n"
      "Install the matching cudaq-core wheel into the same "
      "environment as cudaq-devel, or point "
      "-DCUDAQ_INSTALL_PREFIX at a CUDA-Q "
      "prefix that includes the Python bindings.")
  endif()
  if(NOT _cudaq_logical_python_support)
    message(FATAL_ERROR
      "CUDA-Q installation is missing MLIRPythonSupport-cudaq "
      "(expected under ${_cudaq_logical_mlir_libs}).\n"
      "Install the matching cudaq-core wheel into the same "
      "environment as cudaq-devel.")
  endif()

  if(NOT TARGET cudaq::cudaqMLIRCAPI)
    add_library(cudaq::cudaqMLIRCAPI SHARED IMPORTED)
    set_target_properties(cudaq::cudaqMLIRCAPI PROPERTIES
      IMPORTED_LOCATION "${_cudaq_logical_mlir_capi}"
      IMPORTED_NO_SONAME FALSE)
  endif()
  if(NOT TARGET cudaq::nanobind)
    if(_cudaq_logical_nanobind)
      add_library(cudaq::nanobind SHARED IMPORTED)
      set_target_properties(cudaq::nanobind PROPERTIES
        IMPORTED_LOCATION "${_cudaq_logical_nanobind}")
    endif()
  endif()
  if(NOT TARGET cudaq::MLIRPythonSupport)
    add_library(cudaq::MLIRPythonSupport SHARED IMPORTED)
    set_target_properties(cudaq::MLIRPythonSupport PROPERTIES
      IMPORTED_LOCATION "${_cudaq_logical_python_support}")
    target_link_libraries(cudaq::MLIRPythonSupport INTERFACE cudaq::cudaqMLIRCAPI)
    if(TARGET cudaq::nanobind)
      target_link_libraries(cudaq::MLIRPythonSupport INTERFACE cudaq::nanobind)
    endif()
  endif()
  message(STATUS "cudaqMLIRCAPI:   ${_cudaq_logical_mlir_capi}")
  message(STATUS "MLIRPythonSupport: ${_cudaq_logical_python_support}")

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
endif()


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
# In-tree, the enclosing build's own lit is by definition the right one.
if(QLX_IN_TREE AND LLVM_EXTERNAL_LIT AND NOT LIT_PATH)
  set(LIT_PATH "${LLVM_EXTERNAL_LIT}" CACHE FILEPATH "Path to lit or llvm-lit" FORCE)
endif()
find_program(LIT_PATH NAMES lit llvm-lit
  DOC "Path to lit or llvm-lit")
find_program(FILECHECK_PATH FileCheck
  HINTS "${LLVM_TOOLS_BINARY_DIR}" DOC "Path to FileCheck")
find_program(MLIR_TRANSLATE_PATH mlir-translate
  HINTS "${LLVM_TOOLS_BINARY_DIR}" DOC "Path to mlir-translate")
find_program(CUDAQ_LOGICAL_NOT_PATH not
  HINTS "${LLVM_TOOLS_BINARY_DIR}" DOC "Path to not")
find_program(CUDAQ_LOGICAL_COUNT_PATH count
  HINTS "${LLVM_TOOLS_BINARY_DIR}" DOC "Path to count")
find_program(CLANGXX_PATH clang++
  HINTS "${LLVM_TOOLS_BINARY_DIR}"
  DOC "Path to clang++ (used by fabric-execute lit tests)")

foreach(_cudaq_logical_tool IN ITEMS
  "lit;${LIT_PATH}"
  "FileCheck;${FILECHECK_PATH}"
  "mlir-translate;${MLIR_TRANSLATE_PATH}"
  "not;${CUDAQ_LOGICAL_NOT_PATH}"
  "count;${CUDAQ_LOGICAL_COUNT_PATH}"
  "clang++;${CLANGXX_PATH}")
  list(GET _cudaq_logical_tool 0 _cudaq_logical_tool_name)
  list(GET _cudaq_logical_tool 1 _cudaq_logical_tool_path)
  if(_cudaq_logical_tool_path)
    message(STATUS
      "${_cudaq_logical_tool_name}: ${_cudaq_logical_tool_path}")
  endif()
endforeach()
