# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[=======================================================================[.rst:
QLXPythonEnv
------------

Python environment detection for the CUDA-Q Logical project.

Provides:

``qlx_configure_python_env([REQUIRED])``
  Locate Python 3.10+ (Interpreter + Development.Module) and probe the
  active environment for ``nanobind``.  When ``REQUIRED`` is given,
  fail with ``FATAL_ERROR`` if either Python or nanobind is missing.

  Sets the standard ``Python3_*`` and ``Python_*`` variables
  (``find_package`` caches these) so that both CUDA-Q Logical's Python extension
  (``python/CMakeLists.txt``) and a bundled MLIR build's Python
  bindings see the same interpreter.  When nanobind is found, caches
  ``nanobind_DIR`` so MLIR's ``MLIRDetectPythonEnv`` skips its own
  ``import nanobind`` probe and ``find_package(nanobind)`` succeeds
  immediately.

  Sets ``QLX_PYTHON_ENV_FOUND`` to ``TRUE``/``FALSE`` to indicate
  whether a usable environment was found (always ``TRUE`` when
  ``REQUIRED`` is passed, since the macro otherwise fails fatally).

``qlx_check_python_module(<module> [<result_var>] [REQUIRED])``
  Check whether a Python module is importable in the configured
  environment.  When ``REQUIRED`` is given, fail with ``FATAL_ERROR``
  if the import fails.  When ``<result_var>`` is given, set it to
  ``TRUE``/``FALSE`` in the caller's scope.

  Examples::

    qlx_check_python_module(numpy QLX_HAS_NUMPY)
    qlx_check_python_module(nanobind REQUIRED)

#]=======================================================================]

include_guard(GLOBAL)

# --------------------------------------------------------------------------- #
# qlx_check_python_module(<module> [<result_var>] [REQUIRED])
# --------------------------------------------------------------------------- #
function(qlx_check_python_module module)
  cmake_parse_arguments(ARG "REQUIRED" "" "" ${ARGN})
  set(_out_var ${ARG_UNPARSED_ARGUMENTS})

  if(NOT Python3_EXECUTABLE)
    if(ARG_REQUIRED)
      message(FATAL_ERROR
        "qlx_check_python_module(${module}): Python3 has not been "
        "configured yet.  Call qlx_configure_python_env() first.")
    endif()
    if(_out_var)
      set(${_out_var} FALSE PARENT_SCOPE)
    endif()
    return()
  endif()

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import ${module}"
    RESULT_VARIABLE _status
    OUTPUT_QUIET ERROR_QUIET)

  if(_status EQUAL 0)
    if(_out_var)
      set(${_out_var} TRUE PARENT_SCOPE)
    endif()
  else()
    if(_out_var)
      set(${_out_var} FALSE PARENT_SCOPE)
    endif()
    if(ARG_REQUIRED)
      message(FATAL_ERROR
        "Required Python module '${module}' not found in "
        "${Python3_EXECUTABLE}.\n"
        "Install via 'pip install ${module}'.")
    endif()
  endif()
endfunction()

# --------------------------------------------------------------------------- #
# qlx_configure_python_env([REQUIRED])
# --------------------------------------------------------------------------- #
macro(qlx_configure_python_env)
  cmake_parse_arguments(_qlx_py "REQUIRED" "" "" ${ARGN})

  if(_qlx_py_REQUIRED)
    set(_qlx_py_required REQUIRED)
  else()
    set(_qlx_py_required "")
  endif()

  # We need Interpreter (to drive lit / pytest / mlir-tblgen helpers
  # and to probe modules) and Development.Module (for nanobind to
  # build extension modules).  We do NOT request Development.Embed:
  # nothing in CUDA-Q Logical embeds a Python interpreter.
  #
  # 3.10 is the floor supported by the matching CUDA-Q development SDK.
  # CUDA-Q Logical's Python extensions must use the same ABI as that SDK's MLIR Python
  # extensions.
  find_package(Python3 3.10
    COMPONENTS Interpreter Development.Module
    ${_qlx_py_required})

  # MLIR's Python binding cmake (MLIRDetectPythonEnv) and nanobind's
  # own cmake look for "Python" (no "3" suffix).  Force the
  # un-suffixed Python find to use the SAME interpreter we just found
  # for Python3 -- otherwise CMake's two find_package modules can pick
  # different interpreters (e.g. Python3 = .venv 3.12, Python = system
  # 3.13), and MLIR's Python C extensions then get built against a
  # different ABI than CUDA-Q Logical's tests run against.
  if(NOT Python_EXECUTABLE OR NOT Python_EXECUTABLE STREQUAL Python3_EXECUTABLE)
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)
  endif()
  find_package(Python 3.10
    COMPONENTS Interpreter Development.Module
    ${_qlx_py_required})

  if(Python3_Interpreter_FOUND)
    message(STATUS "CUDA-Q Logical Python: ${Python3_EXECUTABLE} (${Python3_VERSION})")
    if(Python3_INCLUDE_DIRS)
      message(STATUS "  include dirs:    ${Python3_INCLUDE_DIRS}")
    endif()
    if(Python3_SOABI)
      message(STATUS "  extension SOABI: ${Python3_SOABI}")
    endif()

    # Probe nanobind once and cache its CMake config dir.  This makes
    # MLIRDetectPythonEnv's nanobind probe a no-op (it skips the
    # `python -c "import nanobind"` call when nanobind_DIR is already
    # set) and ensures the qlx Python extension can call
    # nanobind_add_module().
    if(NOT nanobind_DIR)
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
          "import nanobind; print(nanobind.cmake_dir(), end='')"
        RESULT_VARIABLE _qlx_nanobind_rc
        OUTPUT_VARIABLE _qlx_nanobind_dir
        ERROR_QUIET)
      if(_qlx_nanobind_rc EQUAL 0)
        set(nanobind_DIR "${_qlx_nanobind_dir}" CACHE PATH
          "Path to nanobind's CMake config (probed via Python)")
        message(STATUS "  nanobind:        ${nanobind_DIR}")
      elseif(_qlx_py_REQUIRED)
        message(FATAL_ERROR
          "qlx_configure_python_env(REQUIRED): nanobind not found in "
          "${Python3_EXECUTABLE}.\n"
          "Install via 'pip install nanobind' or pass "
          "-Dnanobind_DIR=/path/to/nanobind/cmake.")
      else()
        message(STATUS
          "  nanobind:        not found "
          "('pip install nanobind' to enable Python bindings)")
      endif()
    else()
      message(STATUS "  nanobind:        ${nanobind_DIR} (preset)")
    endif()

    set(QLX_PYTHON_ENV_FOUND TRUE)
  else()
    set(QLX_PYTHON_ENV_FOUND FALSE)
  endif()

  unset(_qlx_py_REQUIRED)
  unset(_qlx_py_required)
  unset(_qlx_nanobind_rc)
  unset(_qlx_nanobind_dir)
endmacro()
