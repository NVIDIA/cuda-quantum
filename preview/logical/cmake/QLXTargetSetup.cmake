# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[=======================================================================[.rst:
QLXTargetSetup
--------------

Shared helpers for CUDA-Q Logical-internal CMake targets, factoring out the
LLVM/MLIR-style "match the host LLVM's RTTI / EH settings" boilerplate
that otherwise gets duplicated across every dialect / conversion /
transform CMakeLists.

Functions:

``qlx_select_mlir_link_libraries(<output> <components...>)``
  Selects CUDA-Q's shared MLIR target for an internal logical target. The
  component list documents the precise MLIR surface used by the target, while
  the SDK supplies those symbols through ``cudaq::MLIR``.

``qlx_propagate_llvm_rtti_eh(<target> [EXCEPTIONS])``
  Adds ``-fno-rtti`` and ``-fno-exceptions`` to ``<target>``'s compile
  options whenever the host LLVM was built with those features
  disabled. This is what every MLIR dialect needs so the generated
  TableGen op classes match the upstream LLVM ABI.
  ``EXCEPTIONS`` keeps RTTI consistent with LLVM but enables exception
  unwinding for allocating analyses whose C boundary handles resource failures.

Position-independent code: the top-level ``CMakeLists.txt`` sets
``CMAKE_POSITION_INDEPENDENT_CODE ON`` globally, so per-target
``POSITION_INDEPENDENT_CODE`` properties are unnecessary.

#]=======================================================================]

function(qlx_select_mlir_link_libraries output)
  set(${output} cudaq::MLIR PARENT_SCOPE)
endfunction()

function(_qlx_collect_directory_targets directory output)
  get_property(_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  get_property(_subdirs DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(_subdir IN LISTS _subdirs)
    _qlx_collect_directory_targets("${_subdir}" _sub_targets)
    list(APPEND _targets ${_sub_targets})
  endforeach()
  set(${output} ${_targets} PARENT_SCOPE)
endfunction()

#[=======================================================================[.rst:
``qlx_drop_static_llvm_support(<aggregate-target>)``
  Removes ``LLVMSupport`` and ``LLVMDemangle`` from the link interface of
  ``<aggregate-target>`` and of every static library in this project.

  ``add_mlir_library`` links each MLIR library, including the aggregate
  itself, against the static ``LLVMSupport``. The aggregate
  (``QLXPythonCAPI``) then embeds a private copy of LLVM's command-line
  parser next to the one in ``libcudaqMLIR``. On macOS each image binds to
  its own copy, so pass options constructed in ``QLXPythonCAPI`` register
  with the private parser and ``PassOptions::parseFromString`` in
  ``libcudaqMLIR`` fails with ``no such option``. Link order does not help:
  ld64 prefers the archive definition over the dylib for these references.
  ``cudaq::MLIR`` already provides every LLVM symbol the project needs.
#]=======================================================================]
function(qlx_drop_static_llvm_support aggregate)
  _qlx_collect_directory_targets("${PROJECT_SOURCE_DIR}" _qlx_targets)
  foreach(_target IN LISTS _qlx_targets)
    get_target_property(_type ${_target} TYPE)
    if(NOT _type STREQUAL "STATIC_LIBRARY" AND NOT _target STREQUAL aggregate)
      continue()
    endif()
    foreach(_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
      get_target_property(_libs ${_target} ${_property})
      if(NOT _libs)
        continue()
      endif()
      set(_filtered ${_libs})
      list(FILTER _filtered EXCLUDE REGEX "^(LLVMSupport|LLVMDemangle)$")
      if(NOT _filtered STREQUAL _libs)
        set_property(TARGET ${_target} PROPERTY ${_property} "${_filtered}")
      endif()
    endforeach()
  endforeach()
endfunction()

function(qlx_propagate_llvm_rtti_eh target)
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(${target} PRIVATE -fno-rtti)
  endif()
  if("EXCEPTIONS" IN_LIST ARGN)
    target_compile_options(${target} PRIVATE -fexceptions)
    # add_mlir_library compiles sources in an object library before adding
    # them to the aggregate target. Enable unwinding where compilation occurs.
    if(TARGET obj.${target})
      target_compile_options(obj.${target} PRIVATE -fexceptions)
    endif()
  elseif(NOT LLVM_ENABLE_EH)
    target_compile_options(${target} PRIVATE -fno-exceptions)
  endif()
endfunction()
