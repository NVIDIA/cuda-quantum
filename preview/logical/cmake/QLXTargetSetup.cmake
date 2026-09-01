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

``qlx_propagate_llvm_rtti_eh(<target>)``
  Adds ``-fno-rtti`` and ``-fno-exceptions`` to ``<target>``'s compile
  options whenever the host LLVM was built with those features
  disabled. This is what every MLIR dialect needs so the generated
  TableGen op classes match the upstream LLVM ABI.

Position-independent code: the top-level ``CMakeLists.txt`` sets
``CMAKE_POSITION_INDEPENDENT_CODE ON`` globally, so per-target
``POSITION_INDEPENDENT_CODE`` properties are unnecessary.

#]=======================================================================]

function(qlx_propagate_llvm_rtti_eh target)
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(${target} PRIVATE -fno-rtti)
  endif()
  if(NOT LLVM_ENABLE_EH)
    target_compile_options(${target} PRIVATE -fno-exceptions)
  endif()
endfunction()
