# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CUDAQ_TARGET_CONFIG_OPTIONS)
# When LLVM is built as a shared library (dylib), we must NOT disable dylib
# linking to avoid duplicate LLVM symbols (e.g., APFloat semantics like IEEEdouble).
# Multiple copies of these static symbols cause assertion failures in MLIR.
if(NOT LLVM_LINK_LLVM_DYLIB)
  set(CUDAQ_TARGET_CONFIG_OPTIONS DISABLE_LLVM_LINK_LLVM_DYLIB)
endif()

add_llvm_library(CUDAQTargetConfigUtil
  TargetConfig.cpp

  ${CUDAQ_TARGET_CONFIG_OPTIONS}

  LINK_COMPONENTS
  Support
  )

# macOS Two-Level Namespace: Symbol Hiding via Unexport List
#
# Problem: CUDAQTargetConfigUtil links LLVMSupport, pulling in LLVM symbols.
# When other libraries also link LLVM, macOS two-level namespace causes each
# library to have its own copy of static LLVM data (e.g., cl::Options).
#
# Solution: Hide LLVM/MLIR symbols from CUDAQTargetConfigUtil so consumers
# get these symbols from cudaq-mlir-runtime instead (which centralizes them).
#
# The generator expression ensures this only applies to SHARED_LIBRARY targets,
# not executables (which need their own LLVM symbols for GlobalParser).
if(APPLE)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/llvm_unexport.txt
"_ZN4llvm*
_ZN4mlir*
")
  target_link_options(CUDAQTargetConfigUtil INTERFACE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:-Wl,-unexported_symbols_list,${CMAKE_CURRENT_BINARY_DIR}/llvm_unexport.txt>")
endif()
