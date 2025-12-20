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
