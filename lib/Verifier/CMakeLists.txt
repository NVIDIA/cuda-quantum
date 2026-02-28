# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_library(cudaq-qir-verifier
  NVQIRCalls.cpp

  LINK_LIBS PUBLIC
    MLIRLLVMDialect
    MLIRIR
)
