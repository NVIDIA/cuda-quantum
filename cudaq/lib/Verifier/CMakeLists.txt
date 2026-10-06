# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_library(cudaq-qir-verifier
  NVQIRCalls.cpp
  QIRLLVMIRDialect.cpp
  QIRSpec.cpp

  DEPENDS
    CCDialect
    QuakeDialect)

# Link MLIR dynamically (default) or statically (if CUDAQ_MLIR_STATIC is set).
target_link_libraries(cudaq-qir-verifier PUBLIC
  "$<IF:$<BOOL:$<TARGET_PROPERTY:CUDAQ_MLIR_STATIC>>,cudaq-mlir-static,cudaqMLIR>")
