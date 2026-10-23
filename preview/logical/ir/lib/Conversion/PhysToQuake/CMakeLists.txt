# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Lowers P3 to Quake, so it links CUDA-Q's Quake and CC dialects and builder
# helpers through `QLXCUDAQuantumDialects` (see Conversion/QuakeToQLX). That
# target exists only when QLX_USE_CUDAQ_SDK is ON, the default.

qlx_select_mlir_link_libraries(_qlx_mlir_link_libraries
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)

add_mlir_conversion_library(MLIRPhysToQuake
  PhysToQuake.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Conversion

  DEPENDS
  MLIRPhysToQuakePassesIncGen

  LINK_LIBS PUBLIC
  MLIRCflowDialect
  MLIREventDialect
  MLIRPhysDialect
  QLXCUDAQuantumDialects
  ${_qlx_mlir_link_libraries}
)

qlx_propagate_llvm_rtti_eh(MLIRPhysToQuake)
