# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Unit test for qlx/Conversion/PauliProductToQuake.h. Needs Quake, which is
# available only through CUDA-Q's shared compiler target (see
# lib/Conversion/QuakeToQLX), so the caller guards this on QLX_USE_CUDAQ_SDK.
# The test is registered with CTest from test/CMakeLists.txt, which is added
# after enable_testing().

add_llvm_executable(qlx-pauli-product-test
  PauliProductToQuakeTest.cpp
)

qlx_select_mlir_link_libraries(_qlx_mlir_link_libraries
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransformUtils
)

target_link_libraries(qlx-pauli-product-test PRIVATE
  MLIRQLXDialect
  MLIRLVMDialect
  MLIRPhysDialect
  MLIRCflowDialect
  MLIREventDialect
  MLIRFabricDialect
  QLXCUDAQuantumDialects
  ${_qlx_mlir_link_libraries}
)

qlx_propagate_llvm_rtti_eh(qlx-pauli-product-test)
