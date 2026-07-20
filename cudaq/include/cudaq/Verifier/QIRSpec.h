/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class Module;
}

namespace cudaq::verifier {

struct LLVMVerifierOptions {
  bool isBaseProfile : 1;
  bool isAdaptiveProfile : 1;
  bool allowAllInstructions : 1;
  bool integerComputations : 1;
  bool floatComputations : 1;
};

/// Verify that only LLVM instructions allowed by the QIR specification.
mlir::LogicalResult verifyLLVMInstructions(llvm::Module *llvmModule,
                                           LLVMVerifierOptions options);
} // namespace cudaq::verifier
