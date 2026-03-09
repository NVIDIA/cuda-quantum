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

namespace cudaq::verifier {

/// Verify some QIR constraints when the IR is lowered to LLVM-IR dialect.
mlir::LogicalResult checkQIRLLVMIRDialect(mlir::ModuleOp module,
                                          mlir::StringRef profile);
} // namespace cudaq::verifier
