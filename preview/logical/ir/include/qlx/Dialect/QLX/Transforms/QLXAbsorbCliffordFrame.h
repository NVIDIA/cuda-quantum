/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace qlx {

/// Absorb exact Clifford actions into a device-free Pauli frame while
/// retaining non-Clifford Pauli rotations for downstream architecture
/// selection. Folded repeat bodies are accepted only when their Clifford
/// frame closes at the carry boundary.
mlir::LogicalResult absorbCliffordFrame(mlir::ModuleOp module);

/// Verify the output contract of absorbCliffordFrame.
mlir::LogicalResult verifyCliffordFrameForm(mlir::ModuleOp module,
                                            std::string &error);

} // namespace qlx
