/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace cudaq {

/// Translates the given operation to OpenQASM code. The operation, `op`,  or
/// operations in its region must be in Quake memory reference form. Also,
/// vectors of qubit references cannot be of unknown size.
mlir::LogicalResult translateToOpenQASM(mlir::Operation *op,
                                        llvm::raw_ostream &os);

} // namespace cudaq
