/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

/// Translate a static Quake circuit to the LQCloud circuit JSON format.
mlir::LogicalResult translateToLQCloudJson(mlir::Operation *op,
                                           llvm::raw_ostream &os);

} // namespace cudaq
