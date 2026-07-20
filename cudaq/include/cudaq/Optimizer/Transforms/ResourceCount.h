/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Resources.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace cudaq::opt {

/// Extract resource counts from a Quake IR module using static analysis.
/// Runs ResourceCountPreprocess to count gates with qubit indices for depth.
/// Counted gates are erased from the module. Returns the accumulated counts,
/// or failure if the pass pipeline fails.
mlir::FailureOr<cudaq::Resources> countResourcesFromIR(mlir::ModuleOp module);

} // namespace cudaq::opt
