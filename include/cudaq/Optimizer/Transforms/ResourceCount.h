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

struct ResourceCountResult {
  cudaq::Resources counts;
  bool fullyStatic; ///< True if all gates were pre-counted (no dynamic ops).
};

/// Extract resource counts from a Quake IR module using static analysis.
/// Runs ResourceCountPreprocess to count gates with qubit indices for depth.
/// Counted gates are erased from the module. Returns the counts and whether
/// the circuit is fully static (no dynamic quantum ops remain).
mlir::FailureOr<ResourceCountResult>
countResourcesFromIR(mlir::ModuleOp module);

} // namespace cudaq::opt
