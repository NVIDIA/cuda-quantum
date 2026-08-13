/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/IR/Value.h"
#include <optional>

namespace cudaq::opt {

enum class ScalarWireTraversalDirection { Forward, Backward };

/// One exact use-def step along a scalar wire. `wire` is the value reached in
/// the selected direction. A direct step retains the input wire, while a scope
/// step crosses to the matching wire on the other side of the scope.
/// `continueOperand` is non-null only for the latter.
struct ScalarWireStep {
  mlir::Value wire;
  mlir::Operation *operation;
  mlir::Block *block;
  mlir::OpOperand *continueOperand = nullptr;
};

/// Follow one exact scalar-wire step in either direction. Direct steps follow
/// the sole use or defining operation and may enter nested single-block
/// lexical scopes. Scope steps follow a `cc.continue` that forwards a scalar
/// wire to the corresponding scope result. Forks, branches, loops, calls,
/// unwinds, and every other unsupported boundary return `std::nullopt`. The
/// helper describes value flow only; callers decide whether the reached
/// operation is suitable for their analysis or rewrite.
std::optional<ScalarWireStep>
traverseScalarWire(mlir::Value wire, ScalarWireTraversalDirection direction);

} // namespace cudaq::opt
