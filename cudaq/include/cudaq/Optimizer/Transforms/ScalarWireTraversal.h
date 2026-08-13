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
  mlir::OpOperand *continueOperand = nullptr;
};

/// Follows one exact scalar-wire step in either direction. It follows direct
/// def-use links and `cc.continue` forwarding through single-block lexical
/// scopes, with continuation operands mapped positionally to scope results.
/// All other boundaries return `std::nullopt`. Callers decide whether the
/// reached operation is suitable for their analysis or rewrite.
std::optional<ScalarWireStep>
traverseScalarWire(mlir::Value wire, ScalarWireTraversalDirection direction);

} // namespace cudaq::opt
