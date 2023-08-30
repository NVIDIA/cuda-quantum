/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

namespace cudaq::opt {

namespace internal {

template <typename A>
concept isOpTestFunction =
    std::is_convertible_v<A, std::function<bool(mlir::Operation &)>>;

/// The hasCharacteristic() template function recursively tests if an Op or a
/// Op's regions (e.g., cc.if, cc.loop, cc.scope) have any Op that matches a
/// particular condition as specified in the function parameter `test`.
template <typename A>
  requires isOpTestFunction<A>
bool hasCharacteristic(A &&test, mlir::Operation &op) {
  for (auto &region : op.getRegions()) {
    if (region.empty())
      continue;
    for (auto &block : region)
      for (auto &op : block)
        if (test(op) || (op.getNumRegions() && hasCharacteristic(test, op)))
          return true;
  }
  return test(op);
}

} // namespace internal

//===----------------------------------------------------------------------===//
// Some predefined recursive tests on Regions of Ops.
//===----------------------------------------------------------------------===//

/// hasQuantum recursively tests for an Op with the QuantumGate trait.
inline bool hasQuantum(mlir::Operation &op) {
  return internal::hasCharacteristic(
      [](mlir::Operation &op) { return op.hasTrait<QuantumGate>(); }, op);
}

/// hasCallOp recursively tests for an Op that has call-like semantics.
inline bool hasCallOp(mlir::Operation &op) {
  return internal::hasCharacteristic(
      [](mlir::Operation &op) { return mlir::isa<mlir::CallOpInterface>(op); },
      op);
}
inline bool hasCallOp(mlir::Operation *op) { return hasCallOp(*op); }
template <typename A>
inline bool hasCallOp(A &op) {
  return hasCallOp(op.getOperation());
}

/// hasMeasureOp recursively tests for the presence of measurement operations.
inline bool hasMeasureOp(mlir::Operation &op) {
  return internal::hasCharacteristic(
      [](mlir::Operation &op) {
        return mlir::isa<quake::MeasurementInterface>(op);
      },
      op);
}
inline bool hasMeasureOp(mlir::Operation *op) { return hasMeasureOp(*op); }
template <typename A>
inline bool hasMeasureOp(A &op) {
  return hasMeasureOp(op.getOperation());
}

} // namespace cudaq::opt
