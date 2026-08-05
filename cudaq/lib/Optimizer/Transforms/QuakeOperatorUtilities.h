/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include <cassert>

namespace cudaq::opt {

template <typename Op>
inline llvm::SmallVector<bool> getControlPolarities(Op op) {
  llvm::SmallVector<bool> polarities(op.getControls().size(), false);
  if (auto negated = op.getNegatedQubitControls())
    for (auto [index, value] : llvm::enumerate(*negated))
      polarities[index] = value;
  return polarities;
}

/// Return the wire result types for a Quake operator with the given controls
/// and targets. Quake orders wire results by controls first, then targets.
inline llvm::SmallVector<mlir::Type>
getWireResultTypes(mlir::OpBuilder &builder, mlir::ValueRange controls,
                   mlir::ValueRange targets) {
  auto wireType = cudaq::quake::WireType::get(builder.getContext());
  llvm::SmallVector<mlir::Type> resultTypes;
  for (mlir::Value control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      resultTypes.push_back(wireType);
  for (mlir::Value target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      resultTypes.push_back(wireType);
  return resultTypes;
}

/// Update controls and targets to the corresponding wire results of a newly
/// created Quake operator.
template <typename Op>
inline void threadWireResults(Op op,
                              llvm::MutableArrayRef<mlir::Value> controls,
                              llvm::MutableArrayRef<mlir::Value> targets) {
  unsigned result = 0;
  for (mlir::Value &control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      control = op.getWires()[result++];
  for (mlir::Value &target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      target = op.getWires()[result++];
  assert(result == op.getWires().size() &&
         "gate result count does not match its wire operands");
}

/// Collect threaded values in Quake's wire-result order.
inline llvm::SmallVector<mlir::Value> getWireValues(mlir::ValueRange controls,
                                                    mlir::ValueRange targets) {
  llvm::SmallVector<mlir::Value> values;
  for (mlir::Value control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      values.push_back(control);
  for (mlir::Value target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      values.push_back(target);
  return values;
}

} // namespace cudaq::opt
