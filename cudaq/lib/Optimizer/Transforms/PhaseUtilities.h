/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "QuakeOperatorUtilities.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"

namespace cudaq::opt {

inline llvm::SmallVector<bool>
getControlPolarities(cudaq::quake::PhaseOp phase) {
  llvm::SmallVector<bool> polarities(phase.getControls().size(), false);
  if (auto negated = phase.getNegatedQubitControls())
    for (auto [index, value] : llvm::enumerate(*negated))
      polarities[index] = value;
  return polarities;
}

inline mlir::DenseBoolArrayAttr
makeNegatedControlsAttr(mlir::OpBuilder &builder,
                        llvm::ArrayRef<bool> polarities) {
  if (llvm::none_of(polarities, [](bool value) { return value; }))
    return {};
  return builder.getDenseBoolArrayAttr(polarities);
}

} // namespace cudaq::opt
