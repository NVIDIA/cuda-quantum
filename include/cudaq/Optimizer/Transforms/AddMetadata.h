/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
class Operation;
}

namespace quake::detail {
/// Define a type to contain the Quake Function Metadata
struct QuakeMetadata {
  bool hasConditionalsOnMeasure = false;

  // If the following flag is set, it means we've detected quantum to classical
  // back to quantum data-flow in the kernel. This could be a problem for
  // quantum hardware.
  bool hasQuantumDataflowViaClassical = false;

  // If the following flag is set, this pass was run early enough that function
  // calls have not been inlined and we have quantum computation that excapes
  // the kernel. We flag this condition pessimistically, since we may not know
  // what the called function will do.
  bool hasUnexpectedCalls = false;
};

/// We'll define a type mapping a Quake Function to its metadata
using QuakeFunctionInfo = mlir::DenseMap<mlir::Operation *, QuakeMetadata>;

/// The analysis on an a Quake function which will attach
/// metadata under certain situations.
struct QuakeFunctionAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuakeFunctionAnalysis)

  QuakeFunctionAnalysis(mlir::Operation *op) { performAnalysis(op); }
  const QuakeFunctionInfo &getAnalysisInfo() const { return infoMap; }

private:
  // Scan the body of a function for ops that will lead to the
  // addition of metadata.
  void performAnalysis(mlir::Operation *operation);

  QuakeFunctionInfo infoMap;
};
} // namespace quake::detail
