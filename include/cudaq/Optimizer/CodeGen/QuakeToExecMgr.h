/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"

namespace mlir {
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace cudaq::opt {
cc::PointerType getCudaqQubitType(mlir::MLIRContext *context);
cc::StructType getCudaqQubitSpanType(mlir::MLIRContext *context);

void populateQuakeToCCPatterns(mlir::TypeConverter &typeConverter,
                               mlir::RewritePatternSet &patterns);
void populateQuakeToCCPrepPatterns(mlir::RewritePatternSet &patterns);

/// After mx/my decomposition, move measurement ops to the end of each block
/// so that basis-change gates don't trigger premature sampling flushes.
/// Measurements on qubits that are measured more than once in the block are
/// left in place to preserve mid-circuit measurement semantics.
void delayMeasurementsInBlock(mlir::Block &block);
} // namespace cudaq::opt
