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
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace cudaq::opt {
cc::PointerType getCudaqQubitType(mlir::MLIRContext *context);
cc::StructType getCudaqQubitSpanType(mlir::MLIRContext *context);

void populateQuakeToCCPatterns(mlir::TypeConverter &typeConverter,
                               mlir::RewritePatternSet &patterns);
void setQuakeToCCLegality(mlir::ConversionTarget &target);

void populateQuakeToCCPrepPatterns(mlir::RewritePatternSet &patterns);
void setQuakeToCCPrepLegality(mlir::ConversionTarget &target);
} // namespace cudaq::opt
