/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace cudaq::verify {

/**
   Verify that the MLIR module only calls functions that are within the set of
   valid targets for NVQIR.
 */
mlir::LogicalResult checkNvqirCalls(mlir::ModuleOp module);
} // namespace cudaq::verify
