/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_LOGICAL_INTERFACES_PHYSICALINTERFACES_H
#define CUDAQ_LOGICAL_INTERFACES_PHYSICALINTERFACES_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "CUDAQLogical/Interfaces/PhysicalInterfaces.h.inc"

namespace cudaq::logical {

/// Verify that `operation` implements the physical-measurement contract and
/// that every value reported by the interface belongs to the operation. This
/// checks interface integrity only; target legality remains a separate
/// provider-owned decision.
mlir::LogicalResult
verifyPhysicalMeasurementContract(mlir::Operation *operation);

} // namespace cudaq::logical

#endif // CUDAQ_LOGICAL_INTERFACES_PHYSICALINTERFACES_H
