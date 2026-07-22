// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"

// Declarations only — .cpp.inc definitions live in CuDensityMatDialect.cpp
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.h.inc"

using namespace mlir;

namespace cudm {

LogicalResult CreateStateOp::verify() {
  auto extents = getModeExtents();
  if (extents.empty())
    return emitOpError("mode_extents must have at least one element");
  for (int64_t e : extents) {
    if (e < 2)
      return emitOpError("each mode extent must be >= 2, got ") << e;
  }
  return success();
}

LogicalResult CreateElementaryOpOp::verify() {
  auto extents = getModeExtents();
  if (extents.empty())
    return emitOpError("mode_extents must have at least one element");
  for (int64_t e : extents) {
    if (e < 2)
      return emitOpError("each mode extent must be >= 2, got ") << e;
  }
  return success();
}

LogicalResult AppendElementaryProductOp::verify() {
  if (getElemOps().empty())
    return emitOpError("at least one elementary operator must be provided");
  auto modes = getModesActedOn();
  auto duality = getDuality();
  if (modes.size() != getElemOps().size())
    return emitOpError("modes_acted_on size (")
           << modes.size() << ") must match number of elementary operators ("
           << getElemOps().size() << ")";
  if (duality.size() != getElemOps().size())
    return emitOpError("duality size (")
           << duality.size() << ") must match number of elementary operators ("
           << getElemOps().size() << ")";
  return success();
}

LogicalResult OperatorAppendTermOp::verify() {
  int32_t d = getDuality();
  if (d != 0 && d != 1)
    return emitOpError("duality must be 0 (bra) or 1 (ket), got ") << d;
  return success();
}

LogicalResult OperatorPrepareActionOp::verify() {
  if (getWorkspaceLimit() < 0)
    return emitOpError("workspace_limit must be non-negative, got ")
           << getWorkspaceLimit();
  return success();
}

LogicalResult OperatorComputeActionOp::verify() { return success(); }

LogicalResult ExpectationComputeOp::verify() { return success(); }

LogicalResult SSEEvolveOp::verify() {
  if (getNumTrajectories() < 1)
    return emitOpError("num_trajectories must be >= 1, got ")
           << getNumTrajectories();
  if (getNumSteps() < 1)
    return emitOpError("num_steps must be >= 1, got ") << getNumSteps();
  double tStart = getTStart().convertToDouble();
  double tEnd = getTEnd().convertToDouble();
  if (tEnd <= tStart)
    return emitOpError("t_end (")
           << tEnd << ") must be greater than t_start (" << tStart << ")";
  return success();
}

} // namespace cudm

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.cpp.inc"
