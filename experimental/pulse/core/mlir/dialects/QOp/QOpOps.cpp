// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"

// Declarations only — .cpp.inc definitions live in QOpDialect.cpp
#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

using namespace mlir;

namespace qop {

LogicalResult SpinOp::verify() {
  auto k = getKind();
  if (k != HandlerKind::SpinI && k != HandlerKind::SpinX &&
      k != HandlerKind::SpinY && k != HandlerKind::SpinZ &&
      k != HandlerKind::SpinLowering && k != HandlerKind::SpinRaising)
    return emitOpError("invalid handler kind for spin operator");
  return success();
}

LogicalResult BosonOp::verify() {
  auto k = getKind();
  if (k != HandlerKind::BosonIdentity && k != HandlerKind::BosonCreate &&
      k != HandlerKind::BosonAnnihilate && k != HandlerKind::BosonNumber)
    return emitOpError("invalid handler kind for boson operator");
  if (getDimension() < 2)
    return emitOpError("boson dimension must be >= 2, got ") << getDimension();
  return success();
}

LogicalResult FermionOp::verify() {
  auto k = getKind();
  if (k != HandlerKind::FermionIdentity && k != HandlerKind::FermionCreate &&
      k != HandlerKind::FermionAnnihilate && k != HandlerKind::FermionNumber)
    return emitOpError("invalid handler kind for fermion operator");
  return success();
}

LogicalResult MatrixLeafOp::verify() {
  if (getTargets().empty())
    return emitOpError("matrix leaf must target at least one mode");
  if (getTargets().size() != getDimensions().size())
    return emitOpError("number of targets (")
           << getTargets().size() << ") must match number of dimensions ("
           << getDimensions().size() << ")";
  return success();
}

LogicalResult CallbackScalarOp::verify() {
  if (getCallback().empty())
    return emitOpError("callback symbol name must not be empty");
  return success();
}

LogicalResult MakeSumOp::verify() {
  if (getTerms().empty())
    return emitOpError("sum must have at least one product term");
  return success();
}

LogicalResult ToMatrixOp::verify() {
  if (getDimensions().empty())
    return emitOpError("dimensions array must not be empty");
  for (int64_t d : getDimensions()) {
    if (d < 2)
      return emitOpError("each dimension must be >= 2, got ") << d;
  }
  return success();
}

LogicalResult DegreesOp::verify() { return success(); }

} // namespace qop

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.cpp.inc"
