/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Dialect/Common/Ops.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeRange.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

LogicalResult qtx::AllocaOp::verify() {
  auto resultType = dyn_cast<qtx::WireArrayType>(getWireOrArray().getType());
  if (resultType && resultType.getDead() > 0)
    return emitOpError("must not return a wire array with dead wires");
  return success();
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

LogicalResult qtx::DeallocOp::verify() {
  for (auto wireOrArray : getTargets()) {
    auto arrayType = dyn_cast<qtx::WireArrayType>(wireOrArray.getType());
    if (arrayType && arrayType.getDead() > 0)
      return emitOpError("must not deallocate a wire array with dead wires");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

LogicalResult qtx::ArrayCreateOp::verify() {
  if (getWires().size() == 0)
    return emitOpError("must be provided at least one wire as input");
  auto resultType = dyn_cast<qtx::WireArrayType>(getArray().getType());
  if (resultType.getSize() != getWires().size())
    return emitOpError("must return a wire array of size ")
           << getWires().size() << " (!qtx.wire_array<" << getWires().size()
           << ">) in this case.  There is a mismatch between the number of "
              "wires provided and the size of the returned array ("
           << getWires().size() << " != " << resultType.getSize() << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// ArraySplitOp
//===----------------------------------------------------------------------===//

LogicalResult qtx::ArraySplitOp::verify() {
  auto arrayType = dyn_cast<qtx::WireArrayType>(getArray().getType());
  if (arrayType.getDead() > 0)
    return emitOpError("must not split a wire array with dead wires");
  if (arrayType.getSize() != getWires().size())
    return emitOpError("must return a list with ")
           << arrayType.getSize() << " wires instead of " << getWires().size()
           << ". Otherwise, there will be a mismatch between the array size "
              "and "
              "the number of returned wires";
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayBorrowOp
//===----------------------------------------------------------------------===//

LogicalResult qtx::ArrayBorrowOp::verify() {
  auto arrayType = dyn_cast<qtx::WireArrayType>(getArray().getType());
  size_t numAliveWires = arrayType.getSize() - arrayType.getDead();
  size_t numRequestedWires = getIndices().size();
  size_t numResultWires = getWires().size();
  if (numAliveWires == 0)
    return emitOpError("cannot borrow from an array of dead wire(s)");
  if (numRequestedWires > numAliveWires)
    return emitOpError("cannot borrow ")
           << numRequestedWires << " wire(s) from an array that only has "
           << numAliveWires << " wire(s) alive";
  if (numRequestedWires != numResultWires)
    return emitOpError("must return a list with ")
           << numRequestedWires << " wire(s) instead of " << numResultWires
           << ". Otherwise, there will be a mismatch between the number of "
              "requested wires and the number of returned wires";
  auto newArrayType = dyn_cast<qtx::WireArrayType>(getNewArray().getType());
  size_t requiredDead = arrayType.getDead() + numRequestedWires;
  if (newArrayType.getSize() != arrayType.getSize())
    return emitOpError(
        "must return an array with same size of the input array");
  if (newArrayType.getDead() != requiredDead)
    return emitOpError("must return an array with ")
           << requiredDead << " dead wire(s)";
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayYieldOp
//===----------------------------------------------------------------------===//

LogicalResult qtx::ArrayYieldOp::verify() {
  auto arrayType = dyn_cast<qtx::WireArrayType>(getArray().getType());
  size_t numDeadWires = arrayType.getDead();
  size_t numWires = getWires().size();
  if (numDeadWires == 0)
    return emitOpError("cannot yield a wire back to an array of alive wire(s)");
  if (numWires > numDeadWires)
    return emitOpError("cannot yield ")
           << numWires << " wires back to an array that only has "
           << numDeadWires << " dead wire(s)";
  auto newArrayType = dyn_cast<qtx::WireArrayType>(getNewArray().getType());
  size_t requiredDead = arrayType.getDead() - numWires;
  if (newArrayType.getSize() != arrayType.getSize())
    return emitOpError(
        "must return an array with same size of the input array");
  if (newArrayType.getDead() != requiredDead)
    return emitOpError("must return an array with ")
           << requiredDead << " dead wire(s)";
  return success();
}

//===----------------------------------------------------------------------===//
// Measurements (MxOp, MyOp, MzOp)
//===----------------------------------------------------------------------===//

// Common verification for measurement operations.
static LogicalResult verifyMeasurements(Operation *const op,
                                        TypeRange targetsType,
                                        const Type bitsType) {
  unsigned size = 0u;
  for (Type type : targetsType) {
    if (auto arrayType = type.dyn_cast_or_null<qtx::WireArrayType>()) {
      if (arrayType.getDead())
        return op->emitOpError("cannot measure an array with dead wires");
      size += arrayType.getSize();
      continue;
    }
    size += 1;
  }
  auto vectorType = bitsType.dyn_cast_or_null<VectorType>();
  // bitsType is either I1 or a vector of I1 (operation constraint)
  auto bitsSize = vectorType ? vectorType.getNumElements() : 1;
  if (size > 1 && size != bitsSize)
    return op->emitOpError("must return a vector capable of holding all wires "
                           "being measured, i.e., ")
           << "vector<" << size << "xi1>";
  if (size != bitsSize)
    return op->emitOpError("must return either a `i1` or `vector<1xi1>` , when "
                           "measuring a wire or an array with one wire ");
  return success();
}

LogicalResult qtx::MxOp::verify() {
  return verifyMeasurements(getOperation(), getTargets().getType(),
                            getBits().getType());
}

LogicalResult qtx::MyOp::verify() {
  return verifyMeasurements(getOperation(), getTargets().getType(),
                            getBits().getType());
}

LogicalResult qtx::MzOp::verify() {
  return verifyMeasurements(getOperation(), getTargets().getType(),
                            getBits().getType());
}

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

using namespace cudaq;

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.cpp.inc"
