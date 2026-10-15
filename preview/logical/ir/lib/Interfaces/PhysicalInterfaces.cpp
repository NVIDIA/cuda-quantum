/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CUDAQLogical/Interfaces/PhysicalInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace cudaq::logical;

LogicalResult
cudaq::logical::verifyPhysicalMeasurementContract(Operation *operation) {
  auto measurement = dyn_cast<PhysicalMeasurementOpInterface>(operation);
  if (!measurement)
    return operation->emitOpError(
        "must implement cudaq::logical::PhysicalMeasurementOpInterface");

  SmallVector<Value> inputs = measurement.getPhysicalMeasurementInputs();
  SmallVector<Value> outputs = measurement.getPhysicalMeasurementOutputs();
  Value record = measurement.getPhysicalMeasurementRecord();
  StringAttr recordId = measurement.getPhysicalMeasurementRecordId();

  if (inputs.empty())
    return operation->emitOpError(
        "physical measurement interface requires at least one state input");
  if (!record)
    return operation->emitOpError(
        "physical measurement interface requires one record result");
  if (!recordId || recordId.getValue().empty())
    return operation->emitOpError(
        "physical measurement interface requires a nonempty record identity");

  llvm::SmallDenseSet<Value, 4> uniqueInputs;
  for (Value input : inputs) {
    if (!llvm::is_contained(operation->getOperands(), input))
      return operation->emitOpError(
          "physical measurement interface reported a state input that is not "
          "an operation operand");
    if (!uniqueInputs.insert(input).second)
      return operation->emitOpError(
          "physical measurement interface reported a duplicate state input");
  }

  llvm::SmallDenseSet<Value, 4> uniqueOutputs;
  for (Value output : outputs) {
    if (!llvm::is_contained(operation->getResults(), output))
      return operation->emitOpError(
          "physical measurement interface reported a successor state that is "
          "not an operation result");
    if (!uniqueOutputs.insert(output).second)
      return operation->emitOpError(
          "physical measurement interface reported a duplicate successor "
          "state");
  }
  if (!llvm::is_contained(operation->getResults(), record))
    return operation->emitOpError(
        "physical measurement interface reported a record that is not an "
        "operation result");
  if (llvm::is_contained(outputs, record))
    return operation->emitOpError(
        "physical measurement record cannot also be a successor state");
  return success();
}

#include "CUDAQLogical/Interfaces/PhysicalInterfaces.cpp.inc"
