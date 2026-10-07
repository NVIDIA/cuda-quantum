/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "test_ftqc/TestFTQCDialect.h"
#include "test_ftqc/TestFTQCOps.h"

#include "CUDAQLogical/Interfaces/PhysicalInterfaces.h"

using namespace mlir;
using namespace test_ftqc;

#include "test_ftqc/TestFTQCDialect.cpp.inc"

#define GET_OP_CLASSES
#include "test_ftqc/TestFTQCOps.cpp.inc"

static bool hasTestMode(MeasureRoundOp operation, llvm::StringRef mode) {
  auto value = operation.getTestModeAttr();
  return value && value.getValue() == mode;
}

SmallVector<Value> MeasureRoundOp::getPhysicalMeasurementInputs() {
  if (hasTestMode(*this, "duplicate_input"))
    return {getInput(), getInput()};
  return {getInput()};
}

SmallVector<Value> MeasureRoundOp::getPhysicalMeasurementOutputs() {
  if (hasTestMode(*this, "omit_output"))
    return {};
  if (hasTestMode(*this, "duplicate_output"))
    return {getOutput(), getOutput()};
  if (hasTestMode(*this, "record_as_output"))
    return {getRecord()};
  return {getOutput()};
}

Value MeasureRoundOp::getPhysicalMeasurementRecord() { return getRecord(); }

StringAttr MeasureRoundOp::getPhysicalMeasurementRecordId() {
  return getRecordIdAttr();
}

LogicalResult MeasureRoundOp::verify() {
  return cudaq::logical::verifyPhysicalMeasurementContract(getOperation());
}

SmallVector<Value> MeasureWithPayloadOp::getPhysicalMeasurementInputs() {
  return {getInput()};
}

SmallVector<Value> MeasureWithPayloadOp::getPhysicalMeasurementOutputs() {
  return {getOutput()};
}

Value MeasureWithPayloadOp::getPhysicalMeasurementRecord() {
  return getRecord();
}

StringAttr MeasureWithPayloadOp::getPhysicalMeasurementRecordId() {
  return getRecordIdAttr();
}

LogicalResult MeasureWithPayloadOp::verify() {
  return cudaq::logical::verifyPhysicalMeasurementContract(getOperation());
}

SmallVector<Value> MeasureWithExtraRecordOp::getPhysicalMeasurementInputs() {
  return {getInput()};
}

SmallVector<Value> MeasureWithExtraRecordOp::getPhysicalMeasurementOutputs() {
  return {getOutput()};
}

Value MeasureWithExtraRecordOp::getPhysicalMeasurementRecord() {
  return getRecord();
}

StringAttr MeasureWithExtraRecordOp::getPhysicalMeasurementRecordId() {
  return getRecordIdAttr();
}

LogicalResult MeasureWithExtraRecordOp::verify() {
  return cudaq::logical::verifyPhysicalMeasurementContract(getOperation());
}

void TestFTQCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "test_ftqc/TestFTQCOps.cpp.inc"
      >();
}
