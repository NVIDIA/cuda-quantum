/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_LOGICAL_INTERFACES_PHYSICALINTERFACES_TD
#define CUDAQ_LOGICAL_INTERFACES_PHYSICALINTERFACES_TD

include "mlir/IR/OpBase.td"

def CUDAQLogical_PhysicalMeasurementOpInterface
    : OpInterface<"PhysicalMeasurementOpInterface"> {
  let cppNamespace = "::cudaq::logical";
  let description = [{
    Describes one stage-correct P3 measurement event without naming a target
    dialect or a concrete operation. The interface exposes existing SSA state
    and record values. Each successor state preserves the physical resource of
    one input state; movement or replacement requires a separate P3 operation.
    A measurement produces exactly one record. The interface does not add
    target legality, scheduling, noise, or detector semantics.
  }];
  let methods = [
    InterfaceMethod<
      "Return the physical state values consumed by the measurement.",
      "::llvm::SmallVector<::mlir::Value>",
      "getPhysicalMeasurementInputs"
    >,
    InterfaceMethod<
      "Return successor physical state values, if the measurement preserves them.",
      "::llvm::SmallVector<::mlir::Value>",
      "getPhysicalMeasurementOutputs"
    >,
    InterfaceMethod<
      "Return the classical record produced by the measurement.",
      "::mlir::Value", "getPhysicalMeasurementRecord"
    >,
    InterfaceMethod<
      "Return the stable record identity already carried by the operation.",
      "::mlir::StringAttr", "getPhysicalMeasurementRecordId"
    >
  ];
}

#endif // CUDAQ_LOGICAL_INTERFACES_PHYSICALINTERFACES_TD
