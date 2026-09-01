//===- FabricInterfaces.td - Fabric operation interfaces ----*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef FABRIC_INTERFACES_TD
#define FABRIC_INTERFACES_TD

include "mlir/IR/OpBase.td"

def Fabric_MeasurementBundleOpInterface
    : OpInterface<"MeasurementBundleOpInterface"> {
  let cppNamespace = "::qlx::fabric";
  let description = [{
    Identifies a complete or selected partition measurement whose bit result
    retains typed patch, partition, selection, and basis provenance across
    target-specific lowering.
  }];
  let methods = [
    InterfaceMethod<
      "Return the successor patch produced by the measurement.",
      "::mlir::Value", "getMeasurementPatchOut"
    >,
    InterfaceMethod<
      "Return the measured Fabric partition.",
      "::qlx::fabric::PartitionAttr", "getMeasurementPartition"
    >,
    InterfaceMethod<
      "Return explicit measured indices, or a null attribute for the full partition.",
      "::mlir::DenseI64ArrayAttr", "getMeasurementIndices"
    >,
    InterfaceMethod<
      "Return the basis in which the selected carriers were measured.",
      "::qlx::fabric::Prep", "getMeasurementBasis"
    >
  ];
}

#endif // FABRIC_INTERFACES_TD
