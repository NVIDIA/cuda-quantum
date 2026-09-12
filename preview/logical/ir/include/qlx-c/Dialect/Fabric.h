/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_C_DIALECT_FABRIC_H
#define QLX_C_DIALECT_FABRIC_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Fabric, fabric);

//===----------------------------------------------------------------------===//
// Parameterized types
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool fabricTypeIsAPatch(MlirType type);
MLIR_CAPI_EXPORTED MlirType fabricPatchTypeGet(MlirContext ctx,
                                               MlirStringRef codeSymbol);
MLIR_CAPI_EXPORTED MlirTypeID fabricPatchTypeGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricTypeIsASyndrome(MlirType type);
MLIR_CAPI_EXPORTED MlirType fabricSyndromeTypeGet(MlirContext ctx,
                                                  MlirStringRef codeSymbol);
MLIR_CAPI_EXPORTED MlirTypeID fabricSyndromeTypeGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricTypeIsAResourceState(MlirType type);
MLIR_CAPI_EXPORTED MlirType fabricResourceStateTypeGet(MlirContext ctx,
                                                       MlirStringRef resource);
MLIR_CAPI_EXPORTED MlirTypeID fabricResourceStateTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// Simple types
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool fabricTypeIsABit(MlirType type);
MLIR_CAPI_EXPORTED MlirType fabricBitTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirTypeID fabricBitTypeGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricTypeIsAFrame(MlirType type);
MLIR_CAPI_EXPORTED MlirType fabricFrameTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirTypeID fabricFrameTypeGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricTypeIsASlot(MlirType type);
MLIR_CAPI_EXPORTED MlirType fabricSlotTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirTypeID fabricSlotTypeGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricTypeIsAMachine(MlirType type);
MLIR_CAPI_EXPORTED MlirType fabricMachineTypeGet(MlirContext ctx);
MLIR_CAPI_EXPORTED MlirTypeID fabricMachineTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// Enum attributes
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool fabricAttributeIsAPartition(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute fabricPartitionAttrGet(MlirContext ctx,
                                                        MlirStringRef value);
MLIR_CAPI_EXPORTED MlirStringRef
fabricPartitionAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID fabricPartitionAttrGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricAttributeIsARole(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute fabricRoleAttrGet(MlirContext ctx,
                                                   MlirStringRef value);
MLIR_CAPI_EXPORTED MlirStringRef fabricRoleAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID fabricRoleAttrGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricAttributeIsAPrep(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute fabricPrepAttrGet(MlirContext ctx,
                                                   MlirStringRef value);
MLIR_CAPI_EXPORTED MlirStringRef fabricPrepAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID fabricPrepAttrGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricAttributeIsAMergeBasis(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute fabricMergeBasisAttrGet(MlirContext ctx,
                                                         MlirStringRef value);
MLIR_CAPI_EXPORTED MlirStringRef
fabricMergeBasisAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID fabricMergeBasisAttrGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricAttributeIsABoundary(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute fabricBoundaryAttrGet(MlirContext ctx,
                                                       MlirStringRef value);
MLIR_CAPI_EXPORTED MlirStringRef fabricBoundaryAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID fabricBoundaryAttrGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricAttributeIsAResource(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute fabricResourceAttrGet(MlirContext ctx,
                                                       MlirStringRef value);
MLIR_CAPI_EXPORTED MlirStringRef fabricResourceAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID fabricResourceAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Composite attributes
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool fabricAttributeIsAFloorplan(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute fabricFloorplanAttrGet(MlirContext ctx,
                                                        MlirStringRef layout,
                                                        intptr_t numParams,
                                                        const int64_t *params);
MLIR_CAPI_EXPORTED MlirTypeID fabricFloorplanAttrGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricAttributeIsAFlow(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute fabricFlowAttrGet(MlirContext ctx,
                                                   MlirStringRef xTo,
                                                   MlirStringRef zTo);
MLIR_CAPI_EXPORTED MlirTypeID fabricFlowAttrGetTypeID(void);

MLIR_CAPI_EXPORTED bool fabricAttributeIsASpecOnly(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute fabricSpecOnlyAttrGet(MlirContext ctx,
                                                       MlirStringRef name);
MLIR_CAPI_EXPORTED MlirStringRef fabricSpecOnlyAttrGetName(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID fabricSpecOnlyAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // QLX_C_DIALECT_FABRIC_H
