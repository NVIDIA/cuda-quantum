/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_C_PASSES_H
#define QLX_C_PASSES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register every native QLX pass with the global pass registry.
MLIR_CAPI_EXPORTED void qlxRegisterAllPasses(void);

/// Run typed QLX/PBC transforms through the process-wide common C API image.
MLIR_CAPI_EXPORTED MlirLogicalResult
qlxVerifyCliffordTModule(MlirModule module);
MLIR_CAPI_EXPORTED MlirLogicalResult
qlxAbsorbCliffordFrameModule(MlirModule module);
MLIR_CAPI_EXPORTED MlirLogicalResult
qlxVerifyCliffordFrameModule(MlirModule module);
MLIR_CAPI_EXPORTED MlirLogicalResult qlxLowerToPBCModule(MlirModule module);
MLIR_CAPI_EXPORTED MlirLogicalResult qlxVerifyPBCModule(MlirModule module);

/// Materialize authenticated static and analytical lower-tier results in one
/// compiler-owned pass transaction. This internal entry point is for callers
/// that already hold a verified immutable Build clone.
MLIR_CAPI_EXPORTED MlirLogicalResult qlxMaterializeVerifiedAnalyticalLowerTier(
    MlirModule module, MlirStringRef root, MlirStringRef device,
    MlirStringRef staticResult, MlirStringRef analyticalResult,
    double physicalError, double failureBudget, double cycleTime,
    double scalingPrefactor, double scalingThreshold,
    int requireEstablishedDistance);

/// Derive a native Tier-3 schedule estimate without mutating the module.
/// The callback receives a small JSON evidence payload on success.
MLIR_CAPI_EXPORTED MlirLogicalResult qlxEstimateScheduleJSON(
    MlirModule module, MlirStringRef schedule, MlirStringRef lowerTier,
    MlirStringCallback callback, void *userData);

/// Termination-aware counterpart. A nonzero fullWorkload prices the complete
/// scheduled workload while retaining runtime abort evidence.
MLIR_CAPI_EXPORTED MlirLogicalResult qlxEstimateScheduleJSONWithTermination(
    MlirModule module, MlirStringRef schedule, MlirStringRef lowerTier,
    int fullWorkload, MlirStringCallback callback, void *userData);

/// Internal counterpart for a compiler-authenticated in-process module.
MLIR_CAPI_EXPORTED MlirLogicalResult qlxEstimateVerifiedScheduleJSON(
    MlirModule module, MlirStringRef schedule, MlirStringRef lowerTier,
    MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
qlxEstimateVerifiedScheduleJSONWithTermination(
    MlirModule module, MlirStringRef schedule, MlirStringRef lowerTier,
    int fullWorkload, MlirStringCallback callback, void *userData);

/// Mutate the live module with one native schedule and estimate its transient
/// typed rows without reparsing the emitted schedule representation.
MLIR_CAPI_EXPORTED MlirLogicalResult qlxScheduleAndEstimateJSON(
    MlirModule module, MlirStringRef graph, MlirStringRef schedule,
    MlirStringRef lowerTier, MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult qlxScheduleAndEstimateJSONWithTermination(
    MlirModule module, MlirStringRef graph, MlirStringRef schedule,
    MlirStringRef lowerTier, int fullWorkload, MlirStringCallback callback,
    void *userData);

/// Internal fused counterpart for a compiler-authenticated P3 module.
MLIR_CAPI_EXPORTED MlirLogicalResult qlxScheduleVerifiedAndEstimateJSON(
    MlirModule module, MlirStringRef graph, MlirStringRef schedule,
    MlirStringRef lowerTier, MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
qlxScheduleVerifiedAndEstimateJSONWithTermination(
    MlirModule module, MlirStringRef graph, MlirStringRef schedule,
    MlirStringRef lowerTier, int fullWorkload, MlirStringCallback callback,
    void *userData);

#ifdef __cplusplus
}
#endif

#endif // QLX_C_PASSES_H
