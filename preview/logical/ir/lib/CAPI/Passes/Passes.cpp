/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx-c/Passes.h"

#include "qlx/Dialect/Fabric/Transforms/Passes.h"
#include "qlx/Dialect/Phys/Transforms/Passes.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/Transforms/QLXAbsorbCliffordFrame.h"
#include "qlx/Dialect/QLX/Transforms/QLXSynthesize.h"
#include "qlx/Dialect/QLX/Transforms/QLXToPBC.h"
#include "qlx/Dialect/QLX/Transforms/QLXVerifyPBC.h"
#include "qlx/InitAllPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"

#include <string>

void qlxRegisterAllPasses(void) { qlx::registerAllQLXPasses(); }

MlirLogicalResult qlxVerifyCliffordTModule(MlirModule module) {
  return mlir::succeeded(qlx::verifyCliffordT(unwrap(module)))
             ? mlirLogicalResultSuccess()
             : mlirLogicalResultFailure();
}

MlirLogicalResult qlxAbsorbCliffordFrameModule(MlirModule module) {
  return mlir::succeeded(qlx::absorbCliffordFrame(unwrap(module)))
             ? mlirLogicalResultSuccess()
             : mlirLogicalResultFailure();
}

MlirLogicalResult qlxVerifyCliffordFrameModule(MlirModule module) {
  std::string error;
  return mlir::succeeded(qlx::verifyCliffordFrameForm(unwrap(module), error))
             ? mlirLogicalResultSuccess()
             : mlirLogicalResultFailure();
}

MlirLogicalResult qlxLowerToPBCModule(MlirModule module) {
  return mlir::succeeded(qlx::lowerToPBC(unwrap(module)))
             ? mlirLogicalResultSuccess()
             : mlirLogicalResultFailure();
}

MlirLogicalResult qlxVerifyPBCModule(MlirModule module) {
  std::string error;
  return mlir::succeeded(qlx::verifyPBCForm(unwrap(module), error))
             ? mlirLogicalResultSuccess()
             : mlirLogicalResultFailure();
}

MlirLogicalResult qlxMaterializeVerifiedAnalyticalLowerTier(
    MlirModule module, MlirStringRef root, MlirStringRef device,
    MlirStringRef staticResult, MlirStringRef analyticalResult,
    double physicalError, double failureBudget, double cycleTime,
    double scalingPrefactor, double scalingThreshold,
    int requireEstablishedDistance) {
  mlir::ModuleOp nativeModule = unwrap(module);
  auto text = [](MlirStringRef value) {
    return std::string(value.data, value.length);
  };

  qlx::fabric::FabricCountOptions countOptions;
  countOptions.rootSymbol = text(root);
  countOptions.deviceSymbol = text(device);
  countOptions.resultSymbol = text(staticResult);
  qlx::fabric::FabricEstimateAnalyticalOptions analyticalOptions;
  analyticalOptions.rootSymbol = countOptions.rootSymbol;
  analyticalOptions.countsSymbol = countOptions.resultSymbol;
  analyticalOptions.deviceSymbol = countOptions.deviceSymbol;
  analyticalOptions.resultSymbol = text(analyticalResult);
  analyticalOptions.physicalError = physicalError;
  analyticalOptions.failureBudget = failureBudget;
  analyticalOptions.cycleTime = cycleTime;
  analyticalOptions.scalingPrefactor = scalingPrefactor;
  analyticalOptions.scalingThreshold = scalingThreshold;
  analyticalOptions.requireEstablished = requireEstablishedDistance != 0;

  mlir::PassManager manager(nativeModule.getContext());
  manager.enableVerifier(false);
  manager.addPass(qlx::fabric::createFabricCount(countOptions));
  manager.addPass(qlx::fabric::createFabricEstimateAnalyticalForFreshCounts(
      analyticalOptions));
  if (mlir::failed(manager.run(nativeModule)))
    return mlirLogicalResultFailure();

  mlir::SymbolTable symbols(nativeModule);
  auto counts =
      symbols.lookup<qlx::EstimateResultOp>(countOptions.resultSymbol);
  auto analytical =
      symbols.lookup<qlx::EstimateResultOp>(analyticalOptions.resultSymbol);
  if (!counts || !analytical || mlir::failed(counts.verifyInvariantsImpl()) ||
      mlir::failed(counts.verify()) ||
      mlir::failed(analytical.verifyInvariantsImpl()) ||
      mlir::failed(analytical.verify()))
    return mlirLogicalResultFailure();
  return mlirLogicalResultSuccess();
}

MlirLogicalResult qlxEstimateScheduleJSON(MlirModule module,
                                          MlirStringRef schedule,
                                          MlirStringRef lowerTier,
                                          MlirStringCallback callback,
                                          void *userData) {
  return qlxEstimateScheduleJSONWithTermination(
      module, schedule, lowerTier, /*fullWorkload=*/0, callback, userData);
}

MlirLogicalResult qlxEstimateScheduleJSONWithTermination(
    MlirModule module, MlirStringRef schedule, MlirStringRef lowerTier,
    int fullWorkload, MlirStringCallback callback, void *userData) {
  auto result = qlx::phys::estimateScheduleJSON(
      unwrap(module), llvm::StringRef(schedule.data, schedule.length),
      llvm::StringRef(lowerTier.data, lowerTier.length), fullWorkload != 0);
  if (mlir::failed(result))
    return mlirLogicalResultFailure();
  callback(mlirStringRefCreate(result->data(), result->size()), userData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult qlxEstimateVerifiedScheduleJSON(MlirModule module,
                                                  MlirStringRef schedule,
                                                  MlirStringRef lowerTier,
                                                  MlirStringCallback callback,
                                                  void *userData) {
  return qlxEstimateVerifiedScheduleJSONWithTermination(
      module, schedule, lowerTier, /*fullWorkload=*/0, callback, userData);
}

MlirLogicalResult qlxEstimateVerifiedScheduleJSONWithTermination(
    MlirModule module, MlirStringRef schedule, MlirStringRef lowerTier,
    int fullWorkload, MlirStringCallback callback, void *userData) {
  auto result = qlx::phys::estimateVerifiedScheduleJSON(
      unwrap(module), llvm::StringRef(schedule.data, schedule.length),
      llvm::StringRef(lowerTier.data, lowerTier.length), fullWorkload != 0);
  if (mlir::failed(result))
    return mlirLogicalResultFailure();
  callback(mlirStringRefCreate(result->data(), result->size()), userData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult
qlxScheduleAndEstimateJSON(MlirModule module, MlirStringRef graph,
                           MlirStringRef schedule, MlirStringRef lowerTier,
                           MlirStringCallback callback, void *userData) {
  return qlxScheduleAndEstimateJSONWithTermination(
      module, graph, schedule, lowerTier, /*fullWorkload=*/0, callback,
      userData);
}

MlirLogicalResult qlxScheduleAndEstimateJSONWithTermination(
    MlirModule module, MlirStringRef graph, MlirStringRef schedule,
    MlirStringRef lowerTier, int fullWorkload, MlirStringCallback callback,
    void *userData) {
  auto result = qlx::phys::scheduleAndEstimateJSON(
      unwrap(module), llvm::StringRef(graph.data, graph.length),
      llvm::StringRef(schedule.data, schedule.length),
      llvm::StringRef(lowerTier.data, lowerTier.length), fullWorkload != 0);
  if (mlir::failed(result))
    return mlirLogicalResultFailure();
  callback(mlirStringRefCreate(result->data(), result->size()), userData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult qlxScheduleVerifiedAndEstimateJSON(
    MlirModule module, MlirStringRef graph, MlirStringRef schedule,
    MlirStringRef lowerTier, MlirStringCallback callback, void *userData) {
  return qlxScheduleVerifiedAndEstimateJSONWithTermination(
      module, graph, schedule, lowerTier, /*fullWorkload=*/0, callback,
      userData);
}

MlirLogicalResult qlxScheduleVerifiedAndEstimateJSONWithTermination(
    MlirModule module, MlirStringRef graph, MlirStringRef schedule,
    MlirStringRef lowerTier, int fullWorkload, MlirStringCallback callback,
    void *userData) {
  auto result = qlx::phys::scheduleVerifiedAndEstimateJSON(
      unwrap(module), llvm::StringRef(graph.data, graph.length),
      llvm::StringRef(schedule.data, schedule.length),
      llvm::StringRef(lowerTier.data, lowerTier.length), fullWorkload != 0);
  if (mlir::failed(result))
    return mlirLogicalResultFailure();
  callback(mlirStringRefCreate(result->data(), result->size()), userData);
  return mlirLogicalResultSuccess();
}
