/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

// The OptTransforms library includes passes that transform MLIR in some way.
// These transforms can generally be thought of as "optimizations" or "rewrites"
// on the IR.

#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace cudaq::opt {

/// Pass to generate the device code loading stubs.
std::unique_ptr<mlir::Pass> createGenerateKernelExecution();

/// Pass to generate the device code loading stubs.
std::unique_ptr<mlir::Pass>
createGenerateDeviceCodeLoader(bool genAsQuake = false);

/// Create an inlining pass with a nested pipeline that transforms any indirect
/// quantum calls to direct quantum calls.
std::unique_ptr<mlir::Pass> createAggressiveEarlyInlining();

/// Create the pass to convert indirect calls to direct calls.
std::unique_ptr<mlir::Pass> createConvertToDirectCalls();

void registerConvertToDirectCalls();
void registerGenerateKernelExecution();
void registerGenerateDeviceCodeLoaderPass();
void registerConversionPipelines();

std::unique_ptr<mlir::Pass> createApplyOpSpecializationPass();
std::unique_ptr<mlir::Pass>
createApplyOpSpecializationPass(bool computeActionOpt);
std::unique_ptr<mlir::Pass> createCCMemToRegPass();
std::unique_ptr<mlir::Pass> createExpandMeasurementsPass();
std::unique_ptr<mlir::Pass> createLambdaLiftingPass();
std::unique_ptr<mlir::Pass> createLoopUnrollPass();
std::unique_ptr<mlir::Pass> createLoopUnrollPass(std::size_t maxIterations);
std::unique_ptr<mlir::Pass> createLowerToCFGPass();
std::unique_ptr<mlir::Pass> createQuakeAddMetadata();
std::unique_ptr<mlir::Pass> createQuakeAddDeallocs();
std::unique_ptr<mlir::Pass> createQuakeObserveAnsatzPass();
std::unique_ptr<mlir::Pass> createQuakeObserveAnsatzPass(std::vector<bool> &);
std::unique_ptr<mlir::Pass> createQuakeSynthesizer();
std::unique_ptr<mlir::Pass> createQuakeSynthesizer(std::string_view, void *);
std::unique_ptr<mlir::Pass> createRaiseToAffinePass();
std::unique_ptr<mlir::Pass> createUnwindLoweringPass();
std::unique_ptr<mlir::Pass> createOpCancellationPass();
std::unique_ptr<mlir::Pass> createOpDecompositionPass();
std::unique_ptr<mlir::Pass> createSplitArraysPass();
std::unique_ptr<mlir::Pass> createConvertFuncToQTXPass();
std::unique_ptr<mlir::Pass> createConvertQTXToQuakePass();
std::unique_ptr<mlir::Pass> createConvertQuakeToQTXPass();

// declarative passes
#define GEN_PASS_REGISTRATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"

} // namespace cudaq::opt
