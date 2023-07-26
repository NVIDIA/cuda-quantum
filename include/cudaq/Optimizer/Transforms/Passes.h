/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// The OptTransforms library includes passes that transform MLIR in some way.
// These transforms can generally be thought of as "optimizations" or "rewrites"
// on the IR.

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace cudaq::opt {

/// Pass to generate the device code loading stubs.
std::unique_ptr<mlir::Pass> createGenerateKernelExecution();

/// Pass to generate the device code loading stubs.
std::unique_ptr<mlir::Pass>
createGenerateDeviceCodeLoader(bool genAsQuake = false);

/// Add a pass pipeline to transform call between kernels to direct calls that
/// do not go through the runtime layers, inline all calls, and detect if calls
/// to kernels remain in the fully inlined into entry point kernel.
void addAggressiveEarlyInlining(mlir::OpPassManager &pm);
void registerAggressiveEarlyInlining();

void registerUnrollingPipeline();

std::unique_ptr<mlir::Pass> createApplyOpSpecializationPass();
std::unique_ptr<mlir::Pass>
createApplyOpSpecializationPass(bool computeActionOpt);
std::unique_ptr<mlir::Pass> createExpandMeasurementsPass();
std::unique_ptr<mlir::Pass> createLambdaLiftingPass();
std::unique_ptr<mlir::Pass> createLowerToCFGPass();
std::unique_ptr<mlir::Pass> createQuakeAddMetadata();
std::unique_ptr<mlir::Pass> createQuakeAddDeallocs();
std::unique_ptr<mlir::Pass> createQuakeObserveAnsatzPass();
std::unique_ptr<mlir::Pass> createQuakeObserveAnsatzPass(std::vector<bool> &);
std::unique_ptr<mlir::Pass> createQuakeSynthesizer();
std::unique_ptr<mlir::Pass> createQuakeSynthesizer(std::string_view, void *);
std::unique_ptr<mlir::Pass> createRaiseToAffinePass();
std::unique_ptr<mlir::Pass> createUnwindLoweringPass();

// declarative passes
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"

/// Helper to run the memory to register pass on classical values. Does not
/// convert the quantum code to register (wire) form.
inline std::unique_ptr<mlir::Pass> createClassicalMemToReg() {
  MemToRegOptions m2rOpt = {/*classical=*/true, /*quantum=*/false};
  return createMemToReg(m2rOpt);
}

/// Helper to run the memory to register pass on quantum wires. Does not convert
/// classical code to register form.
inline std::unique_ptr<mlir::Pass> createQuantumMemToReg() {
  MemToRegOptions m2rOpt = {/*classical=*/false, /*quantum=*/true};
  return createMemToReg(m2rOpt);
}

} // namespace cudaq::opt
