/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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

/// Add a pass pipeline to transform call between kernels to direct calls that
/// do not go through the runtime layers, inline all calls, and detect if calls
/// to kernels remain in the fully inlined into entry point kernel.
void addAggressiveEarlyInlining(mlir::OpPassManager &pm);
void registerAggressiveEarlyInlining();

void registerUnrollingPipeline();
void registerMappingPipeline();

std::unique_ptr<mlir::Pass> createApplyOpSpecializationPass();
std::unique_ptr<mlir::Pass>
createApplyOpSpecializationPass(bool computeActionOpt);
std::unique_ptr<mlir::Pass> createDelayMeasurementsPass();
std::unique_ptr<mlir::Pass> createExpandMeasurementsPass();
std::unique_ptr<mlir::Pass> createLambdaLiftingPass();
std::unique_ptr<mlir::Pass> createLowerToCFGPass();
std::unique_ptr<mlir::Pass> createObserveAnsatzPass(const std::vector<bool> &);
std::unique_ptr<mlir::Pass> createQuakeAddMetadata();
std::unique_ptr<mlir::Pass> createQuakeAddDeallocs();
std::unique_ptr<mlir::Pass> createQuakeSynthesizer();
std::unique_ptr<mlir::Pass>
createQuakeSynthesizer(std::string_view, const void *,
                       std::size_t startingArgIdx = 0,
                       bool sameAddressSpace = false);
std::unique_ptr<mlir::Pass> createRaiseToAffinePass();
std::unique_ptr<mlir::Pass> createUnwindLoweringPass();

std::unique_ptr<mlir::Pass>
createPySynthCallableBlockArgs(const llvm::SmallVector<llvm::StringRef> &,
                               bool removeBlockArg = false);
inline std::unique_ptr<mlir::Pass> createPySynthCallableBlockArgs() {
  return createPySynthCallableBlockArgs({}, false);
}

/// Helper function to build an argument synthesis pass. The names of the
/// functions and the substitutions text can be built as an unzipped pair of
/// lists.
std::unique_ptr<mlir::Pass>
createArgumentSynthesisPass(mlir::ArrayRef<mlir::StringRef> funcNames,
                            mlir::ArrayRef<mlir::StringRef> substitutions);

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

/// Name of `quake.wire_set` generated prior to mapping
static constexpr const char topologyAgnosticWiresetName[] = "wires";

} // namespace cudaq::opt
