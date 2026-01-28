/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
void addAggressiveInlining(mlir::OpPassManager &pm, bool fatalCheck = false);
void registerAggressiveInliningPipeline();

void registerPhaseFoldingPipeline();
void registerUnrollingPipeline();
void registerClassicalOptimizationPipeline();
void registerMappingPipeline();
void registerToCFGPipeline();

/// This pipeline is run on every kernel decorator immediately after its
/// definition has been processed by the Python bridge. It converts the
/// `ModuleOp` to a target agnostic form which is amenable to further lowering,
/// etc. by the Python interpreter.
void createPythonAOTPipeline(mlir::OpPassManager &pm, bool autoGenRunStack);

/// Create and append the common target finalization pipeline. This pipeline is
/// run just prior to code generation for all targets and for both AOT and JIT
/// compilation. Primarily, it does a final round of IR canonicalization and
/// cleanup.
void createTargetFinalizePipeline(mlir::OpPassManager &pm);

/// Helper function for adding the `decompositon` pass as pass options of type
/// ListOption may not always be initialized properly resulting in mystery
/// crashes.
void addDecompositionPass(
    mlir::OpPassManager &pm, mlir::ArrayRef<std::string> enabledPats,
    mlir::ArrayRef<std::string> disabledPats = std::nullopt);

void registerAOTPipelines();
void registerJITPipelines();

/// Add a pass pipeline to apply the requisite passes to optimize classical
/// code. When converting to a quantum circuit, the static control program is
/// fully expanded to eliminate control flow.
/// Default values are threshold = 1024, allow break = true, and allow closed
/// interval = true.
void createClassicalOptimizationPipeline(
    mlir::OpPassManager &pm, std::optional<unsigned> threshold = std::nullopt,
    std::optional<bool> allowBreak = std::nullopt,
    std::optional<bool> allowClosedInterval = std::nullopt);

std::unique_ptr<mlir::Pass> createDelayMeasurementsPass();
std::unique_ptr<mlir::Pass> createExpandMeasurementsPass();
void addLowerToCFG(mlir::OpPassManager &pm);
std::unique_ptr<mlir::Pass> createObserveAnsatzPass(const std::vector<bool> &);
std::unique_ptr<mlir::Pass> createQuakeAddMetadata();
std::unique_ptr<mlir::Pass> createQuakeAddDeallocs();
std::unique_ptr<mlir::Pass> createQuakeSynthesizer();
std::unique_ptr<mlir::Pass>
createQuakeSynthesizer(std::string_view, const void *,
                       std::size_t startingArgIdx = 0,
                       bool sameAddressSpace = false);

std::unique_ptr<mlir::Pass>
createPySynthCallableBlockArgs(const llvm::SmallVector<llvm::StringRef> &,
                               bool removeBlockArg = false);
inline std::unique_ptr<mlir::Pass> createPySynthCallableBlockArgs() {
  return createPySynthCallableBlockArgs({}, false);
}

/// Helper function to build an argument synthesis pass. The names of the
/// functions and the substitutions text can be built as an unzipped pair of
/// lists. \p changeSemantics ought to be `false`, but defaults to `true` for
/// legacy reasons. When set to true, the function's original calling semantics
/// are erased, breaking any and all calls to that function.
std::unique_ptr<mlir::Pass>
createArgumentSynthesisPass(mlir::ArrayRef<mlir::StringRef> funcNames,
                            mlir::ArrayRef<mlir::StringRef> substitutions,
                            bool changeSemantics = true);

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
