/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/ResourceCount.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

mlir::FailureOr<cudaq::Resources>
cudaq::opt::countResourcesFromIR(ModuleOp module) {
  // Check upfront whether all qubit allocations have statically known sizes.
  // If any veq has a dynamic size we cannot count qubits statically, so bail
  // out before running the gate-erasing pass manager.
  std::size_t allocated = 0;
  bool unresolvedVeq = false;
  module.walk([&](cudaq::quake::AllocaOp alloc) {
    if (isa<cudaq::quake::RefType>(alloc.getType())) {
      allocated++;
    } else if (auto size = cudaq::quake::getVeqSize(alloc.getResult())) {
      allocated += *size;
    } else {
      unresolvedVeq = true;
    }
  });
  if (unresolvedVeq)
    return failure();

  // All qubit sizes are statically known — proceed to count gates and erase
  // them from the IR so the subsequent JIT compiles a near-empty module.
  cudaq::Resources counts;
  auto countGate = [&counts](std::string gate,
                             std::vector<std::size_t> controls,
                             std::vector<std::size_t> targets, size_t count) {
    for (size_t i = 0; i < count; i++)
      counts.appendInstruction(gate, controls, targets);
  };
  ResourceCountPreprocessOptions opt{countGate};
  // The countGate callback captures &counts, a shared mutable Resources.
  // createResourceCountPreprocess runs as addNestedPass<func::FuncOp>, which
  // MLIR executes in parallel across functions. Disable threading for this
  // PassManager so the callback is called sequentially.
  auto *ctx = module.getContext();
  bool wasThreadingEnabled = ctx->isMultithreadingEnabled();
  ctx->disableMultithreading();
  PassManager pm(ctx);
  // Resource counting is a terminal consumer of Quake IR. Complete the phase
  // lifecycle before counting so global phase corrections are either erased
  // (when uncontrolled) or lowered to physical gates (when controlled).
  cudaq::opt::addPhaseLifecycle(pm);
  // LowerPhase may preserve negative controls on the physical gates it emits.
  // Expand them before collecting the final gate counts.
  pm.addNestedPass<func::FuncOp>(createExpandControlNegations());
  // Keep this verifier before ResourceCountPreprocess, which erases every
  // counted operator and could otherwise hide an unlowered PhaseOp.
  pm.addPass(createVerifyNoPhase());
  pm.addNestedPass<func::FuncOp>(createResourceCountPreprocess(opt));
  pm.addPass(createCanonicalizerPass());
  auto pmResult = pm.run(module);
  if (wasThreadingEnabled)
    ctx->enableMultithreading();
  if (failed(pmResult))
    return failure();

  counts.setNumQubits(allocated);
  return counts;
}
