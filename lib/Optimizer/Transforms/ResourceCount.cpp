/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Transforms/ResourceCount.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

mlir::FailureOr<cudaq::Resources>
cudaq::opt::countResourcesFromIR(ModuleOp module) {
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
  pm.addNestedPass<func::FuncOp>(createResourceCountPreprocess(opt));
  pm.addPass(createCanonicalizerPass());
  auto pmResult = pm.run(module);
  if (wasThreadingEnabled)
    ctx->enableMultithreading();
  if (failed(pmResult))
    return failure();

  // Count allocated qubits from the IR.
  std::size_t allocated = 0;
  bool unresolvedVeq = false;
  module.walk([&](quake::AllocaOp alloc) {
    if (isa<quake::RefType>(alloc.getType())) {
      allocated++;
    } else if (auto size = quake::getVeqSize(alloc.getResult())) {
      allocated += *size;
    } else {
      unresolvedVeq = true;
    }
  });
  if (unresolvedVeq)
    return failure();
  counts.setNumQubits(allocated);

  return counts;
}
