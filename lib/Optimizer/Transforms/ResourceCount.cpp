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

mlir::FailureOr<cudaq::opt::ResourceCountResult>
cudaq::opt::countResourcesFromIR(ModuleOp module) {
  ResourceCountResult result;
  auto countGate = [&result](std::string gate,
                             std::vector<std::size_t> controls,
                             std::vector<std::size_t> targets, size_t count) {
    for (size_t i = 0; i < count; i++)
      result.counts.appendInstruction(gate, controls, targets);
  };
  ResourceCountPreprocessOptions opt{countGate};
  PassManager pm(module.getContext());
  pm.addNestedPass<func::FuncOp>(createResourceCountPreprocess(opt));
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(module)))
    return failure();

  // Check if any quantum gate ops remain (dynamic gates that couldn't
  // be pre-counted). If none remain, the circuit is fully static.
  result.fullyStatic = true;
  module.walk([&](Operation *op) {
    if (dyn_cast<quake::OperatorInterface>(op) &&
        !isa<quake::MeasurementInterface>(op))
      result.fullyStatic = false;
  });

  return result;
}
