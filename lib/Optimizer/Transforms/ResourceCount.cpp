/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Transforms/ResourceCount.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

mlir::FailureOr<cudaq::Resources>
cudaq::opt::countResourcesFromIR(ModuleOp module) {
  cudaq::Resources resourceCounts;
  auto countGate =
      [&resourceCounts](std::string gate, std::vector<std::size_t> controls,
                        std::vector<std::size_t> targets, size_t count) {
        for (size_t i = 0; i < count; i++)
          resourceCounts.appendInstruction(gate, controls, targets);
      };
  ResourceCountPreprocessOptions opt{countGate};
  PassManager pm(module.getContext());
  pm.addNestedPass<func::FuncOp>(createResourceCountPreprocess(opt));
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(module)))
    return failure();
  return resourceCounts;
}
