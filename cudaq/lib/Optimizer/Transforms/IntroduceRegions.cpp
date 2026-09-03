/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_INTRODUCEREGIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
class IntroduceRegionsPass
    : public cudaq::opt::impl::IntroduceRegionsBase<IntroduceRegionsPass> {
  using Base = cudaq::opt::impl::IntroduceRegionsBase<IntroduceRegionsPass>;

public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (numRegions == 0) {
      mod.emitError("introduce-regions: num-regions must be > 0");
      return signalPassFailure();
    }

    // Find the first function as the insertion point.
    Operation *insertBefore = nullptr;
    for (Operation &op : *mod.getBody())
      if (isa<func::FuncOp>(op)) {
        insertBefore = &op;
        break;
      }

    OpBuilder builder(mod.getBodyRegion());
    builder.setInsertionPoint(insertBefore ? insertBefore
                                           : &mod.getBody()->back());

    for (unsigned i = 0; i < numRegions; ++i) {
      std::string name = "r" + std::to_string(i);
      if (mod.lookupSymbol<cudaq::quake::RegionOp>(name))
        continue;
      builder.create<cudaq::quake::RegionOp>(
          mod.getLoc(), builder.getStringAttr(name),
          builder.getI32IntegerAttr(static_cast<int32_t>(regionSize)));
    }
  }
};
} // namespace

namespace cudaq::opt {
std::unique_ptr<mlir::Pass> createIntroduceRegionsPass() {
  return std::make_unique<IntroduceRegionsPass>();
}
} // namespace cudaq::opt
