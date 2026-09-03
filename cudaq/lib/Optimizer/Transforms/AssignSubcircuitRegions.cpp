/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ASSIGNSUBCIRCUITREGIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
class AssignSubcircuitRegionsPass
    : public cudaq::opt::impl::AssignSubcircuitRegionsBase<
          AssignSubcircuitRegionsPass> {
  using Base =
      cudaq::opt::impl::AssignSubcircuitRegionsBase<AssignSubcircuitRegionsPass>;

public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    ModuleOp mod = func->getParentOfType<ModuleOp>();
    if (!mod)
      return;

    // Collect region names from quake.region declarations in the module.
    SmallVector<StringRef> regionNames;
    mod.walk([&](cudaq::quake::RegionOp regionOp) {
      regionNames.push_back(regionOp.getSymName());
    });
    if (regionNames.empty())
      return;

    // Collect lambdas in program order and assign regions round-robin.
    MLIRContext *ctx = func.getContext();
    auto attrName = StringAttr::get(ctx, "region");
    unsigned idx = 0;
    func.walk([&](cudaq::cc::CreateLambdaOp lambda) {
      lambda->setAttr(attrName, FlatSymbolRefAttr::get(
                                    ctx, regionNames[idx % regionNames.size()]));
      ++idx;
    });
  }
};
} // namespace

namespace cudaq::opt {
std::unique_ptr<mlir::Pass> createAssignSubcircuitRegionsPass() {
  return std::make_unique<AssignSubcircuitRegionsPass>();
}
} // namespace cudaq::opt
