/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/CommutationAwareRewrite.h"
#include "cudaq/Support/Plugin.h"

// Here is an example MLIR Pass that one can write externally and
// use via the cudaq-opt tool, with the --load-cudaq-plugin flag.
// The pass here is simple, replace Hadamard operations with S operations.

using namespace mlir;

namespace {

struct ReplaceH : public OpRewritePattern<cudaq::quake::HOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(cudaq::quake::HOp hOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cudaq::quake::SOp>(
        hOp, hOp.isAdj(), hOp.getParameters(), hOp.getControls(),
        hOp.getTargets());
    return success();
  }
};

class CustomPassPlugin
    : public PassWrapper<CustomPassPlugin, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CustomPassPlugin)

  llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

  void runOnOperation() override {
    auto circuit = getOperation();
    auto ctx = circuit.getContext();

    cudaq::opt::CommutationAwareRewriteDriver driver(*ctx);
    driver.getPatterns().add<ReplaceH>(ctx);
    if (failed(driver.run(circuit.getBody()))) {
      circuit.emitOpError("simple pass failed");
      signalPassFailure();
    }
  }
};

} // namespace

CUDAQ_REGISTER_MLIR_PASS(CustomPassPlugin)
