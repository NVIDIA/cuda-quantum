/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_EXPANDEVINCEVEQS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "expand-evince-veqs"

using namespace mlir;

namespace {
// quake.evince %veq, %r : (!quake.veq<n>, !quake.ref) -> ()
// ───────────────────────────────────────────────────────────────────
// %0 = quake.extract_ref %veq[0] : (!quake.veq<n>) -> !quake.ref
// ...
// %n = quake.extract_ref %veq[n-1] : (!quake.veq<n>) -> !quake.ref
// quake.evince %0, ..., %n, %r : (!quake.ref, ..., !quake.ref,
//     !quake.ref) -> ()
//
class ExpandEvincePattern
    : public OpRewritePattern<cudaq::quake::EvinceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::EvinceOp evin,
                                PatternRewriter &rewriter) const override {
    auto loc = evin.getLoc();
    SmallVector<Value> newArgs;
    bool didExpand = false;
    for (Value arg : evin.getArgs()) {
      auto size = cudaq::quake::getVeqSize(arg);
      if (!size) {
        newArgs.push_back(arg);
        continue;
      }

      // extract_ref requires the sized source of a relaxed vector.
      Value vector = arg;
      if (auto relax = arg.getDefiningOp<cudaq::quake::RelaxSizeOp>())
        vector = relax.getInputVec();
      for (std::size_t i = 0; i < *size; ++i)
        newArgs.push_back(
            cudaq::quake::ExtractRefOp::create(rewriter, loc, vector, i));
      didExpand = true;
    }
    if (!didExpand)
      return failure();

    rewriter.replaceOpWithNewOp<cudaq::quake::EvinceOp>(
        evin, newArgs, evin.getCompilerGenerated());
    return success();
  }
};

struct ExpandEvinceVeqsPass
    : public cudaq::opt::impl::ExpandEvinceVeqsBase<ExpandEvinceVeqsPass> {
  using ExpandEvinceVeqsBase::ExpandEvinceVeqsBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<ExpandEvincePattern>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<cudaq::quake::QuakeDialect>();
    target.addDynamicallyLegalOp<cudaq::quake::EvinceOp>(
        [](cudaq::quake::EvinceOp evin) {
          return llvm::none_of(evin.getArgs(), [](Value arg) {
            return cudaq::quake::getVeqSize(arg).has_value();
          });
        });
    if (failed(applyPartialConversion(func.getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
