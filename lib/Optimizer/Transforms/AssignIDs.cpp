/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_ASSIGNIDS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {
static int allocaSize(quake::AllocaOp alloc) {
  if (isa<quake::RefType>(alloc.getType()))
    return 1;
  if (auto veqTy = dyn_cast<quake::VeqType>(alloc.getType()))
    return veqTy.getSize();
  else
    return 0;
}

class AllocaPat : public OpRewritePattern<quake::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  using Base = OpRewritePattern<quake::AllocaOp>;

  unsigned* counter;

  AllocaPat(MLIRContext* context, unsigned* c)
  : OpRewritePattern<quake::AllocaOp>(context), counter(c) {}

  LogicalResult matchAndRewrite(quake::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    auto size = allocaSize(alloc);
    if (!size)
      return success();

    rewriter.startRootUpdate(alloc);
    auto newAttr = rewriter.getIndexAttr(*counter);
    alloc->setAttr("qid_alloc", newAttr);
    rewriter.finalizeRootUpdate(alloc);

    //alloc.dump();

    for (auto *users : alloc->getUsers()) {
      if (auto ext = dyn_cast<quake::ExtractRefOp>(users)) {
        if (ext.hasConstantIndex()) {
          rewriter.startRootUpdate(ext);
          auto newAttr = rewriter.getIndexAttr(*counter + ext.getConstantIndex());
          ext->setAttr("qid_ref", newAttr);
          rewriter.finalizeRootUpdate(ext);
          //ext.dump();
        }
      }
    }

    (*counter) += size;

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct AssignIDsPass
    : public cudaq::opt::impl::AssignIDsBase<AssignIDsPass> {
  using AssignIDsBase::AssignIDsBase;

  void runOnOperation() override {
    assign();
  }

  void assign() {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    unsigned x = 0;
    patterns.insert<AllocaPat>(ctx, &x);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    //target.addLegalOp<quake::AllocaOp>();
    target.addDynamicallyLegalOp<quake::AllocaOp>([&](quake::AllocaOp alloc) {
      return allocaSize(alloc) == 0 || alloc->hasAttr("qid_alloc");
    });
    target.addDynamicallyLegalOp<quake::ExtractRefOp>([&](quake::ExtractRefOp ext) {
      return !ext.hasConstantIndex() || ext->hasAttr("qid_ref");
    });
    if (failed(applyPartialConversion(func.getOperation(), target,
                                      std::move(patterns)))) {
      func.emitOpError("factoring quantum allocations failed");
      signalPassFailure();
    }
  }
};

} // namespace
