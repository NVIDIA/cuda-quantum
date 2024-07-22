/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
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
inline bool isMeasureOp(Operation *op) {
  return dyn_cast<quake::MxOp>(*op) || dyn_cast<quake::MyOp>(*op) ||
         dyn_cast<quake::MzOp>(*op);
}

inline bool hasClassicalInput(Operation *op) {
  return dyn_cast<quake::RxOp>(*op) || dyn_cast<quake::RyOp>(*op) ||
         dyn_cast<quake::RzOp>(*op);
}

inline bool isBeginOp(Operation *op) {
  return dyn_cast<quake::UnwrapOp>(*op) || dyn_cast<quake::ExtractRefOp>(*op) ||
         dyn_cast<quake::NullWireOp>(*op);
}

class NullWirePat : public OpRewritePattern<quake::NullWireOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  using Base = OpRewritePattern<quake::NullWireOp>;

  unsigned *counter;

  NullWirePat(MLIRContext *context, unsigned *c)
      : OpRewritePattern<quake::NullWireOp>(context), counter(c) {}

  LogicalResult matchAndRewrite(quake::NullWireOp alloc,
                                PatternRewriter &rewriter) const override {
    if (alloc->hasAttr("qid"))
      return failure();

    auto qid = (*counter)++;

    rewriter.startRootUpdate(alloc);
    alloc->setAttr("qid", rewriter.getUI32IntegerAttr(qid));
    rewriter.finalizeRootUpdate(alloc);

    return success();
  }
};

std::optional<uint> findQid(Value v) {
  auto defop = v.getDefiningOp();
  if (!defop)
    return std::nullopt;

  if (!isa<quake::WireType>(v.getType()))
    return std::nullopt;

  if (!quake::isLinearValueForm(defop))
    defop->emitOpError("assign-ids requires operations to be in value form");

  if (isBeginOp(defop)) {
    assert(defop->hasAttr("qid") && "qid not present for beginOp");
    uint qid = defop->getAttr("qid").cast<IntegerAttr>().getUInt();
    return std::optional<uint>(qid);
  }

  // Figure out matching operand
  size_t i = 0;
  for (; i < defop->getNumResults(); i++)
    if (defop->getResult(i) == v)
      break;

  // Special cases where result # != operand #
  if (isMeasureOp(defop))
    i = 0;
  else if (hasClassicalInput(defop))
    i++;
  else if (auto ccif = dyn_cast<cudaq::cc::IfOp>(defop))
    i++;

  return findQid(defop->getOperand(i));
}

class SinkOpPat : public OpRewritePattern<quake::SinkOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  using Base = OpRewritePattern<quake::SinkOp>;

  SinkOpPat(MLIRContext *context) : OpRewritePattern<quake::SinkOp>(context) {}

  LogicalResult matchAndRewrite(quake::SinkOp release,
                                PatternRewriter &rewriter) const override {
    auto qid = findQid(release.getOperand());

    if (!qid.has_value())
      release->emitOpError(
          "Corresponding null_wire not found for sink, illegal ops present");

    rewriter.startRootUpdate(release);
    release->setAttr("qid", rewriter.getUI32IntegerAttr(qid.value()));
    rewriter.finalizeRootUpdate(release);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct AssignIDsPass : public cudaq::opt::impl::AssignIDsBase<AssignIDsPass> {
  using AssignIDsBase::AssignIDsBase;

  void runOnOperation() override {
    auto func = getOperation();

    // Blocks will cause problems for assign-ids, ensure there's only one
    if (func.getBlocks().size() != 1) {
      func.emitOpError("multiple blocks not currently supported in assign-ids");
      signalPassFailure();
      return;
    }

    assign();
  }

  void assign() {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    unsigned x = 0;
    patterns.insert<NullWirePat>(ctx, &x);
    patterns.insert<SinkOpPat>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addDynamicallyLegalOp<quake::NullWireOp>(
        [&](quake::NullWireOp alloc) { return alloc->hasAttr("qid"); });
    target.addDynamicallyLegalOp<quake::SinkOp>(
        [&](quake::SinkOp sink) { return sink->hasAttr("qid"); });
    if (failed(applyPartialConversion(func.getOperation(), target,
                                      std::move(patterns)))) {
      func.emitOpError("assigning qids failed");
      signalPassFailure();
    }
  }
};

} // namespace
