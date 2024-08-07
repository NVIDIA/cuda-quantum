/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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
#define GEN_PASS_DEF_WIRESTOWIRESETS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {
class NullWirePat : public OpRewritePattern<quake::NullWireOp> {
public:
  unsigned *counter;
  StringRef setName;

  NullWirePat(MLIRContext *context, unsigned *c, StringRef name)
      : OpRewritePattern<quake::NullWireOp>(context), counter(c), setName(name) {}

  LogicalResult matchAndRewrite(quake::NullWireOp alloc,
                                PatternRewriter &rewriter) const override {

    auto index = (*counter)++;
    auto wirety = quake::WireType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<quake::BorrowWireOp>(alloc, wirety, setName, index);

    return success();
  }
};

class SinkOpPat : public OpRewritePattern<quake::SinkOp> {
public:
  SinkOpPat(MLIRContext *context) : OpRewritePattern<quake::SinkOp>(context) {}

  LogicalResult matchAndRewrite(quake::SinkOp release,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<quake::ReturnWireOp>(release, release.getOperand());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct WiresToWiresetsPass : public cudaq::opt::impl::WiresToWiresetsBase<WiresToWiresetsPass> {
  using WiresToWiresetsBase::WiresToWiresetsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    StringRef setName = "wires";
    OpBuilder builder(mod.getBodyRegion());
    builder.create<quake::WireSetOp>(builder.getUnknownLoc(), setName,
                                     INT_MAX, ElementsAttr{});

    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    unsigned x = 0;
    patterns.insert<NullWirePat>(ctx, &x, setName);
    patterns.insert<SinkOpPat>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::NullWireOp>();
    target.addIllegalOp<quake::SinkOp>();
    target.addLegalOp<quake::BorrowWireOp>();
    target.addLegalOp<quake::ReturnWireOp>();
    if (failed(applyPartialConversion(mod, target,
                                      std::move(patterns)))) {
      mod->emitOpError("Converting individual wires to wireset wires failed");
      signalPassFailure();
    }
  }
};

} // namespace