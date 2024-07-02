/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuakeToCodegen.h"
#include "CodeGenOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// Code generation: converts the Quake IR to Codegen IR.
//===----------------------------------------------------------------------===//

class CodeGenRAIIPattern : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp init,
                                PatternRewriter &rewriter) const override {
    Value mem = init.getTargets();
    auto alloc = mem.getDefiningOp<quake::AllocaOp>();
    if (!alloc)
      return init.emitOpError("init_state must have alloca as input");
    rewriter.replaceOpWithNewOp<cudaq::codegen::RAIIOp>(
        init, init.getType(), init.getState(),
        cast<cudaq::cc::PointerType>(init.getState().getType())
            .getElementType(),
        alloc.getType(), alloc.getSize());
    rewriter.eraseOp(alloc);
    return success();
  }
};
} // namespace

void cudaq::codegen::populateQuakeToCodegenPatterns(
    mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.insert<CodeGenRAIIPattern>(ctx);
}
