/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "qir-remove-measurements"

namespace cudaq::opt {
#define GEN_PASS_DEF_REMOVEMEASUREMENTS
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
class EraseMeasurements : public OpRewritePattern<LLVM::CallOp> {
public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (auto callee = call.getCallee()) {
      if (callee->equals(cudaq::opt::QIRMeasureBody) ||
          callee->equals(cudaq::opt::QIRRecordOutput)) {
        rewriter.eraseOp(call);
        return success();
      }
    }
    return failure();
  }
};

/// Remove Measurements
///
/// This pass removes measurements and the corresponding output recording calls.
/// This is needed for backends that don't support selective measurement calls.
/// For example: https://github.com/NVIDIA/cuda-quantum/issues/512
struct RemoveMeasurementsPass
    : public cudaq::opt::impl::RemoveMeasurementsBase<RemoveMeasurementsPass> {
  using RemoveMeasurementsBase::RemoveMeasurementsBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<EraseMeasurements>(context);
    LLVM_DEBUG(llvm::dbgs() << "Before measurement erasure:\n" << *op);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After measurement erasure:\n" << *op);
  }
};

} // namespace
