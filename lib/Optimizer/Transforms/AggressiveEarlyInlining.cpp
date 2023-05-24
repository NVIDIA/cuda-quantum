/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "aggressive-early-inlining"

using namespace mlir;

namespace {

/// Conversion of func.call class. [TODO: This should work for the quantum
/// dialect calls and callables as well.]
class RewriteCall : public OpRewritePattern<func::CallOp> {
public:
  RewriteCall(MLIRContext *ctx, llvm::StringMap<llvm::StringRef> &indirectMap)
      : OpRewritePattern(ctx), indirectMap(indirectMap) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    auto callee = op.getCallee();
    llvm::StringRef directName = indirectMap[callee];
    op.setCalleeAttr(SymbolRefAttr::get(op.getContext(), directName));
    LLVM_DEBUG(llvm::dbgs() << "Rewriting " << directName << '\n');
    rewriter.finalizeRootUpdate(op);
    return success();
  }

private:
  llvm::StringMap<llvm::StringRef> &indirectMap;
};

/// Translate indirect calls to direct calls.
class ConvertToDirectCallsPass
    : public cudaq::opt::ConvertToDirectCallsBase<ConvertToDirectCallsPass> {
public:
  ConvertToDirectCallsPass() = default;

  void runOnOperation() override {
    auto op = getOperation();
    auto *ctx = &getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    auto indirectMap = getConversionMap(module);

    LLVM_DEBUG(llvm::dbgs() << "Processing: " << op << '\n');
    RewritePatternSet patterns(ctx);
    patterns.insert<RewriteCall>(ctx, indirectMap);
    ConversionTarget target(*ctx);

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return !isIndirectFunc(op.getCallee(), indirectMap);
    });

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }

  static bool isIndirectFunc(llvm::StringRef funcName,
                             llvm::StringMap<llvm::StringRef> indirectMap) {
    return indirectMap.find(funcName) != indirectMap.end();
  }

  // Return the inverted mangled name map.
  static llvm::StringMap<llvm::StringRef> getConversionMap(ModuleOp module) {
    llvm::StringMap<llvm::StringRef> result;
    auto mangledNameMap =
        module->getAttrOfType<DictionaryAttr>("quake.mangled_name_map");
    for (auto namedAttr : mangledNameMap) {
      auto key = namedAttr.getName();
      auto val = namedAttr.getValue().cast<StringAttr>().getValue();
      result.insert({val, key});
    }
    return result;
  }
};

} // namespace

std::unique_ptr<Pass> cudaq::opt::createConvertToDirectCalls() {
  return std::make_unique<ConvertToDirectCallsPass>();
}

static void defaultInlinerOptPipeline(OpPassManager &pm) {
  pm.addPass(cudaq::opt::createConvertToDirectCalls());
}

std::unique_ptr<Pass> cudaq::opt::createAggressiveEarlyInlining() {
  llvm::StringMap<OpPassManager> opPipelines;
  return createInlinerPass(opPipelines, defaultInlinerOptPipeline);
}
