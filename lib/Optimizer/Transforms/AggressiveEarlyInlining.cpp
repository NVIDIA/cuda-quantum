/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_CONVERTTODIRECTCALLS
#define GEN_PASS_DEF_CHECKKERNELCALLS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "aggressive-early-inlining"

using namespace mlir;

static bool isIndirectFunc(llvm::StringRef funcName,
                           llvm::StringMap<llvm::StringRef> indirectMap) {
  return indirectMap.find(funcName) != indirectMap.end();
}

// Return the inverted mangled name map.
static std::optional<llvm::StringMap<llvm::StringRef>>
getConversionMap(ModuleOp module) {
  llvm::StringMap<llvm::StringRef> result;
  if (auto mangledNameMap =
          module->getAttrOfType<DictionaryAttr>("quake.mangled_name_map")) {
    for (auto namedAttr : mangledNameMap) {
      auto key = namedAttr.getName();
      auto val = namedAttr.getValue().cast<StringAttr>().getValue();
      result.insert({val, key});
    }
    return result;
  }
  return {};
}

namespace {

/// Conversion of func.call class. [TODO: This should work for the quantum
/// dialect calls and callables as well.]
class RewriteCall : public OpRewritePattern<func::CallOp> {
public:
  RewriteCall(MLIRContext *ctx, llvm::StringMap<llvm::StringRef> &indirectMap)
      : OpRewritePattern(ctx), indirectMap(indirectMap) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!isIndirectFunc(op.getCallee(), indirectMap))
      return failure();

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
class ConvertToDirectCalls
    : public cudaq::opt::impl::ConvertToDirectCallsBase<ConvertToDirectCalls> {
public:
  using ConvertToDirectCallsBase::ConvertToDirectCallsBase;

  void runOnOperation() override {
    auto op = getOperation();
    auto *ctx = &getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    if (auto indirectMapOpt = getConversionMap(module)) {
      LLVM_DEBUG(llvm::dbgs() << "Processing: " << op << '\n');
      RewritePatternSet patterns(ctx);
      patterns.insert<RewriteCall>(ctx, *indirectMapOpt);
      if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
        signalPassFailure();
    }
  }
};

/// Check that all calls to quantum kernels have been inlined.
class CheckKernelCalls
    : public cudaq::opt::impl::CheckKernelCallsBase<CheckKernelCalls> {
public:
  using CheckKernelCallsBase::CheckKernelCallsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.empty() || !func->hasAttr(cudaq::kernelAttrName))
      return;

    auto module = func->template getParentOfType<ModuleOp>();
    bool passFailed = false;
    func.walk([&](func::CallOp call) {
      auto callee = call.getCallee();
      if (auto *decl = module.lookupSymbol(callee))
        if (decl->hasAttr(cudaq::kernelAttrName)) {
          call.emitOpError("kernel call was not inlined, "
                           "possible recursion in call tree");
          passFailed = true;
        }
    });

    if (passFailed)
      signalPassFailure();
  }
};

} // namespace

static void defaultInlinerOptPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
}

/// Run the passes in the correct order.
/// 1) Convert calls between kernels to direct calls (on the QPU).
/// 2) Aggressively inline all calls.
/// 3) Detect if kernel inlining has failed and left behind calls to kernels.
/// Such a failure is most likely a sign that there is a cycle in the call
/// graph.
void cudaq::opt::addAggressiveEarlyInlining(OpPassManager &pm) {
  llvm::StringMap<OpPassManager> opPipelines;
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createConvertToDirectCalls());
  pm.addPass(createInlinerPass(opPipelines, defaultInlinerOptPipeline));
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCheckKernelCalls());
}

void cudaq::opt::registerAggressiveEarlyInlining() {
  PassPipelineRegistration<>(
      "aggressive-early-inlining",
      "Convert calls between kernels to direct calls and inline functions.",
      addAggressiveEarlyInlining);
}
