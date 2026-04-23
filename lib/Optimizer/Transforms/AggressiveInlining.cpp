/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
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

#define DEBUG_TYPE "aggressive-inlining"

using namespace mlir;

static bool isIndirectFunc(StringRef funcName,
                           llvm::StringMap<StringRef> indirectMap) {
  return indirectMap.find(funcName) != indirectMap.end();
}

// Return the inverted mangled name map.
static std::optional<llvm::StringMap<StringRef>>
getConversionMap(ModuleOp module) {
  llvm::StringMap<StringRef> result;
  if (auto mangledNameMap = module->getAttrOfType<DictionaryAttr>(
          cudaq::runtime::mangledNameMap)) {
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
  RewriteCall(MLIRContext *ctx, llvm::StringMap<StringRef> &indirectMap,
              ModuleOp m)
      : OpRewritePattern(ctx), indirectMap(indirectMap), module(m) {}

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (!isIndirectFunc(call.getCallee(), indirectMap))
      return failure();

    auto callee = call.getCallee();
    StringRef directName = indirectMap[callee];
    auto *ctx = rewriter.getContext();
    auto loc = call.getLoc();
    auto funcTy = call.getCalleeType();
    cudaq::opt::factory::getOrAddFunc(loc, directName, funcTy, module);
    rewriter.startRootUpdate(call);
    call.setCalleeAttr(SymbolRefAttr::get(ctx, directName));
    rewriter.finalizeRootUpdate(call);
    LLVM_DEBUG(llvm::dbgs() << "Rewriting " << directName << '\n');
    return success();
  }

private:
  llvm::StringMap<StringRef> &indirectMap;
  ModuleOp module;
};

/// Translate indirect calls to direct calls.
class ConvertToDirectCalls
    : public cudaq::opt::impl::ConvertToDirectCallsBase<ConvertToDirectCalls> {
public:
  using ConvertToDirectCallsBase::ConvertToDirectCallsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = &getContext();
    if (auto indirectMapOpt = getConversionMap(module)) {
      LLVM_DEBUG(llvm::dbgs() << "Processing: " << module << '\n');
      RewritePatternSet patterns(ctx);
      patterns.insert<RewriteCall>(ctx, *indirectMapOpt, module);
      if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
        signalPassFailure();
    }
  }
};

/// Check that all calls to quantum kernels have been inlined. This pass is
/// deprecated.
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
/// graph. [This check is a bad idea: this should be deferred to final codegen
/// when translating the final Quake IR.]
void cudaq::opt::addAggressiveInlining(OpPassManager &pm, bool fatalChecks) {
  llvm::StringMap<OpPassManager> opPipelines;
  pm.addPass(cudaq::opt::createConvertToDirectCalls());
  pm.addPass(createInlinerPass(opPipelines, defaultInlinerOptPipeline));
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createEraseVectorCopyCtor());
  if (fatalChecks)
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createCheckKernelCalls());
}

namespace {
struct AggressiveInliningPipelineOptions
    : public PassPipelineOptions<AggressiveInliningPipelineOptions> {
  // Running the inlining checks here defeats the compiler engineering principle
  // of having composable passes. It is therefore highly discouraged.
  PassOptions::Option<bool> runFatalChecker{
      *this, "fatal-check",
      llvm::cl::desc("run checker and produce fatal errors immediately"),
      llvm::cl::init(false)};
};
} // namespace

void cudaq::opt::registerAggressiveInliningPipeline() {
  PassPipelineRegistration<AggressiveInliningPipelineOptions>(
      "aggressive-inlining",
      "Convert calls between kernels to direct calls and inline functions.",
      [](OpPassManager &pm, const AggressiveInliningPipelineOptions &opt) {
        addAggressiveInlining(pm, opt.runFatalChecker);
      });
}
