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
      auto val = cast<StringAttr>(namedAttr.getValue()).getValue();
      result.insert({val, key});
    }
    return result;
  }
  return {};
}

namespace {

/// Translate indirect calls to direct calls.
class ConvertToDirectCalls
    : public cudaq::opt::impl::ConvertToDirectCallsBase<ConvertToDirectCalls> {
public:
  using ConvertToDirectCallsBase::ConvertToDirectCallsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto indirectMap = [&]() -> llvm::StringMap<StringRef> {
      auto indirectMapOpt = getConversionMap(mod);
      if (indirectMapOpt)
        return *indirectMapOpt;
      return {};
    }();
    LLVM_DEBUG(llvm::dbgs() << "Processing: " << mod << '\n');
    mod.walk([&](Operation *op) {
      auto call = dyn_cast<CallOpInterface>(op);
      if (!call)
        return;

      if (!isa<SymbolUserOpInterface>(op))
        return;

      // Check that no one misguidedly attempts to add SymbolUserOpInterface to
      // these Ops.
      if (isa<quake::ApplyOp, cudaq::cc::CallCallableOp,
              cudaq::cc::CallIndirectCallableOp>(op)) {
        op->emitOpError("Internal bug was introduced.");
        return;
      }

      auto calleeAttr = cast<SymbolRefAttr>(call.getCallableForCallee());
      StringRef callee = calleeAttr.getRootReference().getValue();
      OpBuilder rewriter(op);
      // If this is an indirect call, convert it to a direct call in place.
      if (isIndirectFunc(callee, indirectMap)) {
        StringRef directName = indirectMap[callee];
        auto *ctx = rewriter.getContext();
        auto loc = call.getLoc();
        auto indirectFn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, calleeAttr);
        auto funcTy = indirectFn.getFunctionType();
        cudaq::opt::factory::getOrAddFunc(loc, directName, funcTy, mod);
        auto directAttr = FlatSymbolRefAttr::get(ctx, directName);
        call.setCalleeFromCallable(directAttr);
        LLVM_DEBUG(llvm::dbgs() << "Rewriting " << directName << '\n');
      }

      if (!isa<cudaq::cc::DeviceCallOp, cudaq::cc::NoInlineCallOp>(op)) {
        // Move the call into a scope so as to preserve any live-ranges for
        // allocated resources.
        auto loc = call.getLoc();
        auto scope = cudaq::cc::ScopeOp::create(
            rewriter, loc, call->getResultTypes(),
            [&](OpBuilder &builder, Location loc) {
              auto *clone = call->clone();
              builder.insert(clone);
              cudaq::cc::ContinueOp::create(builder, loc, clone->getResults());
            });
        LLVM_DEBUG(llvm::dbgs() << "Call moved into scope " << scope << '\n');
        op->replaceAllUsesWith(scope);
        op->erase();
      }
      return;
    });
    LLVM_DEBUG(llvm::dbgs() << "Finished: " << mod << '\n');
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

    auto mod = func->template getParentOfType<ModuleOp>();
    bool passFailed = false;
    func.walk([&](func::CallOp call) {
      auto callee = call.getCallee();
      if (auto *decl = mod.lookupSymbol(callee))
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

static void defaultInlinerOptPipeline(OpPassManager &pm) {}

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
  pm.addPass(createCanonicalizerPass());
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
