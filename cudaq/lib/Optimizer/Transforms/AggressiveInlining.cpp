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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_CONVERTTODIRECTCALLS
#define GEN_PASS_DEF_CHECKKERNELCALLS
#define GEN_PASS_DEF_VERIFYATOMICQUANTUMREGIONS
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
      if (isa<cudaq::quake::ApplyOp, cudaq::cc::CallCallableOp,
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
        calleeAttr = directAttr;
        LLVM_DEBUG(llvm::dbgs() << "Rewriting " << directName << '\n');
      }

      if (!isa<cudaq::cc::DeviceCallOp, cudaq::cc::NoInlineCallOp>(op)) {
        auto calleeFunc = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, calleeAttr);
        const bool isAtomicQuantumRegion =
            calleeFunc &&
            calleeFunc->hasAttr(cudaq::cc::atomicQuantumRegionAttrName);
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
        if (isAtomicQuantumRegion)
          scope.setAtomicQuantumRegionAttr(rewriter.getUnitAttr());
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

/// Reject qubit allocations and measurements in an atomic quantum region.
/// A region is entered in two shapes and both are checked.
/// `ConvertToDirectCalls` materializes a marked `cc.scope` at each call site of
/// a marked kernel. A marked kernel launched directly as the entry point has no
/// call site, so only the attribute on its `func.func` identifies it.
class AtomicQuantumRegionVerifier {
public:
  explicit AtomicQuantumRegionVerifier(ModuleOp module) : module(module) {}

  void verify() { verifyOperation(module, /*insideAtomicRegion=*/false); }

  bool hasViolations() const { return passFailed; }

private:
  static bool startsAtomicRegion(Operation *op) {
    if (auto function = dyn_cast<func::FuncOp>(op))
      return function->hasAttr(cudaq::cc::atomicQuantumRegionAttrName);
    if (auto scope = dyn_cast<cudaq::cc::ScopeOp>(op))
      return scope.getAtomicQuantumRegionAttr() != nullptr;
    return false;
  }

  /// Report an operation at most once, and report at most one operation per
  /// source location. Inlining copies a violating callee into every marked
  /// caller and leaves the original definition in place, so a single line of
  /// user source becomes several violating operations.
  bool shouldReport(Operation *op,
                    llvm::SmallDenseSet<Location> &reportedSourceLocations) {
    if (!reportedOperations.insert(op).second)
      return false;
    if (auto sourceLoc = op->getLoc()->findInstanceOf<FileLineColLoc>())
      return reportedSourceLocations.insert(sourceLoc).second;
    return true;
  }

  void verifyForbiddenOperation(Operation *op) {
    if (isa<cudaq::quake::MeasurementInterface>(op)) {
      if (!shouldReport(op, reportedMeasurementLocations))
        return;
      op->emitOpError(
          "measurement operations are not supported inside an atomic quantum "
          "region; measure outside the region");
      passFailed = true;
      return;
    }

    // Note: `quake.alloca` is the only qubit allocation form reachable here.
    // This pass runs on reference semantics, before the conversion that turns
    // allocations into `quake.null_wire` and `quake.borrow_wire`.
    auto allocation = dyn_cast<cudaq::quake::AllocaOp>(op);
    if (!allocation || !shouldReport(op, reportedAllocationLocations))
      return;
    allocation.emitOpError(
        "qubit allocations are not supported inside an atomic quantum region; "
        "allocate in the caller and pass the qubits as arguments");
    passFailed = true;
  }

  /// Descend into a callee that the inliner left behind so a marked kernel
  /// cannot hide a violation in an ordinary helper. `verifiedCallees` bounds
  /// the traversal: a recursive or mutually recursive `cc.noinline_call` chain
  /// would not terminate otherwise. A callee with no body is not checked. It
  /// may be defined in another translation unit, and the link step runs this
  /// pass again on the merged module.
  void verifyCallee(Operation *call, FlatSymbolRefAttr calleeAttr) {
    if (!calleeAttr)
      return;
    auto callee =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, calleeAttr);
    if (!callee || callee.isDeclaration() ||
        !verifiedCallees.insert(callee.getOperation()).second)
      return;
    verifyOperation(callee, /*insideAtomicRegion=*/true);
  }

  void verifyResidualCall(Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      verifyCallee(op, call.getCalleeAttr());
      return;
    }
    if (auto call = dyn_cast<cudaq::cc::NoInlineCallOp>(op))
      verifyCallee(op, call.getCalleeAttr());
  }

  /// Regions nest but never reopen, so `insideAtomicRegion` only ever gets set.
  void verifyOperation(Operation *op, bool insideAtomicRegion) {
    insideAtomicRegion |= startsAtomicRegion(op);
    if (insideAtomicRegion) {
      verifyForbiddenOperation(op);
      verifyResidualCall(op);
    }

    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          verifyOperation(&nested, insideAtomicRegion);
  }

  ModuleOp module;
  bool passFailed = false;
  llvm::DenseSet<Operation *> reportedOperations;
  llvm::DenseSet<Operation *> verifiedCallees;
  llvm::SmallDenseSet<Location> reportedAllocationLocations;
  llvm::SmallDenseSet<Location> reportedMeasurementLocations;
};

class VerifyAtomicQuantumRegions
    : public cudaq::opt::impl::VerifyAtomicQuantumRegionsBase<
          VerifyAtomicQuantumRegions> {
public:
  using VerifyAtomicQuantumRegionsBase::VerifyAtomicQuantumRegionsBase;

  void runOnOperation() override {
    AtomicQuantumRegionVerifier verifier(getOperation());
    verifier.verify();
    if (verifier.hasViolations())
      signalPassFailure();
  }
};

} // namespace

static void defaultInlinerOptPipeline(OpPassManager &pm) {}

/// Run the passes in the correct order.
/// 1) Optionally lower unwind control flow before creating call-site scopes.
/// 2) Convert calls between kernels to direct calls (on the QPU).
/// 3) Aggressively inline all calls.
/// 4) Reject measurements and qubit allocations in atomic quantum regions.
/// 5) Detect if kernel inlining has failed and left behind calls to kernels.
/// Such a failure is most likely a sign that there is a cycle in the call
/// graph. [This check is a bad idea: this should be deferred to final codegen
/// when translating the final Quake IR.]
void cudaq::opt::addAggressiveInlining(OpPassManager &pm, bool fatalChecks,
                                       bool lowerUnwind) {
  llvm::StringMap<OpPassManager> opPipelines;
  if (lowerUnwind)
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
  pm.addPass(cudaq::opt::createConvertToDirectCalls());
  pm.addPass(createInlinerPass(opPipelines, defaultInlinerOptPipeline));
  pm.addPass(cudaq::opt::createVerifyAtomicQuantumRegions());
  if (fatalChecks)
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createCheckKernelCalls());
  // Cleanup after inlining. We want to remove any copies between buffers for
  // the original called function returning a span to the calling function as
  // they are now the same function.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<LLVM::LLVMFuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createEraseVectorCopyCtor());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
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
  PassOptions::Option<bool> lowerUnwind{
      *this, "lower-unwind",
      llvm::cl::desc("lower unwind operations before inlining"),
      llvm::cl::init(true)};
};
} // namespace

void cudaq::opt::registerAggressiveInliningPipeline() {
  PassPipelineRegistration<AggressiveInliningPipelineOptions>(
      "aggressive-inlining",
      "Convert calls between kernels to direct calls and inline functions.",
      [](OpPassManager &pm, const AggressiveInliningPipelineOptions &opt) {
        addAggressiveInlining(pm, opt.runFatalChecker, opt.lowerUnwind);
      });
}
