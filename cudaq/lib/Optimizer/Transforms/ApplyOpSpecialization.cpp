/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Characteristics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_APPLYSPECIALIZATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "apply-op-specialization"

using namespace mlir;

// MLIR dependency: internal name used by tablegen.
static constexpr const char segmentSizes[] = "operandSegmentSizes";

namespace {
/// A Quake ApplyOp can indicate any of the following: a regular call to a
/// Callable (kernel), a call to a variant of a Callable with some control
/// qubits, a call to a variant of a Callable in adjoint form, or a call to a
/// Callable that is both adjoint and has control qubits.
struct ApplyVariants {
  bool needsControlVariant = false;
  bool needsAdjointVariant = false;
  bool needsAdjointControlVariant = false;

  // Merge the variants from that set into this set of variants. Return true if
  // any variants are added to this set.
  bool merge(ApplyVariants that) {
    bool rv = false;

    auto checkAndSet = [&](bool &bit0, bool bit1) {
      rv |= !bit0 & bit1;
      bit0 = bit0 | bit1;
    };

    checkAndSet(needsControlVariant, that.needsControlVariant);
    checkAndSet(needsAdjointVariant, that.needsAdjointVariant);
    // `that` has control and uses `this` which has adjoint, or `that` has
    // adjoint and uses `this` which has control, so generate a `.adj.ctrl`
    // variant for `this`, if not already present
    checkAndSet(needsAdjointControlVariant,
                that.needsAdjointControlVariant ||
                    (that.needsControlVariant && needsAdjointVariant) ||
                    (that.needsAdjointVariant && needsControlVariant));
    return rv;
  }
};

/// Map from `func::FuncOp` to the variants to be created.
using ApplyOpAnalysisInfo = DenseMap<Operation *, ApplyVariants>;

/// This analysis scans the IR for `ApplyOp`s to see which ones need to have
/// variants created.
struct ApplyOpAnalysis {
  ApplyOpAnalysis(ModuleOp op, bool constProp)
      : module(op), constProp(constProp) {
    performAnalysis(op.getOperation());
  }

  ApplyOpAnalysisInfo &getMutableAnalysisInfo() { return infoMap; }

  void performAnalysis(Operation *op) {
    scanAndUpdateMap(op);
    propagateTransitiveClosure();
  }

  /// Walk all ApplyOps under \p root and update infoMap. Returns true if any
  /// new variant requirements were added.
  bool scanAndUpdateMap(Operation *root) {
    bool changed = false;
    root->walk(
        [&](cudaq::quake::ApplyOp apply) { changed |= processApplyOp(apply); });
    return changed;
  }

  /// Process a specific list of ApplyOps and update infoMap. Used during the
  /// refinement loop to process only the ApplyOps newly created by variant
  /// generation, rather than rescanning the entire module.
  bool scanAndUpdateMap(SmallVectorImpl<cudaq::quake::ApplyOp> &applyOps) {
    bool changed = false;
    for (auto &apply : applyOps)
      changed |= processApplyOp(apply);
    return changed;
  }

  void propagateTransitiveClosure() {
    // Propagate the transitive closure over the call tree.
    bool changed = true;
    while (changed) {
      changed = false;
      ApplyOpAnalysisInfo cloneMap(infoMap);
      for (auto pr : cloneMap) {
        auto &func = pr.first;
        auto &variant = pr.second;
        func->walk([&](cudaq::quake::ApplyOp apply) {
          auto callee = lookupCallee(apply);
          auto iter = infoMap.find(callee);
          if (iter == infoMap.end()) {
            infoMap.insert(std::make_pair(callee.getOperation(), variant));
            changed = true;
          } else {
            if (infoMap[callee].merge(variant))
              changed = true;
          }
        });
      }
    }
  }

private:
  /// Apply constProp rewrites to \p apply if enabled, then merge any variant
  /// requirements for its callee into infoMap. Returns true if infoMap changed.
  /// \p apply may be updated in place if the op is replaced by constProp.
  bool processApplyOp(cudaq::quake::ApplyOp &apply) {
    if (constProp && apply.getCallee()) {
      // If some of the arguments in getActuals() are constants, then
      // materialize those constants in a clone of the variant. The
      // specialized variant will then be able to perform better constant
      // propagation even if not inlined.
      auto calleeName = apply.getCallee()->getRootReference().str();
      if (auto genericFunc = module.lookupSymbol<func::FuncOp>(calleeName)) {
        SmallVector<Value> newArgs{apply.getActuals().begin(),
                                   apply.getActuals().end()};
        IRMapping mapper;
        SmallVector<Value> preservedArgs;
        SmallVector<Type> inputTys;
        SmallVector<arith::ConstantOp> moveConsts;
        bool updateSignature = false;
        SmallVector<unsigned> specializedPositions;
        for (auto [idx, v] : llvm::enumerate(newArgs)) {
          if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
            auto newConst = c.clone();
            moveConsts.push_back(newConst);
            mapper.map(genericFunc.getArgument(idx), newConst);
            LLVM_DEBUG(llvm::dbgs() << "apply has constant arguments.\n");
          } else {
            if (auto relax = v.getDefiningOp<cudaq::quake::RelaxSizeOp>()) {
              // Also, specialize any relaxed veq types.
              v = relax.getInputVec();
              updateSignature = true;
              specializedPositions.push_back(preservedArgs.size());
              LLVM_DEBUG(llvm::dbgs() << "specializing apply veq argument ("
                                      << v.getType() << ")\n");
            }
            inputTys.push_back(v.getType());
            preservedArgs.push_back(v);
          }
        }

        if (!moveConsts.empty()) {
          // Possible code size improvement: this could avoid cloning
          // duplicates by appending the position and constant value into the
          // new cloned function's name.
          func::FuncOp newFunc = genericFunc.clone(mapper);
          calleeName += std::string{"."} + std::to_string(counter++);
          newFunc.setName(calleeName);
          auto *ctx = apply->getContext();
          if (updateSignature) {
            newFunc.setFunctionType(
                FunctionType::get(ctx, inputTys, newFunc.getResultTypes()));
            for (auto [arg, ty] :
                 llvm::zip(newFunc.front().getArguments(), inputTys))
              arg.setType(ty);
            for (unsigned pos : specializedPositions) {
              auto *ctx = newFunc.getContext();
              OpBuilder builder(ctx);
              builder.setInsertionPoint(&newFunc.front().front());
              auto relax = cudaq::quake::RelaxSizeOp::create(
                  builder, newFunc.getLoc(),
                  cudaq::quake::VeqType::getUnsized(ctx),
                  newFunc.front().getArgument(pos));
              newFunc.front().getArgument(pos).replaceAllUsesExcept(
                  relax.getResult(), relax.getOperation());
            }
          }
          newFunc.setPrivate();
          Block &entry = newFunc.front();
          for (auto c : moveConsts)
            entry.push_front(c);
          module.push_back(newFunc);
          OpBuilder builder(apply);
          auto newApply = cudaq::quake::ApplyOp::create(
              builder, apply.getLoc(), apply.getResultTypes(),
              SymbolRefAttr::get(ctx, calleeName), apply.getIsAdj(),
              apply.getControls(), preservedArgs);
          apply->replaceAllUsesWith(newApply.getResults());
          apply->dropAllReferences();
          apply->erase();
          LLVM_DEBUG(llvm::dbgs() << "apply specialization including constant "
                                     "propagation of arguments\n"
                                  << newFunc << '\n');
          apply = newApply;
        }
      }
    }

    if (!apply.applyToVariant())
      return false;
    if (auto callee = lookupCallee(apply)) {
      ApplyVariants needed;
      if (apply.getIsAdj() && !apply.getControls().empty())
        needed.needsAdjointControlVariant = true;
      else if (apply.getIsAdj())
        needed.needsAdjointVariant = true;
      else if (!apply.getControls().empty())
        needed.needsControlVariant = true;
      auto *calleeOp = callee.getOperation();
      auto iter = infoMap.find(calleeOp);
      if (iter == infoMap.end()) {
        infoMap.insert({calleeOp, needed});
        return true;
      }
      return iter->second.merge(needed);
    }
    return false;
  }

  func::FuncOp lookupCallee(cudaq::quake::ApplyOp apply) {
    auto callee = apply.getCallee();
    if (callee)
      return module.lookupSymbol<func::FuncOp>(*callee);
    return {};
  }

  ModuleOp module;
  ApplyOpAnalysisInfo infoMap;
  bool constProp;
  unsigned counter = 0;
};
} // namespace

static std::string getAdjCtrlVariantFunctionName(const std::string &n) {
  return n + ".adj.ctrl";
}

static std::string getAdjVariantFunctionName(const std::string &n) {
  return n + ".adj";
}

static std::string getCtrlVariantFunctionName(const std::string &n) {
  return n + ".ctrl";
}

static std::string getVariantFunctionName(cudaq::quake::ApplyOp apply,
                                          const std::string &calleeName) {
  if (apply.getIsAdj() && !apply.getControls().empty())
    return getAdjCtrlVariantFunctionName(calleeName);
  if (apply.getIsAdj())
    return getAdjVariantFunctionName(calleeName);
  if (!apply.getControls().empty())
    return getCtrlVariantFunctionName(calleeName);
  return calleeName;
}

// Returns true if this region contains unstructured control flow. Branches
// between basic blocks in a Region are defined to be unstructured. A Region
// with a single Block which contains cc.scope, cc.loop and cc.if, which
// themselves contain single Blocks recursively, will be considered structured.
// FIXME: Limitation: at present, the compiler does not recover structured
// control flow from a primitive CFG.
static bool regionHasUnstructuredControlFlow(Region &region) {
  if (region.empty())
    return false;
  if (!region.hasOneBlock())
    return true;
  auto &block = region.front();
  for (auto &op : block) {
    if (op.getNumRegions() == 0)
      continue;
    if (op.hasTrait<cudaq::JumpWithUnwind>())
      return true;
    if (!isa<cudaq::cc::IfOp>(op) && !cudaq::opt::isaMonotonicLoop(&op) &&
        op.getNumRegions() > 1)
      return true; // Op has multiple regions but is not a known Op.
    for (auto &reg : op.getRegions())
      if (regionHasUnstructuredControlFlow(reg))
        return true;
  }
  return false;
}

namespace {
/// Replace a quake.apply op with a call to the correct variant function.
struct ApplyOpPattern : public OpRewritePattern<cudaq::quake::ApplyOp> {
  using Base = OpRewritePattern<cudaq::quake::ApplyOp>;

  explicit ApplyOpPattern(MLIRContext *ctx, bool constProp)
      : Base(ctx), constProp(constProp) {}

  LogicalResult matchAndRewrite(cudaq::quake::ApplyOp apply,
                                PatternRewriter &rewriter) const override {
    std::string calleeOrigName;
    FunctionType calleeSignature;
    if (auto callee = apply.getCallee()) {
      calleeOrigName = callee->getRootReference().str();
      auto fn =
          SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(apply, *callee);
      calleeSignature = fn.getFunctionType();
    } else {
      // Check if the first argument is a func.ConstantOp.
      auto calleeVal = apply.getIndirectCallee();
      if (!calleeVal)
        return failure();
      auto fc = calleeVal.getDefiningOp<func::ConstantOp>();
      if (!fc)
        return failure();
      calleeOrigName = fc.getValue().str();
      calleeSignature = dyn_cast<FunctionType>(fc.getResult().getType());
    }
    auto calleeName = getVariantFunctionName(apply, calleeOrigName);
    auto *ctx = apply.getContext();
    auto calleeAttr = FlatSymbolRefAttr::get(ctx, calleeName);
    if (!SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(apply, calleeAttr))
      return failure();
    auto unsizedVeqTy = cudaq::quake::VeqType::getUnsized(ctx);
    SmallVector<Value> newArgs;
    const bool addControls = !apply.getControls().empty();
    if (addControls) {
      auto consOp = cudaq::quake::ConcatOp::create(
          rewriter, apply.getLoc(), unsizedVeqTy, apply.getControls());
      newArgs.push_back(consOp);
    }
    SmallVector<Value> applyActuals{apply.getActuals().begin(),
                                    apply.getActuals().end()};
    // The first actual may be a closure if this apply is calling a callable. If
    // that is the case, update the symbol on the closure instantiation. Also,
    // if we're adding controls, then we must update the callable type to
    // account for the bonus veq argument.

    for (auto [v, toTy] :
         llvm::zip(applyActuals, calleeSignature.getInputs())) {
      if (constProp && v.getDefiningOp<arith::ConstantOp>())
        continue;
      Value arg = v;
      if (toTy == unsizedVeqTy && arg.getType() != toTy) {
        arg = cudaq::quake::ConcatOp::create(rewriter, apply.getLoc(),
                                             unsizedVeqTy, arg);
      } else if (isa<cudaq::cc::CallableType>(toTy) && arg.getType() == toTy) {
        if (auto instan =
                arg.getDefiningOp<cudaq::cc::InstantiateCallableOp>()) {
          arg = cudaq::cc::InstantiateCallableOp::create(
              rewriter, instan.getLoc(), instan.getSignature().getType(),
              calleeAttr, instan.getClosureData());
        }
      }
      newArgs.emplace_back(arg);
    }
    LLVM_DEBUG(llvm::dbgs() << "replacing: " << apply << '\n');
    [[maybe_unused]] auto result = rewriter.replaceOpWithNewOp<func::CallOp>(
        apply, apply.getResultTypes(), calleeAttr, newArgs);
    LLVM_DEBUG(llvm::dbgs() << "with " << result << '\n');
    return success();
  }

  const bool constProp;
};

struct FoldCallable : public OpRewritePattern<cudaq::quake::ApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ApplyOp apply,
                                PatternRewriter &rewriter) const override {
    // If we already know the callee function, there's nothing to do.
    if (apply.getCallee())
      return failure();

    Value ind = apply.getIndirectCallee();
    auto callee = ind.getDefiningOp<cudaq::cc::InstantiateCallableOp>();
    if (!callee)
      return failure();
    auto sym = callee.getCallee();
    SmallVector<Value> newArguments = {ind};
    newArguments.append(apply.getActuals().begin(), apply.getActuals().end());
    LLVM_DEBUG(llvm::dbgs() << "folding callable " << apply << '\n');
    [[maybe_unused]] auto result =
        rewriter.replaceOpWithNewOp<cudaq::quake::ApplyOp>(
            apply, apply.getResultTypes(), sym, apply.getIsAdj(),
            apply.getControls(), newArguments);
    LLVM_DEBUG(llvm::dbgs() << "as " << result << '\n');
    return success();
  }
};

class ApplySpecializationPass
    : public cudaq::opt::impl::ApplySpecializationBase<
          ApplySpecializationPass> {
public:
  using ApplySpecializationBase::ApplySpecializationBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<FoldCallable>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();

    ApplyOpAnalysis analysis(module, constantPropagation);
    auto &applyVariants = analysis.getMutableAnalysisInfo();

    // Iteratively create variants until convergence. During variant creation,
    // CallOpInterface ops inside cloned bodies are converted to ApplyOps (e.g.,
    // a call inside a control variant becomes quake.apply [ctrl]). These new
    // ApplyOps may reference callees not present in the original analysis, so
    // we rescan the module and repeat until no new variant requirements are
    // found.
    bool needsRefinement = true;
    while (needsRefinement) {
      SmallVector<cudaq::quake::ApplyOp> newApplyOps;
      if (failed(step1(applyVariants, newApplyOps)))
        return;
      needsRefinement = analysis.scanAndUpdateMap(newApplyOps);
      if (needsRefinement)
        analysis.propagateTransitiveClosure();
    }
    step2();
  }

  /// Step 1. Instantiate all the implied variants of functions from all
  /// quake.apply operations that were found. Any ApplyOps created from
  /// CallOpInterface conversions during variant generation are appended to
  /// \p newApplyOps for targeted follow-up analysis.
  [[nodiscard]] LogicalResult
  step1(const ApplyOpAnalysisInfo &applyVariants,
        SmallVectorImpl<cudaq::quake::ApplyOp> &newApplyOps) {
    ModuleOp module = getOperation();

    // Loop over all the globals in the module.
    for (auto &global : *module.getBody()) {
      auto variantIter = applyVariants.find(&global);
      if (variantIter == applyVariants.end())
        continue;

      // Found a FuncOp that needs to be specialized.
      auto func = dyn_cast<func::FuncOp>(global);
      assert(func && "global must be a FuncOp");
      auto &variant = variantIter->second;

      // A forward-declared kernel has no body to specialize. Attempting to
      // clone the empty region and read its entry block crashes the compiler
      // (issue #4268). Do not fail the pass here: this pass cannot assume it
      // has full program information, and the body may still be supplied later
      // in the pipeline (e.g. at JIT time). Leave the quake.apply ops in place;
      // any that survive to codegen are diagnosed by ApplyOpTrap, the point at
      // which a lingering apply is unambiguously unlowerable.
      if (func.getBody().empty())
        continue;

      if (variant.needsControlVariant)
        createControlVariantOf(func, newApplyOps);
      if (variant.needsAdjointVariant) {
        auto fnName = func.getName().str();
        if (failed(createAdjointVariantOf(
                func, getAdjVariantFunctionName(fnName), newApplyOps)))
          return failure();
      }
      if (variant.needsAdjointControlVariant)
        if (failed(createAdjointControlVariantOf(func, newApplyOps)))
          return failure();
    }
    return success();
  }

  /// Look for quake.compute_action operations or quake.apply triple patterns in
  /// the FuncOp \p func. In these cases, we do not want to add the controls to
  /// the compute and uncompute functions.
  DenseSet<Operation *> computeActionAnalysis(func::FuncOp func) {
    DenseSet<Operation *> controlNotNeeded;
    if (computeActionOptimization) {
      func->walk([&](Operation *op) {
        if (auto compAct = dyn_cast<cudaq::quake::ComputeActionOp>(op)) {
          // This is clearly a compute action. Mark the compute side.
          if (auto *defOp = compAct.getCompute().getDefiningOp()) {
            controlNotNeeded.insert(defOp);
          } else {
            compAct.emitError("compute value not determined");
            signalPassFailure();
          }
        } else if (auto app0 = dyn_cast<cudaq::quake::ApplyOp>(op)) {
          auto next1 = ++app0->getIterator();
          Operation &op1 = *next1;
          if (auto app1 = dyn_cast<cudaq::quake::ApplyOp>(op1)) {
            auto next2 = ++next1;
            Operation &op2 = *next2;
            if (auto app2 = dyn_cast<cudaq::quake::ApplyOp>(op2);
                app2 && (app0.getCalleeAttr() == app2.getCalleeAttr()) &&
                ((!app0.getIsAdj() && app2.getIsAdj()) ||
                 (app0.getIsAdj() && !app2.getIsAdj())) &&
                !controlNotNeeded.count(app1)) {
              // This is a compute_action lowered to 3 successive apply
              // operations. We want to add the control to ONLY the action, the
              // middle apply op, so mark the compute and uncompute applies.
              controlNotNeeded.insert(app0);
              controlNotNeeded.insert(app2);
            }
          }
        }
      });
    }
    return controlNotNeeded;
  }

  func::FuncOp
  createControlVariantOf(func::FuncOp func,
                         SmallVectorImpl<cudaq::quake::ApplyOp> &newApplyOps) {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();
    // Perform a pre-analysis to determine if func has any compute_action like
    // ops. If it does, then there is an exception case. Instead of applying the
    // controls to the compute kernel, just use the compute kernel (and
    // uncompute kernel) without the controls added.
    auto funcName = getCtrlVariantFunctionName(func.getName().str());
    if (auto lookup = module.lookupSymbol<func::FuncOp>(funcName))
      if (!lookup.getBody().empty())
        return lookup;
    LLVM_DEBUG(llvm::dbgs() << "creating control variant " << funcName << '\n');
    auto funcTy = func.getFunctionType();
    auto veqTy = cudaq::quake::VeqType::getUnsized(ctx);
    auto loc = func.getLoc();
    SmallVector<Type> inTys = {veqTy};
    inTys.append(funcTy.getInputs().begin(), funcTy.getInputs().end());
    auto newFunc = cudaq::opt::factory::createFunction(
        funcName, funcTy.getResults(), inTys, module);
    newFunc.setPrivate();
    IRMapping mapping;
    func.getBody().cloneInto(&newFunc.getBody(), mapping);
    auto controlNotNeeded = computeActionAnalysis(newFunc);
    auto newCond = newFunc.getBody().front().insertArgument(0u, veqTy, loc);

    newFunc.walk([&](Operation *op) {
      OpBuilder builder(op);
      if (op->hasTrait<cudaq::QuantumGate>()) {
        // If op is in a λ expr where the control is not needed, then skip it.
        if (auto parent = op->getParentOfType<cudaq::cc::CreateLambdaOp>())
          if (controlNotNeeded.count(parent))
            return;

        // This is a quantum op. It should be updated with an additional control
        // argument, `newCond`.
        auto arrAttr = cast<DenseI32ArrayAttr>(op->getAttr(segmentSizes));
        SmallVector<std::int32_t> arrRef{arrAttr.asArrayRef().begin(),
                                         arrAttr.asArrayRef().end()};
        SmallVector<Value> operands(op->getOperands().begin(),
                                    op->getOperands().begin() + arrAttr[0]);
        operands.push_back(newCond);
        operands.append(op->getOperands().begin() + arrAttr[0],
                        op->getOperands().end());
        ++arrRef[1];
        auto newArrAttr = DenseI32ArrayAttr::get(ctx, arrRef);
        NamedAttrList attrs(op->getAttrs());
        attrs.set(segmentSizes, newArrAttr);
        OperationState res(op->getLoc(), op->getName().getStringRef(), operands,
                           op->getResultTypes(), attrs);
        // FIXME: Quake quantum gates do have results.
        builder.create(res);
        op->erase();
      } else if (auto apply = dyn_cast<cudaq::quake::ApplyOp>(op)) {
        // If op is an apply and in the set `controlNotNeeded`, then skip it.
        if (controlNotNeeded.count(apply))
          return;
        SmallVector<Value> newControls = {newCond};
        newControls.append(apply.getControls().begin(),
                           apply.getControls().end());
        auto newApply = cudaq::quake::ApplyOp::create(
            builder, apply.getLoc(), apply.getResultTypes(),
            apply.getCalleeAttr(), apply.getIsAdjAttr(), newControls,
            apply.getActuals());
        apply->replaceAllUsesWith(newApply.getResults());
        apply->erase();
      } else if (auto call = dyn_cast<CallOpInterface>(op)) {
        // Since `op` is a vanilla call, we can always assert that it will be
        // replaced with the auto-generated control function.
        auto app = cudaq::quake::ApplyOp::create(
            builder, call->getLoc(), call->getResultTypes(),
            call.getCallableForCallee(), ValueRange{newCond},
            call.getArgOperands());
        LLVM_DEBUG(llvm::dbgs() << "replacing call: " << call
                                << " with an apply: " << app << '\n');
        newApplyOps.push_back(app);
        call->erase();
      }
    });
    return newFunc;
  }

  /// Return true if \p call can be converted to a quake.apply.
  static bool convertibleCallOpInterface(CallOpInterface call) {
    return isa<func::CallOp, cudaq::quake::ApplyOp, cudaq::quake::CallByRefOp,
               cudaq::cc::CallCallableOp, cudaq::cc::NoInlineCallOp>(call);
  }

  /// The adjoint variant of the function is the "reverse" computation. We want
  /// to reverse the flow graph so the gates appear "upside down". This process
  /// is not always possible as this algorithm will \em not go to heroic lengths
  /// to reverse classical computation that has loop-carried side-effects, etc.
  /// In such cases, this pass may fail with an error. That is, this pass
  /// \em{may violate} the composability design rule for autogeneration of
  /// adjoint kernels if and only if there is classical expressions that are not
  /// trivially reversible.
  [[nodiscard]] LogicalResult
  createAdjointVariantOf(func::FuncOp func, std::string &&funcName,
                         SmallVectorImpl<cudaq::quake::ApplyOp> &newApplyOps) {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    if (auto lookup = module.lookupSymbol<func::FuncOp>(funcName))
      if (!lookup.getBody().empty())
        return success();

    LLVM_DEBUG(llvm::dbgs() << "creating adjoint variant " << funcName << '\n');
    auto loc = func.getLoc();
    auto &funcBody = func.getBody();

    // Check our restrictions.
    if (regionHasUnstructuredControlFlow(funcBody)) {
      LLVM_DEBUG(llvm::dbgs() << "cannot make adjoint of " + funcName +
                                     ": unstructured control flow\n");
      if (legacyClassical)
        return failure();
      return func.emitOpError(
          "auto-generation of adjoint " + funcName +
          " failed. cannot reverse the control-flow of this kernel.");
    }
    // quake.apply implements CallOpInterface but can be handled below by
    // toggling isAdj. Some direct calls can be handled by promoting them to
    // quake.apply. Reject any other call-like ops and assume they cannot be
    // reversed.
    if (cudaq::opt::detail::hasCharacteristic(
            [](Operation &op) {
              if (auto call = dyn_cast<CallOpInterface>(op))
                return !convertibleCallOpInterface(call);
              return false;
            },
            *func.getOperation())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "cannot make adjoint of " + funcName + " with calls\n");
      if (legacyClassical)
        return failure();
      return func.emitOpError("auto-generation of adjoint " + funcName +
                              " failed. contains an unanalyzable call graph.");
    }
    if (cudaq::opt::detail::hasCharacteristic(
            [](Operation &op) { return isa<cudaq::cc::CreateLambdaOp>(op); },
            *func.getOperation())) {
      LLVM_DEBUG(llvm::dbgs() << "cannot make adjoint of " + funcName +
                                     " with lambda expressions\n");
      if (legacyClassical)
        return failure();
      return func.emitOpError("auto-generation of adjoint " + funcName +
                              " failed. " + funcName +
                              " contains lambdas. was the lambda-lifting pass "
                              "run before this pass?");
    }
    if (cudaq::opt::hasMeasureOp(func)) {
      LLVM_DEBUG(llvm::dbgs() << "cannot make adjoint of " + funcName +
                                     " with a measurement\n");
      if (legacyClassical)
        return failure();
      return func.emitOpError("auto-generation of adjoint " + funcName +
                              " failed. " + funcName +
                              " contains measurements. was the "
                              "remove-measurements pass run before this pass?");
    }

    auto funcTy = func.getFunctionType();
    auto newFunc = cudaq::opt::factory::createFunction(
        funcName, funcTy.getResults(), funcTy.getInputs(), module);
    newFunc.setPrivate();
    IRMapping mapping;
    funcBody.cloneInto(&newFunc.getBody(), mapping);
    if (failed(reverseTheOpsInTheBlock</*checkEmpty=*/true>(
            loc, newFunc.getBody().front().getTerminator(),
            getOpsToInvert(newFunc.getBody().front()), newApplyOps))) {
      if (legacyClassical)
        return failure();
      return func.emitOpError("auto-generation of adjoint " + funcName +
                              " failed. could not reverse the kernel.");
    }
    return success();
  }

  // Collect all the operations in \p block that we want to emit in reverse
  // order for the adjoint. This includes all calls as they must be considered
  // part of the control-flow of the kernel.
  static SmallVector<Operation *> getOpsToInvert(Block &block) {
    SmallVector<Operation *> ops;
    for (auto &op : block)
      if (cudaq::opt::hasQuantum(op) || cudaq::opt::hasCallOp(op))
        ops.push_back(&op);
    return ops;
  }

  static Value cloneRootSubexpression(OpBuilder &builder, Block &block,
                                      Value root, cudaq::cc::LoopOp loop) {
    if (auto *op = root.getDefiningOp()) {
      if (op->getBlock() == &block) {
        for (Value v : op->getOperands())
          cloneRootSubexpression(builder, block, v, loop);
        return builder.clone(*op)->getResult(0);
      }
      return root;
    }
    auto blkArg = cast<BlockArgument>(root);
    if (blkArg.getOwner() == &block)
      return loop.getInitialArgs()[blkArg.getArgNumber()];
    return root;
  }

  /// Build an `Arith::ConstantOp` for an integral type (including index).
  static Value createIntConstant(OpBuilder &builder, Location loc, Type ty,
                                 std::int64_t val) {
    auto attr = builder.getIntegerAttr(ty, val);
    return arith::ConstantOp::create(builder, loc, ty, attr);
  }

  /// Clone the LoopOp, \p loop, and return a new LoopOp that runs the loop
  /// backwards. The loop is assumed to be a simple monotonic loop (a generator
  /// of a monotonic indexing function). The loop control could be in either the
  /// memory or value domain. The step and bounds of the original loop must be
  /// loop invariant.
  static cudaq::cc::LoopOp cloneReversedLoop(OpBuilder &builder,
                                             cudaq::cc::LoopOp loop) {
    auto loopComponents = cudaq::opt::getLoopComponents(loop);
    assert(loopComponents && "could not determine components of loop");
    auto stepIsAnAddOp = loopComponents->stepIsAnAddOp();
    auto commuteTheAddOp = loopComponents->shouldCommuteStepOp();

    // Now rewrite the loop to run in reverse. `builder` is set at the point we
    // want to insert the new loop.
    auto loc = loop.getLoc();
    Value newTermVal =
        cloneRootSubexpression(builder, loop.getWhileRegion().back(),
                               loopComponents->compareValue, loop);
    Value newStepVal = cloneRootSubexpression(
        builder, loop.getStepRegion().back(), loopComponents->stepValue, loop);
    auto zero = createIntConstant(builder, loc, newStepVal.getType(), 0);
    if (!stepIsAnAddOp) {
      // Negate the step value when arith.subi.
      newStepVal = arith::SubIOp::create(builder, loc, zero, newStepVal);
    }
    Value iters = arith::SubIOp::create(
        builder, loc, newTermVal,
        loop.getInitialArgs()[*loopComponents->induction]);
    auto cmpOp = cast<arith::CmpIOp>(loopComponents->compareOp);
    auto pred = cmpOp.getPredicate();
    auto one = createIntConstant(builder, loc, iters.getType(), 1);
    if (cudaq::opt::isSemiOpenPredicate(pred)) {
      Value negStepCond = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::slt, newStepVal, zero);
      auto negOne = createIntConstant(builder, loc, iters.getType(), -1);
      Value adj = arith::SelectOp::create(builder, loc, iters.getType(),
                                          negStepCond, one, negOne);
      iters = arith::AddIOp::create(builder, loc, iters, adj);
    }
    iters = arith::AddIOp::create(builder, loc, iters, newStepVal);
    iters = arith::DivSIOp::create(builder, loc, iters, newStepVal);
    Value noLoopCond = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, iters, zero);
    iters = arith::SelectOp::create(builder, loc, iters.getType(), noLoopCond,
                                    iters, zero);
    Value lastIter = arith::SubIOp::create(builder, loc, iters, one);
    Value nStep = arith::MulIOp::create(builder, loc, lastIter, newStepVal);
    Value newInitVal = arith::AddIOp::create(
        builder, loc, loopComponents->initialValue, nStep);

    // Create the list of input arguments to loop. We're going to add an
    // argument to the end that is the number of iterations left to execute.
    SmallVector<Value> inputs = loop.getInitialArgs();
    assert(*loopComponents->induction < inputs.size());
    inputs[*loopComponents->induction] = newInitVal;
    inputs.push_back(iters);

    // Create the new LoopOp. This requires threading the new value that is the
    // number of iterations left to execute. In the whileRegion, update the
    // condition test to use the new argument. In the bodyRegion, update to pass
    // through the new argument. In the stepRegion, decrement the new argument
    // by 1 and convert the original step expression to be a negative step.
    IRRewriter rewriter(builder);
    return cudaq::cc::LoopOp::create(
        rewriter, loc, ValueRange{inputs}.getTypes(), inputs,
        /*postCondition=*/false,
        [&](OpBuilder &builder, Location loc, Region &region) {
          IRMapping dummyMap;
          loop.getWhileRegion().cloneInto(&region, dummyMap);
          Block &entry = region.front();
          entry.addArgument(iters.getType(), loc);
          Block &block = region.back();
          auto condOp = cast<cudaq::cc::ConditionOp>(block.back());
          IRRewriter rewriter(builder);
          rewriter.setInsertionPoint(condOp);
          SmallVector<Value> args = condOp.getResults();
          Value trip = block.getArguments().back();
          args.push_back(trip);
          auto zero = createIntConstant(builder, loc, trip.getType(), 0);
          auto newCond = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::sgt, trip, zero);
          rewriter.replaceOpWithNewOp<cudaq::cc::ConditionOp>(condOp, newCond,
                                                              args);
        },
        [&](OpBuilder &builder, Location loc, Region &region) {
          IRMapping dummyMap;
          loop.getBodyRegion().cloneInto(&region, dummyMap);
          Block &entry = region.front();
          entry.addArgument(iters.getType(), loc);
          auto &term = region.back().back();
          IRRewriter rewriter(builder);
          rewriter.setInsertionPoint(&term);
          SmallVector<Value> args(entry.getArguments().begin(),
                                  entry.getArguments().end());
          rewriter.replaceOpWithNewOp<cudaq::cc::ContinueOp>(&term, args);
        },
        [&](OpBuilder &builder, Location loc, Region &region) {
          IRMapping dummyMap;
          loop.getStepRegion().cloneInto(&region, dummyMap);
          Block &entry = region.front();
          entry.addArgument(iters.getType(), loc);
          auto contOp = cast<cudaq::cc::ContinueOp>(region.back().back());
          IRRewriter rewriter(builder);
          rewriter.setInsertionPoint(contOp);
          SmallVector<Value> args = contOp.getOperands();
          // In the value case, replace after the clone since we need to
          // thread the new value and it's trivial to find the stepOp.
          auto *stepOp =
              contOp.getOperand(*loopComponents->induction).getDefiningOp();
          assert(stepOp && "must be a step");
          auto newBump = [&]() -> Value {
            if (stepIsAnAddOp)
              return arith::SubIOp::create(
                  rewriter, loc, stepOp->getOperand(commuteTheAddOp ? 1 : 0),
                  stepOp->getOperand(commuteTheAddOp ? 0 : 1));
            return arith::AddIOp::create(rewriter, loc, stepOp->getOperands());
          }();
          args[*loopComponents->induction] = newBump;
          auto one = createIntConstant(rewriter, loc, iters.getType(), 1);
          args.push_back(arith::SubIOp::create(
              rewriter, loc, entry.getArguments().back(), one));
          rewriter.replaceOpWithNewOp<cudaq::cc::ContinueOp>(contOp, args);
        });
  }

  /// For each Op in \p invertedOps, visit them in reverse order and move each
  /// to just in front of \p term (the end of the function). This reversal of
  /// the order of quantum operations is done recursively.
  ///
  /// If `checkEmpty` is set to `true` (and we're not in legacy classical
  /// expression mode) then a block without quantum operations to reverse is
  /// considered a fatal error. Autogeneration of an adjoint kernel with no
  /// quantum operations is no longer just naive, but now simply disallowed.
  template <bool checkEmpty = false>
  LogicalResult
  reverseTheOpsInTheBlock(Location loc, Operation *term,
                          SmallVector<Operation *> &&invertedOps,
                          SmallVectorImpl<cudaq::quake::ApplyOp> &newApplyOps) {
    OpBuilder builder(term);
    if (!legacyClassical) {
      if (checkEmpty && invertedOps.empty())
        return term->emitOpError("no quantum operations to reverse.");
      // Check that classical values do not have data-flow to subsequent ops
      auto begin = invertedOps.begin() + 1;
      for (auto *inv : invertedOps) {
        if (inv->getNumResults() == 0)
          continue;
        for (auto res : inv->getResults())
          if (!cudaq::quake::isLinearType(res.getType()))
            for (auto *usr : res.getUsers())
              if (std::find(begin, invertedOps.end(), usr) != invertedOps.end())
                return usr->emitOpError("control-flow def-use not reversible.");
        ++begin;
      }
    }
    for (auto *op : llvm::reverse(invertedOps)) {
      auto invert = [&](Region &reg) {
        if (reg.empty())
          return success();
        auto &block = reg.front();
        // Empty blocks in, for example, else regions are not errors.
        if (failed(reverseTheOpsInTheBlock(loc, block.getTerminator(),
                                           getOpsToInvert(block), newApplyOps)))
          return failure();
        return success();
      };
      if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "moving if: " << ifOp << ".\n");
        auto *newIf = builder.clone(*op);
        op->replaceAllUsesWith(newIf);
        op->erase();
        auto newIfOp = cast<cudaq::cc::IfOp>(newIf);
        if (failed(invert(newIfOp.getThenRegion())))
          if (!legacyClassical)
            return ifOp.emitOpError("then block not reversed.");
        if (failed(invert(newIfOp.getElseRegion())))
          if (!legacyClassical)
            return ifOp.emitOpError("else block not reversed.");
        continue;
      }
      if (auto loopOp = dyn_cast<cudaq::cc::LoopOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "moving loop: " << loopOp << ".\n");
        auto newLoopOp = cloneReversedLoop(builder, loopOp);
        LLVM_DEBUG(llvm::dbgs() << "  to: " << newLoopOp << ".\n");
        op->replaceAllUsesWith(newLoopOp->getResults().drop_back());
        op->erase();
        if (failed(invert(newLoopOp.getBodyRegion())))
          if (!legacyClassical)
            return loopOp.emitOpError("loop not reversed.");
        continue;
      }
      if (auto scopeOp = dyn_cast<cudaq::cc::ScopeOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "moving scope: " << scopeOp << ".\n");
        auto *newScope = builder.clone(*op);
        op->replaceAllUsesWith(newScope);
        op->erase();
        auto newScopeOp = cast<cudaq::cc::ScopeOp>(newScope);
        if (failed(invert(newScopeOp.getInitRegion())))
          if (!legacyClassical)
            return scopeOp.emitOpError("scope not reversed.");
        continue;
      }

      if (auto applyOp = dyn_cast<cudaq::quake::ApplyOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "moving apply: " << applyOp << ".\n");
        // Adjoint of an ApplyOp: toggles the isAdj flag.
        UnitAttr newIsAdj = applyOp.getIsAdj()
                                ? UnitAttr{}
                                : UnitAttr::get(builder.getContext());
        [[maybe_unused]] auto newCall = cudaq::quake::ApplyOp::create(
            builder, applyOp.getLoc(), applyOp.getResultTypes(),
            applyOp.getCalleeAttr(), newIsAdj, applyOp.getControls(),
            applyOp.getActuals());
        LLVM_DEBUG(llvm::dbgs() << "toggled as: " << newCall << ".\n");
        applyOp->erase();
        continue;
      }

      if (auto call = dyn_cast<CallOpInterface>(op)) {
        // Since `op` is a vanilla call, we can always assert that it will be
        // replaced with the auto-generated adjoint function.
        auto app = cudaq::quake::ApplyOp::create(
            builder, call->getLoc(), call->getResultTypes(),
            call.getCallableForCallee(),
            /*is_adj=*/true, call.getArgOperands());
        LLVM_DEBUG(llvm::dbgs() << "replacing call: " << call
                                << " with an apply " << app << '\n');
        newApplyOps.push_back(app);
        call->replaceAllUsesWith(app->getResults());
        call->erase();
        continue;
      }

      bool opWasNegated = false;
      IRMapping mapper;
      LLVM_DEBUG(llvm::dbgs() << "moving quantum op: " << *op << ".\n");
      auto arrAttr = cast<DenseI32ArrayAttr>(op->getAttr(segmentSizes));
      // Walk over any floating-point parameters to `op` and negate them.
      for (auto iter = op->getOperands().begin(),
                endIter = op->getOperands().begin() + arrAttr[0];
           iter != endIter; ++iter) {
        Value val = *iter;
        Value neg = arith::NegFOp::create(builder, loc, val.getType(), val);
        mapper.map(val, neg);
        opWasNegated = true;
      }

      // If this is a quantum op that is not self adjoint, we need to adjoint
      // it.
      if (auto quantumOp =
              dyn_cast_or_null<cudaq::quake::OperatorInterface>(op);
          !quantumOp->hasTrait<cudaq::Hermitian>() && !opWasNegated) {
        if (op->hasAttr("is_adj"))
          op->removeAttr("is_adj");
        else
          op->setAttr("is_adj", builder.getUnitAttr());
      }

      [[maybe_unused]] auto *newOp = builder.clone(*op, mapper);
      assert(newOp->getNumResults() == 0);
      op->erase();
    }
    return success();
  }

  /// This is the combination of adjoint and control transformations. We will
  /// create a control variant here, even if it wasn't needed to simplify
  /// things. The dead variant can be eliminated as unreferenced.
  [[nodiscard]] LogicalResult createAdjointControlVariantOf(
      func::FuncOp func, SmallVectorImpl<cudaq::quake::ApplyOp> &newApplyOps) {
    ModuleOp module = getOperation();
    auto funcName = func.getName().str();
    auto ctrlFuncName = getCtrlVariantFunctionName(funcName);
    auto ctrlFunc = module.lookupSymbol<func::FuncOp>(ctrlFuncName);
    if (!ctrlFunc)
      ctrlFunc = createControlVariantOf(func, newApplyOps);

    auto newFuncName = getAdjCtrlVariantFunctionName(funcName);
    return createAdjointVariantOf(ctrlFunc, std::move(newFuncName),
                                  newApplyOps);
  }

  /// Step 2. Specialize all the quake.apply ops and convert them to calls.
  void step2() {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ApplyOpPattern>(ctx, constantPropagation);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After apply specialization:\n"
                            << module << "\n\n");
  }
};
} // namespace
