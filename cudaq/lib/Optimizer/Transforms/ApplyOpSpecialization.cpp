/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
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

namespace {
/// A Quake ApplyOp can indicate any of the following: a regular call to a
/// Callable (kernel), a call to a variant of a Callable with some control
/// qubits, a call to a variant of a Callable in adjoint form, or a call to a
/// Callable that is both adjoint and has control qubits. Finally, it can be
/// used as an implicit wrap/call/unwrap shorthand to pass quake value types to
/// quake reference type argument positions with updated result types.
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
        SmallVector<Value> preservedArgs;
        SmallVector<Type> inputTys;
        SmallVector<arith::ConstantOp> moveConsts;
        SmallVector<unsigned> constArgIndices;
        bool updateSignature = false;
        SmallVector<unsigned> specializedPositions;
        for (auto [idx, v] : llvm::enumerate(newArgs)) {
          if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
            // Materialize the constant in the cloned function's body so it
            // can be folded/propagated there, but still pass the original
            // constant through as an (now dead) actual argument. Keeping the
            // formal parameter in place - rather than pruning it - keeps the
            // clone's signature identical to `genericFunc`'s, so the call
            // site and callee always agree on arity without any additional
            // bookkeeping.
            moveConsts.push_back(c);
            constArgIndices.push_back(static_cast<unsigned>(idx));
            LLVM_DEBUG(llvm::dbgs() << "apply has constant arguments.\n");
          }
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

        if (!moveConsts.empty()) {
          // Possible code size improvement: this could avoid cloning
          // duplicates by appending the position and constant value into the
          // new cloned function's name.
          //
          // Plain clone (no IRMapping substitution): pre-populating a mapper
          // with `genericFunc.getArgument(idx) -> constant` before cloning
          // corrupts the clone's block-argument list (it desyncs from the
          // unchanged FunctionType). Instead, clone faithfully, then splice
          // the constants into the entry block and redirect the still-present
          // (now dead) formal arguments' uses to them.
          func::FuncOp newFunc = genericFunc.clone();
          auto specializedName =
              calleeName + std::string{"."} + std::to_string(counter++);
          newFunc.setName(specializedName);
          auto *ctx = apply->getContext();
          {
            OpBuilder constBuilder(ctx);
            constBuilder.setInsertionPointToStart(&newFunc.front());
            for (auto [argIdx, constOp] :
                 llvm::zip(constArgIndices, moveConsts)) {
              auto *newConst = constBuilder.clone(*constOp);
              newFunc.front().getArgument(argIdx).replaceAllUsesWith(
                  newConst->getResult(0));
            }
          }
          for (std::size_t i = 0, N = preservedArgs.size(); i != N; ++i) {
            auto callTy = dyn_cast<cudaq::cc::CallableType>(inputTys[i]);
            if (!callTy)
              continue;
            auto instan =
                preservedArgs[i]
                    .getDefiningOp<cudaq::cc::InstantiateCallableOp>();
            if (!instan)
              continue;
            if (instan.getCallee().getRootReference().str() != calleeName)
              continue;
            // Sync the instantiate_callable with the new apply.
            SmallVector<Type> callInTys{inputTys.begin(), inputTys.begin() + i};
            callInTys.append(inputTys.begin() + i + 1, inputTys.end());
            OpBuilder builder(ctx);
            builder.setInsertionPoint(instan);
            auto newFuncTy =
                FunctionType::get(ctx, callInTys, newFunc.getResultTypes());
            auto sigTy = cudaq::cc::CallableType::get(newFuncTy);
            auto newInstan = cudaq::cc::InstantiateCallableOp::create(
                builder, instan.getLoc(), sigTy,
                SymbolRefAttr::get(ctx, specializedName),
                instan.getClosureData());
            // Only redirect uses that are quake.apply operands.  Other uses
            // (e.g. func.call) keep the original callable type and must not
            // be touched.
            instan.getResult().replaceUsesWithIf(
                newInstan.getResult(), [](mlir::OpOperand &use) {
                  return isa<cudaq::quake::ApplyOp>(use.getOwner());
                });
            preservedArgs[i] = newInstan.getResult();
            inputTys[i] = newInstan.getResult().getType();
            if (instan.getResult().use_empty())
              instan.erase();
            updateSignature = true;
            break;
          }
          if (updateSignature) {
            auto newFuncTy =
                FunctionType::get(ctx, inputTys, newFunc.getResultTypes());
            newFunc.setFunctionType(newFuncTy);
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
          module.push_back(newFunc);
          OpBuilder builder(apply);
          auto newApply = cudaq::quake::ApplyOp::create(
              builder, apply.getLoc(), apply.getResultTypes(),
              SymbolRefAttr::get(ctx, specializedName), apply.getIsAdj(),
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
    // NB: the following means kernels with cc.scope lexical blocks cannot be
    // auto-adjointed.
    if (!isa<cudaq::cc::IfOp>(op) && !cudaq::opt::isaMonotonicLoop(&op) &&
        op.getNumRegions() > 1)
      return true; // Op has multiple regions but is not a known Op.
    for (auto &reg : op.getRegions())
      if (regionHasUnstructuredControlFlow(reg))
        return true;
  }
  return false;
}

/// Build (or look up) a control-closure wrapper thunk for an indirect apply
/// with controls.  The wrapper is modelled on the original trampoline:
///
///   Original trampoline body:
///   ```
///     %data = cc.callable_closure %self : (...) -> (closure_types...)
///     call @lifted_lambda(%data..., %formal_args...) : ...
///   ```
///
///   Wrapper body (modified copy):
///   ```
///     %ctrl0, %ctrl1, ..., %data... =
///         cc.callable_closure %self : (...) -> (ctrl_types...,
///         closure_types...)
///     %veq = quake.concat %ctrl0, %ctrl1, ... : (...) -> !quake.veq<?>
///     call @lifted_lambda.ctrl(%veq, %data..., %formal_args...) : ...
///   ```
///
/// The control variant of the lifted lambda (`@lifted_lambda.ctrl`) is created
/// by the normal analysis / step-1 path; the wrapper merely calls it directly,
/// bypassing the thunk-ctrl layer entirely.  The wrapper's external callable
/// type is identical to the original - no type change propagates outward.
/// If \p ty's input at \p selfArgIdx is a CallableType whose own signature's
/// inputs exactly match \p ty's remaining inputs (i.e. \p ty follows the
/// dynamic trampoline convention - "one of my own arguments is essentially a
/// callable handle to myself"), return that CallableType. Otherwise return
/// null.
///
/// This is the single source of truth for whether a function keeps such a
/// dynamic callable argument, used both by createControlVariantOf (at
/// selfArgIdx 0, deciding whether/how to rebuild that argument's type when
/// it prepends a control veq) and by buildCtrlClosureInstantiation (at
/// selfArgIdx 1, applied to an already-built `.ctrl` variant to recover that
/// same argument's exact type).
static cudaq::cc::CallableType dynamicArgType(FunctionType ty,
                                              unsigned selfArgIdx) {
  if (ty.getNumInputs() <= selfArgIdx)
    return {};
  auto callTy = dyn_cast<cudaq::cc::CallableType>(ty.getInput(selfArgIdx));
  if (!callTy)
    return {};
  SmallVector<Type> rest(ty.getInputs());
  rest.erase(rest.begin() + selfArgIdx);
  if (callTy.getSignature().getInputs() != ArrayRef<Type>(rest))
    return {};
  return callTy;
}

/// If \p targetFnTy - a `.ctrl`/`.adj.ctrl` variant's own function type - keeps
/// a dynamic callable as its own argument at position 1 (position 0 being the
/// control veq every such variant prepends), build a fresh
/// `cc.instantiate_callable` of that exact type, targeting \p targetAttr, with
/// \p closureData as its captured data, and return it. Otherwise return a null
/// Value.
///
/// This is the one place that both decides whether a call to a `.ctrl` variant
/// needs a rebuilt self argument and constructs it. Every caller that threads
/// arguments into such a call must go through this rather than re-deriving the
/// same decision independently (via its own dynamicArgType check plus its own
/// `InstantiateCallableOp::create`).
static Value maybeBuildCtrlSelfArg(PatternRewriter &rewriter, Location loc,
                                   FunctionType targetFnTy,
                                   FlatSymbolRefAttr targetAttr,
                                   ValueRange closureData) {
  auto selfTy = dynamicArgType(targetFnTy, 1);
  if (!selfTy)
    return {};
  return cudaq::cc::InstantiateCallableOp::create(rewriter, loc, selfTy,
                                                  targetAttr, closureData)
      .getResult();
}

/// \p ctrlFn keeps a dynamic callable as its own argument at \p selfArgIdx (see
/// dynamicArgType), captured with some original closure data (e.g. a free
/// variable like an angle - see isApplicativeClosure in
/// buildCtrlClosureInstantiation). At runtime that data is recovered inside \p
/// ctrlFn's own body by a cc.callable_closure op unpacking that argument; \p
/// idx selects which element of that data (matching the position it had in the
/// original closure, e.g. origClosureData[idx]).
///
/// Returns the value inside \p ctrlFn's body that represents that element, or a
/// null Value if no such unpack op is found (e.g. the argument is unused). This
/// is a purely structural lookup - the returned element may or may not be a
/// constant; callers doing constant folding (see
/// buildCtrlClosureInstantiation's constant-prop step) decide that
/// independently, by inspecting whatever caused this function to be called
/// with that closure data in the first place, not this op's contents.
static Value selfClosureElement(func::FuncOp ctrlFn, unsigned selfArgIdx,
                                unsigned idx) {
  for (auto *user : ctrlFn.getArgument(selfArgIdx).getUsers())
    if (auto cco = dyn_cast<cudaq::cc::CallableClosureOp>(user))
      return cco.getResult(idx);
  return {};
}

/// Returns the new cc.instantiate_callable and the FlatSymbolRefAttr of the
/// wrapper function so the caller can redirect its call target.
static std::pair<cudaq::cc::InstantiateCallableOp, FlatSymbolRefAttr>
buildCtrlClosureInstantiation(
    PatternRewriter &rewriter, Location loc, ModuleOp module,
    cudaq::cc::InstantiateCallableOp origInstan, // null for direct-callee case
    StringRef calleeOrigName,          // trampoline OR direct kernel name
    ValueRange ctrlRefs,               // already-converted ref/veq controls
    cudaq::cc::CallableType origSigTy, // callable type (unchanged externally)
    ValueRange callSiteClosureData,    // origInstan.getClosureData() or empty
    ValueRange callSiteFormalArgs,     // apply formal actuals (excl. callable)
    bool constProp,                    // whether to materialize constants
    // Name suffix to append to the inner target's bare name to reach the
    // variant this apply actually needs (".ctrl" or ".adj.ctrl"), and the
    // matching infix for the wrapper's own name (".ctrl_closure" or
    // ".adj_ctrl_closure"). The caller computes these (see
    // getVariantFunctionName) rather than this function re-deriving them
    // from an isAdj flag: the wrapper's name must include the same
    // adjoint-ness that its target does, or a wrapper for the plain control
    // of some callee collides with (and gets silently reused in place of) a
    // distinct wrapper for the control of that callee's adjoint.
    StringRef ctrlNameSuffix, StringRef closureNameInfix,
    unsigned &specializedCounter, // counter for unique ctrl-variant names
    MLIRContext *ctx) {

  auto unsizedVeqTy = cudaq::quake::VeqType::getUnsized(ctx);

  // 1. find the inner function to call in the ctrl variant
  //
  // For the direct case (origInstan == null): calleeOrigName IS the original
  // kernel's name.
  //
  // For the indirect case (origInstan != null): origInstan is the closure
  // instantiation bound to the callable-typed actual at this apply's callee.
  // Two sub-cases:
  //  - Applicative: origInstan.getCallee() == calleeOrigName (this apply's
  //    own callee is a dynamic trampoline - see dynamicArgType above - and
  //    origInstan applies it to itself: it was instantiated with a reference
  //    to that very function). calleeOrigName is the function to specialize;
  //    its own `.ctrl` variant is unconditionally built by step1's
  //    createControlVariantOf (see ApplySpecializationPass::runOnOperation)
  //    before step2 (which calls this function) ever runs.
  //  - Distinct closure: origInstan.getCallee() names some other lambda's
  //    thunk entirely (e.g. a callable argument passed alongside a real
  //    kernel callee) - that name is exactly what to specialize.
  StringRef innerFuncName = calleeOrigName;
  if (origInstan) {
    StringRef instanCallee =
        origInstan.getCallee().getRootReference().getValue();
    if (instanCallee != calleeOrigName)
      innerFuncName = instanCallee;
  }

  // True when origInstan applies calleeOrigName's own dynamic trampoline to
  // itself (its callee is calleeOrigName). In that case
  // origInstan.getClosureData() and this call's own callSiteFormalArgs
  // denote the exact same runtime values - the former captured them into
  // the applicative closure, the latter are those same values passed again
  // as this apply's own actuals (see FoldCallable, which built this
  // direct-callee apply by prepending the applicative instantiation as a
  // new leading argument while leaving the original actuals in place).
  // Below, only one of the two must be forwarded into the inner call, not
  // both, or the inner `.ctrl` variant gets called with roughly twice as
  // many arguments as it declares.
  bool isApplicativeClosure = origInstan && innerFuncName == calleeOrigName;

  // This apply may be the *adjoint* of innerFuncName (e.g. a control added
  // to a `quake.apply<adj> @innerFuncName` found by createControlVariantOf
  // while building some other kernel's `.ctrl` variant), so the caller-
  // supplied suffix (not a hardcoded ".ctrl") must be used here: appending
  // the plain (non-adjoint) suffix would target the wrong function.
  auto innerCtrlName = (innerFuncName + ctrlNameSuffix).str();
  auto innerCtrlAttr = SymbolRefAttr::get(ctx, innerCtrlName);

  // Determine whether the target `.ctrl` variant keeps a dynamic callable
  // as its own leading formal argument, and if so, recover that argument's
  // exact type directly from the variant itself (already built by step1. Input
  // 1 because input 0 is the control veq this variant itself prepends. Reading
  // the type from innerCtrlFunc directly, rather than re-deriving it later from
  // this call's own origSigTy (see the dynamic instantiation below), keeps the
  // two in agreement even when innerFuncName does not name the same function
  // whose closure this call was actually given.
  auto innerCtrlFunc = module.lookupSymbol<func::FuncOp>(innerCtrlName);
  bool needsSelfArg =
      innerCtrlFunc &&
      static_cast<bool>(dynamicArgType(innerCtrlFunc.getFunctionType(), 1));

  // 2. constant propagation into the ctrl variant
  //
  // When constProp is enabled, scan the call-site values (closure data and
  // formal args) for arith.constant ops.  For each constant found, clone the
  // ctrl variant under a fresh name (@kernel.ctrl.<secondary-index>) and
  // replace that element's uses inside the clone with the constant.  The
  // call site continues to pass all args unchanged - no type or call-site
  // rejiggering is required.
  //
  // Formal args, and a *distinct* closure's captured data, both show up as
  // ordinary flat block arguments of innerCtrlFunc (createControlVariantOf
  // keeps a distinct closure's captures as separate parameters), so those
  // elements are replaced directly by block-argument index.  An
  // *applicative* closure's captured data is not a flat block argument -
  // per isApplicativeClosure below, that data is threaded through the
  // self-arg's own closure instead (see maybeBuildCtrlSelfArg) - so its
  // elements are located via selfClosureElement instead, which finds where
  // the target's own body unpacks that data at runtime.
  if (constProp) {
    if (innerCtrlFunc && !innerCtrlFunc.getBody().empty()) {
      // Collect (block-arg-index, constant-op) pairs for formal args, and,
      // for a distinct closure, its closure data too. Block arg 0 is the
      // control veq; skip one more slot when the target itself keeps a
      // dynamic callable as its own leading formal argument.
      SmallVector<std::pair<unsigned, arith::ConstantOp>> toSpecializeArgs;
      unsigned offset = 1; // skip veq
      if (needsSelfArg)
        offset += 1; // skip the dynamic callable argument
      if (!isApplicativeClosure) {
        for (auto [i, v] : llvm::enumerate(callSiteClosureData))
          if (auto c = v.getDefiningOp<arith::ConstantOp>())
            toSpecializeArgs.push_back({static_cast<unsigned>(offset + i), c});
        offset += callSiteClosureData.size();
      }
      for (auto [i, v] : llvm::enumerate(callSiteFormalArgs))
        if (auto c = v.getDefiningOp<arith::ConstantOp>())
          toSpecializeArgs.push_back({static_cast<unsigned>(offset + i), c});

      // Collect (closure-element-index, constant-op) pairs for an
      // applicative closure's own captured data - see selfClosureElement.
      SmallVector<std::pair<unsigned, arith::ConstantOp>> toSpecializeClosure;
      if (isApplicativeClosure && needsSelfArg)
        for (auto [i, v] : llvm::enumerate(callSiteClosureData))
          if (auto c = v.getDefiningOp<arith::ConstantOp>())
            toSpecializeClosure.push_back({static_cast<unsigned>(i), c});

      if (!toSpecializeArgs.empty() || !toSpecializeClosure.empty()) {
        // Clone the ctrl variant and give it a unique secondary name.
        auto specializedName =
            innerCtrlName + "." + std::to_string(specializedCounter++);
        func::FuncOp clone = innerCtrlFunc.clone();
        clone.setName(specializedName);
        clone.setPrivate();
        // Insert constants at the top of the entry block and replace uses.
        OpBuilder b(ctx);
        b.setInsertionPointToStart(&clone.front());
        for (auto [argIdx, constOp] : toSpecializeArgs) {
          auto *newConst = b.clone(*constOp);
          clone.front().getArgument(argIdx).replaceAllUsesWith(
              newConst->getResult(0));
        }
        for (auto [elemIdx, constOp] : toSpecializeClosure) {
          if (Value v = selfClosureElement(clone, 1, elemIdx)) {
            auto *newConst = b.clone(*constOp);
            v.replaceAllUsesWith(newConst->getResult(0));
          }
        }
        module.push_back(clone);
        innerCtrlName = specializedName;
        innerCtrlAttr = SymbolRefAttr::get(ctx, specializedName);
      }
    }
  }

  // The closure layout: [original_closure_data_types..., ctrl_types...]
  //
  // For the indirect case the trampoline's original free-variable captures come
  // first (matching the existing cc.callable_closure result order); then ctrl
  // refs are appended.  For the direct case there are no original captures, so
  // the closure contains only ctrl refs.
  SmallVector<Type> closureTypes;
  if (origInstan)
    for (Value d : origInstan.getClosureData())
      closureTypes.push_back(d.getType());
  for (Value c : ctrlRefs)
    closureTypes.push_back(c.getType());

  // Wrapper name is uniqued per (original callee, adjoint-ness, number of
  // controls, and each control's ref-vs-veq kind). Adjoint-ness must be part
  // of the key: otherwise the plain control-of-callee wrapper and the
  // control-of-adjoint-of-callee wrapper (built for a different apply of the
  // same callee) collide, and the module.lookupSymbol cache check below
  // silently reuses whichever was built first for both cases. Likewise, the
  // per-control ref-vs-veq kind must be part of the key: two call sites can
  // agree on callee, adjoint-ness, and control count while one passes a bare
  // `!quake.ref` control and the other a `!quake.veq<?>` (e.g. adjoint(S)
  // controlled directly vs. adjoint(control(S)), where the control is
  // threaded through an extra level of closure and arrives as a veq).
  std::string ctrlKindSig;
  for (Value c : ctrlRefs)
    ctrlKindSig += isa<cudaq::quake::VeqType>(c.getType()) ? 'v' : 'r';
  auto wrapperName =
      calleeOrigName.str() + closureNameInfix.str() + ctrlKindSig;
  auto wrapperAttr = SymbolRefAttr::get(ctx, wrapperName);

  if (!module.lookupSymbol<func::FuncOp>(wrapperName)) {
    // Wrapper signature: (!cc.callable<origSig>, original_args...) -> results
    FunctionType origFnSig = origSigTy.getSignature();
    SmallVector<Type> wrapperInTys = {origSigTy};
    wrapperInTys.append(origFnSig.getInputs().begin(),
                        origFnSig.getInputs().end());
    auto wrapperFnTy =
        FunctionType::get(ctx, wrapperInTys, origFnSig.getResults());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    auto wrapperFunc =
        func::FuncOp::create(rewriter, loc, wrapperName, wrapperFnTy);
    wrapperFunc.setPrivate();

    SmallVector<Location> argLocs(wrapperInTys.size(), loc);
    Block *entry = rewriter.createBlock(&wrapperFunc.getBody(),
                                        wrapperFunc.getBody().begin(),
                                        wrapperInTys, argLocs);
    rewriter.setInsertionPointToStart(entry);

    Value selfArg = entry->getArgument(0);

    // Unpack (original_closure_data..., ctrl_refs...)
    auto extractOp = cudaq::cc::CallableClosureOp::create(
        rewriter, loc, closureTypes, selfArg);

    unsigned numOrig = origInstan ? origInstan.getClosureData().size() : 0;
    unsigned numCtrls = ctrlRefs.size();
    SmallVector<Value> origClosureData;
    for (unsigned i = 0; i < numOrig; ++i)
      origClosureData.push_back(extractOp.getResult(i));

    SmallVector<Value> extractedCtrls;
    for (unsigned i = numOrig, n = numOrig + numCtrls; i < n; ++i)
      extractedCtrls.push_back(extractOp.getResult(i));

    // Build the control veq mirroring the original trampoline.
    Value ctrlVeq = cudaq::quake::ConcatOp::create(rewriter, loc, unsizedVeqTy,
                                                   extractedCtrls);

    // Call @lifted_lambda.ctrl(%veq, closure_data..., formal_args...).
    //
    // This is exactly the original trampoline's func.call with the veq
    // prepended and the callee switched to the ctrl variant.
    SmallVector<Value> callArgs = {ctrlVeq};
    // When the target `.ctrl` variant keeps a "self" callable as its own
    // leading formal argument (see `needsSelfArg` above), synthesize a
    // placeholder dynamic instantiation to fill it, mirroring how
    // FoldCallable created the original one. It must carry origClosureData,
    // not an empty closure: for the direct-callee case origClosureData is
    // already empty (numOrig == 0, nothing was ever captured), but for an
    // indirect applicative closure (isApplicativeClosure above)
    // origClosureData holds exactly what the original closure captured (e.g.
    // a free variable like an angle) - data the target's own body still
    // extracts via cc.callable_closure. Passing an empty closure there
    // instead silently drops those captured values, producing a self
    // instantiation whose actual operand count/types don't match what the
    // target's body expects to unpack from it.
    // innerCtrlFunc may be null here: unlike the indirect-callee call site
    // (which only calls into this function after confirming the target
    // `.ctrl` variant exists), the direct-callee call site has no such
    // guard, e.g. when aggressive-inlining already absorbed all of
    // calleeOrigName's quantum work into some other function, leaving no
    // `.ctrl` variant to find. needsSelfArg is already false in that case
    // (see above), so this stays safe without redundantly recomputing
    // dynamicArgType on a null function type.
    if (needsSelfArg) {
      if (Value selfArg = maybeBuildCtrlSelfArg(rewriter, loc,
                                                innerCtrlFunc.getFunctionType(),
                                                innerCtrlAttr, origClosureData))
        callArgs.push_back(selfArg);
    }

    // See isApplicativeClosure above: in the applicative case, origClosureData
    // and the wrapper's own trailing entry arguments carry the same values,
    // so only the latter (simpler - no closure-unpack indirection needed) is
    // forwarded.
    if (!isApplicativeClosure)
      callArgs.append(origClosureData.begin(), origClosureData.end());

    // entry's own formal args mirror calleeOrigName's *original*, uncontrolled
    // signature (origFnSig) - entry->getArgument(1) is that signature's own
    // position-0 parameter. In the direct-callee case (origInstan null),
    // origFnSig *is* the inner target's own signature, so when the target is
    // itself dynamic (needsSelfArg), that first argument is the
    // same self/closure slot already supplied above via selfInstan: skip it
    // here or it gets forwarded a second time as an extra, unwanted operand.
    // (Not applicable when origInstan is non-null: there, origFnSig describes
    // the *outer* closure's own external signature, unrelated to whatever
    // shape the inner target's own parameters happen to have.)
    unsigned formalArgsStart = (!origInstan && needsSelfArg) ? 2 : 1;
    for (unsigned i = formalArgsStart, n = entry->getNumArguments(); i < n; ++i)
      callArgs.push_back(entry->getArgument(i));

    func::CallOp::create(rewriter, loc, origFnSig.getResults(), innerCtrlAttr,
                         callArgs);
    func::ReturnOp::create(rewriter, loc, ValueRange{});
  }

  // New instantiate_callable: same external type, richer closure.
  // Layout: [original_closure_data..., ctrl_refs...]
  SmallVector<Value> newClosure;
  if (origInstan)
    newClosure.append(origInstan.getClosureData().begin(),
                      origInstan.getClosureData().end());
  for (Value c : ctrlRefs)
    newClosure.push_back(c);

  auto result = cudaq::cc::InstantiateCallableOp::create(
      rewriter, loc, origSigTy, wrapperAttr, newClosure);
  return {result, wrapperAttr};
}

/// A `cc.loop` and the position of the value in the loop's iteration arguments.
/// (loop-carried value)
using LoopCarriedSlot = std::pair<Operation *, unsigned>;

/// Return the slot use passes its value into, if passing it along is all the
/// use does (a loop's initial argument, or a region terminator forwarding it
/// around the loop).
static std::optional<LoopCarriedSlot> forwardedSlot(OpOperand &use) {
  Operation *user = use.getOwner();
  unsigned pos = use.getOperandNumber();
  if (auto loop = dyn_cast<cudaq::cc::LoopOp>(user))
    return LoopCarriedSlot{loop.getOperation(), pos};
  auto loop = dyn_cast_or_null<cudaq::cc::LoopOp>(user->getParentOp());
  if (!loop)
    return std::nullopt;
  if (isa<cudaq::cc::ConditionOp>(user)) {
    if (pos == 0)
      return std::nullopt;
    return LoopCarriedSlot{loop.getOperation(), pos - 1};
  }
  if (isa<cudaq::cc::ContinueOp, cudaq::cc::BreakOp>(user))
    return LoopCarriedSlot{loop.getOperation(), pos};
  return std::nullopt;
}

/// Collect every value occupying slot pos of loop (the loop's result and
/// the matching entry block argument of each of its regions).
static SmallVector<Value> valuesInSlot(cudaq::cc::LoopOp loop, unsigned pos) {
  SmallVector<Value> values;
  if (pos < loop.getNumResults())
    values.push_back(loop.getResult(pos));
  for (auto *region : loop.getRegions()) {
    if (region->empty())
      continue;
    Block &entry = region->front();
    if (pos < entry.getNumArguments())
      values.push_back(entry.getArgument(pos));
  }
  return values;
}

namespace {
/// Replace a quake.apply op with a call to the correct variant function.
struct ApplyOpPattern : public OpRewritePattern<cudaq::quake::ApplyOp> {
  using Base = OpRewritePattern<cudaq::quake::ApplyOp>;

  explicit ApplyOpPattern(MLIRContext *ctx, bool constProp)
      : Base(ctx), constProp(constProp) {}

  // Counter for uniquely naming constant-specialized ctrl variants.
  mutable unsigned specializedCounter = 0;

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
    auto calleeFn =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(apply, calleeAttr);
    if (!calleeFn)
      return failure();
    auto unsizedVeqTy = cudaq::quake::VeqType::getUnsized(ctx);
    const bool addControls = !apply.getControls().empty();
    // Name pieces for any ctrl-closure wrapper this apply may need (see
    // buildCtrlClosureInstantiation): must reflect this apply's own
    // adjoint-ness so a wrapper for the plain control of some callee never
    // collides with (or gets mistaken for) the wrapper for the control of
    // that callee's adjoint.
    StringRef ctrlNameSuffix = apply.getIsAdj() ? ".adj.ctrl" : ".ctrl";
    StringRef closureNameInfix =
        apply.getIsAdj() ? ".adj_ctrl_closure" : ".ctrl_closure";

    // Track wire controls so we can recover them after the call.
    SmallVector<Value> wrappedCtrlRefs;

    auto loc = apply.getLoc();
    auto wireTy = cudaq::quake::WireType::get(ctx);
    auto refTy = cudaq::quake::RefType::get(ctx);

    // Build the control operand list (as refs) and the pending veq
    // concat. We defer adding the veq to newArgs because the callable-handling
    // branch may choose to capture the controls in the closure instead.
    SmallVector<Value> ctrlRefOperands;
    Value pendingCtrlVeq;
    if (addControls) {
      for (Value ctrl : apply.getControls()) {
        if (isa<cudaq::quake::WireType>(ctrl.getType())) {
          auto refVal =
              cudaq::quake::WrapNewOp::create(rewriter, loc, refTy, ctrl);
          wrappedCtrlRefs.push_back(refVal);
          ctrlRefOperands.push_back(refVal);
        } else if (isa<cudaq::quake::ControlType>(ctrl.getType())) {
          auto wireVal =
              cudaq::quake::FromControlOp::create(rewriter, loc, wireTy, ctrl);
          auto refVal =
              cudaq::quake::WrapNewOp::create(rewriter, loc, refTy, wireVal);
          ctrlRefOperands.push_back(refVal);
        } else {
          ctrlRefOperands.push_back(ctrl);
        }
      }
      pendingCtrlVeq = cudaq::quake::ConcatOp::create(
          rewriter, loc, unsizedVeqTy, ctrlRefOperands);
    }

    // newArgs is built after the actuals loop (see below).
    SmallVector<Value> newArgs;
    bool ctrlsCapturedInClosure = false;

    SmallVector<Value> applyActuals{apply.getActuals().begin(),
                                    apply.getActuals().end()};
    // The first actual may be a closure if this apply is calling a callable.
    // When controls are present, the ctrl-closure approach captures the control
    // refs in the closure instead of prepending a veq to the callable's type.

    // Track wire/cable actuals so we can recover them after the call.
    // Each entry is (argIndex, refValue [or veqValue], numWires).
    struct LinearActualInfo {
      unsigned argIndex; // index into applyActuals
      Value refOrVeq;    // the ref/veq value that wraps the linear actual
      unsigned numWires; // 1 for wire, N for cable<N>
    };
    SmallVector<LinearActualInfo> linearActuals;

    for (auto [idx, entry] : llvm::enumerate(
             llvm::zip(applyActuals, calleeSignature.getInputs()))) {
      auto [v, toTy] = entry;
      Value arg = v;

      if (isa<cudaq::quake::WireType>(v.getType()) &&
          isa<cudaq::quake::RefType>(toTy)) {
        // wire actual → ref formal: coercion required.
        // wrap_new produces a fresh ref; unwrap after the call recovers wire.
        auto refVal = cudaq::quake::WrapNewOp::create(rewriter, loc, refTy, v);
        linearActuals.push_back({static_cast<unsigned>(idx), refVal, 1u});
        arg = refVal;
      } else if (auto cableTy = dyn_cast<cudaq::quake::CableType>(v.getType());
                 cableTy && isa<cudaq::quake::VeqType>(toTy)) {
        // cable<N> actual → veq<N>/veq<?> formal: coercion required.
        //   split_cable → wrap_new each wire → concat → relax_size if needed.
        unsigned n = cudaq::quake::getWireCount(v.getType());
        SmallVector<Type> wireTys(n, wireTy);
        auto split =
            cudaq::quake::SplitCableOp::create(rewriter, loc, wireTys, v);
        SmallVector<Value> wrappedRefs;
        for (unsigned i = 0; i < n; ++i)
          wrappedRefs.push_back(cudaq::quake::WrapNewOp::create(
              rewriter, loc, refTy, split.getResult(i)));
        auto sizedVeqTy = cudaq::quake::VeqType::get(ctx, n);
        Value veqVal = cudaq::quake::ConcatOp::create(rewriter, loc, sizedVeqTy,
                                                      wrappedRefs);
        if (toTy == unsizedVeqTy)
          veqVal = cudaq::quake::RelaxSizeOp::create(rewriter, loc,
                                                     unsizedVeqTy, veqVal);
        linearActuals.push_back({static_cast<unsigned>(idx), veqVal, n});
        arg = veqVal;
        // wire→wire or cable→cable: formal already accepts the linear type,
        // no coercion or extra result needed - pass through unchanged.
      } else if (toTy == unsizedVeqTy && arg.getType() != toTy) {
        arg = cudaq::quake::ConcatOp::create(rewriter, loc, unsizedVeqTy, arg);
      } else if (isa<cudaq::cc::CallableType>(toTy) && arg.getType() == toTy) {
        if (auto instan =
                arg.getDefiningOp<cudaq::cc::InstantiateCallableOp>()) {
          if (addControls) {
            auto module = apply->getParentOfType<ModuleOp>();
            // Only redirect through a ctrl-closure wrapper if the closure's
            // real target actually has a `.ctrl` variant to call. A target with
            // no `.ctrl` variant has nothing left to control: e.g.
            // aggressive-inlining may have already absorbed all of its quantum
            // work into some other function, leaving this instantiate_callable
            // referencing a now fully dead symbol that nothing calls anymore.
            // Building a wrapper that calls a `.ctrl` variant that was never
            // (and will never be) built would just crash; leave the closure as
            // an ordinary, unspecialized instantiation instead.
            //
            // This is the single path for both a distinct closure (some other
            // kernel's thunk) and an applicative closure - one that applies
            // THIS apply's own callee to itself (see FoldCallable, which builds
            // such applicative actuals when folding an indirect call through a
            // callable formal argument back to a direct one) - both always go
            // through a wrapper thunk here rather than being special-cased into
            // a direct `.ctrl` call, so there is exactly one place threading
            // closure data / control refs into a `.ctrl` call correctly (see
            // buildCtrlClosureInstantiation's isApplicativeClosure
            // handling), and constant-prop applies uniformly to both.
            StringRef targetName =
                instan.getCallee().getRootReference().getValue();
            auto targetCtrlName = (targetName + ctrlNameSuffix).str();
            if (module.lookupSymbol<func::FuncOp>(targetCtrlName)) {
              cudaq::cc::CallableType sigTy = instan.getSignature().getType();
              // Ctrl-closure approach: capture the control refs alongside
              // the original closure data in a new wrapper instantiation.
              // A new thunk extracts them, builds the veq, and calls
              // @callee.ctrl. The callable's external type is unchanged.
              // Formal actuals excluding the hidden callable arg (index 0).
              ValueRange formalActuals = ValueRange(applyActuals).drop_front(1);
              auto [wrapperCallable, wrapperAttr] =
                  buildCtrlClosureInstantiation(
                      rewriter, loc, module, instan, calleeOrigName,
                      ctrlRefOperands, sigTy, instan.getClosureData(),
                      formalActuals, constProp, ctrlNameSuffix,
                      closureNameInfix, specializedCounter, ctx);
              calleeAttr = wrapperAttr;
              arg = wrapperCallable;
              ctrlsCapturedInClosure = true;
            } else {
              cudaq::cc::CallableType sigTy = instan.getSignature().getType();
              arg = cudaq::cc::InstantiateCallableOp::create(
                  rewriter, instan.getLoc(), sigTy, instan.getCalleeAttr(),
                  instan.getClosureData());
            }
          } else {
            cudaq::cc::CallableType sigTy = instan.getSignature().getType();
            arg = cudaq::cc::InstantiateCallableOp::create(
                rewriter, instan.getLoc(), sigTy, calleeAttr,
                instan.getClosureData());
          }
        }
      }
      newArgs.emplace_back(arg);
    }

    // For direct callees with controls that were not already handled by the
    // callable-handling branch, apply the same ctrl-closure approach: create a
    // wrapper thunk that captures the ctrl refs in its closure and calls
    // @callee.ctrl internally.  This is the same helper used for indirect
    // callees, with nullptr for origInstan (no original closure data).
    if (addControls && !ctrlsCapturedInClosure) {
      auto callableTy = cudaq::cc::CallableType::get(calleeSignature);
      auto module = apply->getParentOfType<ModuleOp>();
      auto [wrapperCallable, wrapperAttr] = buildCtrlClosureInstantiation(
          rewriter, loc, module, cudaq::cc::InstantiateCallableOp{},
          calleeOrigName, ctrlRefOperands, callableTy,
          /*callSiteClosureData=*/ValueRange{},
          /*callSiteFormalArgs=*/ValueRange(applyActuals), constProp,
          ctrlNameSuffix, closureNameInfix, specializedCounter, ctx);
      calleeAttr = wrapperAttr;
      newArgs.insert(newArgs.begin(), wrapperCallable);
      ctrlsCapturedInClosure = true;
    }

    // Prepend the control veq only when controls were NOT captured in the
    // closure.  When ctrlsCapturedInClosure is true the wrapper thunk handles
    // the veq construction internally.
    if (addControls && !ctrlsCapturedInClosure)
      newArgs.insert(newArgs.begin(), pendingCtrlVeq);

    // The formal results are whatever the callee returns; the apply's appended
    // linear results are recovered below via unwrap/split.
    TypeRange formalResultTys = calleeSignature.getResults();
    LLVM_DEBUG(llvm::dbgs() << "replacing: " << apply << '\n');

    if (linearActuals.empty()) {
      // Fast path: no wire/cable actuals - behaviour identical to before.
      [[maybe_unused]] auto result = rewriter.replaceOpWithNewOp<func::CallOp>(
          apply, apply.getResultTypes(), calleeAttr, newArgs);
      LLVM_DEBUG(llvm::dbgs() << "with " << result << '\n');
      return success();
    }

    // General path: make the call with ref/veq args, then recover the wires.
    rewriter.setInsertionPoint(apply);
    auto callOp = func::CallOp::create(rewriter, loc, formalResultTys,
                                       calleeAttr, newArgs);

    // Build the sequence of recovered linear values (wire or cable) in the same
    // left-to-right order the apply op appended them to its result list.
    SmallVector<Value> recoveredLinear;
    for (auto &info : linearActuals) {
      if (info.numWires == 1) {
        // ref → wire via unwrap.
        recoveredLinear.push_back(cudaq::quake::UnwrapOp::create(
            rewriter, loc, wireTy, info.refOrVeq));
      } else {
        // veq → individual refs → unwrap each → bundle_cable.
        unsigned n = info.numWires;
        // Recover the sized veq in case we had relaxed to unsized.
        Value veq = info.refOrVeq;
        if (veq.getType() == unsizedVeqTy) {
          // The relax_size input is the sized veq; walk back to find it.
          if (auto relax = veq.getDefiningOp<cudaq::quake::RelaxSizeOp>())
            veq = relax.getInputVec();
        }
        SmallVector<Value> extractedWires;
        for (unsigned i = 0; i < n; ++i) {
          Value ref = cudaq::quake::ExtractRefOp::create(rewriter, loc, veq, i);
          extractedWires.push_back(
              cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, ref));
        }
        auto cableTy = cudaq::quake::CableType::get(ctx, n);
        recoveredLinear.push_back(cudaq::quake::BundleCableOp::create(
            rewriter, loc, cableTy, extractedWires));
      }
    }

    // Recover wire controls: unwrap each wrapped-ref back to its wire.
    // These appear in the result list between the formal results and the
    // coerced-actual linear results (matching the verifier's layout).
    SmallVector<Value> recoveredCtrlWires;
    for (Value refVal : wrappedCtrlRefs)
      recoveredCtrlWires.push_back(
          cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, refVal));

    // Replace all uses of the apply's results:
    //   apply results [0..formalN)               come from the callOp.
    //   apply results [formalN..formalN+ctrlN)   are recovered wire controls.
    //   apply results [formalN+ctrlN..)          are recovered linear actuals.
    SmallVector<Value> allResults(callOp.getResults().begin(),
                                  callOp.getResults().end());
    allResults.append(recoveredCtrlWires.begin(), recoveredCtrlWires.end());
    allResults.append(recoveredLinear.begin(), recoveredLinear.end());
    rewriter.replaceOp(apply, allResults);
    LLVM_DEBUG(llvm::dbgs() << "with " << callOp << '\n');
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
    auto callTy = dynamicArgType(funcTy, 0);
    if (callTy) {
      SmallVector<Type> newInTys = {veqTy};
      newInTys.append(funcTy.getInputs().begin() + 1, funcTy.getInputs().end());
      auto newFnTy =
          FunctionType::get(ctx, newInTys, callTy.getSignature().getResults());
      inTys.push_back(cudaq::cc::CallableType::get(newFnTy));
      inTys.append(funcTy.getInputs().begin() + 1, funcTy.getInputs().end());
    } else {
      inTys.append(funcTy.getInputs().begin(), funcTy.getInputs().end());
    }
    auto newFunc = cudaq::opt::factory::createFunction(
        funcName, funcTy.getResults(), inTys, module);
    newFunc.setPrivate();
    if (auto atomicRegion =
            func->getAttr(cudaq::cc::atomicQuantumRegionAttrName))
      newFunc->setAttr(cudaq::cc::atomicQuantumRegionAttrName, atomicRegion);
    IRMapping mapping;
    func.getBody().cloneInto(&newFunc.getBody(), mapping);
    auto controlNotNeeded = computeActionAnalysis(newFunc);
    // Only the dynamic-callable-self-arg case needs arg0's type rebuilt (to
    // the freshly-built callable type in inTys[1], reflecting the prepended
    // control veq in its signature); the plain case leaves arg0 as-is, and
    // func may have no arguments at all to retype.
    if (callTy)
      newFunc.getBody().front().getArgument(0).setType(inTys[1]);
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
        auto arrAttr = cast<DenseI32ArrayAttr>(
            op->getAttr(cudaq::runtime::operandSegmentSizes));
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
        attrs.set(cudaq::runtime::operandSegmentSizes, newArrAttr);

        if (auto quantumOp = dyn_cast<cudaq::quake::OperatorInterface>(op)) {
          if (auto oldPolarities = quantumOp.getNegatedControls()) {
            SmallVector<bool> newPolarities{
                false}; // control from `ApplyOp` is always positive
            newPolarities.append(oldPolarities->begin(), oldPolarities->end());

            attrs.set("negated_qubit_controls",
                      builder.getDenseBoolArrayAttr(newPolarities));
          }
        }

        OperationState res(op->getLoc(), op->getName().getStringRef(), operands,
                           op->getResultTypes(), attrs);
        // FIXME: Quake quantum gates do have results.
        auto *newOp = builder.create(res);
        op->replaceAllUsesWith(newOp->getResults());
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
        call->replaceAllUsesWith(app->getResults());
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
    if (auto atomicRegion =
            func->getAttr(cudaq::cc::atomicQuantumRegionAttrName))
      newFunc->setAttr(cudaq::cc::atomicQuantumRegionAttrName, atomicRegion);
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
    Value initArg = loop.getInitialArgs()[*loopComponents->induction];
    Type origInductionTy = initArg.getType();

    // The bound (newTermVal), the step (newStepVal), and the induction's own
    // initial value (initArg) are all artifacts pulled from the original
    // function and are not guaranteed to share an integer width: e.g. the
    // induction may be declared i32 while the bound is an i64 computed by
    // comparing a widened copy of the induction against something else (a
    // veq size, say). Do all the iteration-count arithmetic below in a
    // single, common (widest) integer type, promoting narrower values up via
    // a signed cc.cast, then narrow the final new initial value back down to
    // the induction's own type at the end.
    auto widerOf = [](Type a, Type b) {
      return cast<IntegerType>(a).getWidth() >= cast<IntegerType>(b).getWidth()
                 ? a
                 : b;
    };
    Type wideTy = widerOf(widerOf(newTermVal.getType(), newStepVal.getType()),
                          origInductionTy);
    auto promote = [&](Value v) -> Value {
      if (v.getType() == wideTy)
        return v;
      return cudaq::cc::CastOp::create(builder, loc, wideTy, v,
                                       cudaq::cc::CastOpMode::Signed);
    };
    newTermVal = promote(newTermVal);
    newStepVal = promote(newStepVal);
    initArg = promote(initArg);
    Value wideInitialValue = promote(loopComponents->initialValue);

    auto zero = createIntConstant(builder, loc, wideTy, 0);
    if (!stepIsAnAddOp) {
      // Negate the step value when arith.subi.
      newStepVal = arith::SubIOp::create(builder, loc, zero, newStepVal);
    }
    Value iters = arith::SubIOp::create(builder, loc, newTermVal, initArg);
    auto cmpOp = cast<arith::CmpIOp>(loopComponents->compareOp);
    auto pred = cmpOp.getPredicate();
    auto one = createIntConstant(builder, loc, wideTy, 1);
    if (cudaq::opt::isSemiOpenPredicate(pred)) {
      Value negStepCond = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::slt, newStepVal, zero);
      auto negOne = createIntConstant(builder, loc, wideTy, -1);
      Value adj = arith::SelectOp::create(builder, loc, wideTy, negStepCond,
                                          one, negOne);
      iters = arith::AddIOp::create(builder, loc, iters, adj);
    }
    iters = arith::AddIOp::create(builder, loc, iters, newStepVal);
    iters = arith::DivSIOp::create(builder, loc, iters, newStepVal);
    Value noLoopCond = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sgt, iters, zero);
    iters =
        arith::SelectOp::create(builder, loc, wideTy, noLoopCond, iters, zero);
    Value lastIter = arith::SubIOp::create(builder, loc, iters, one);
    Value nStep = arith::MulIOp::create(builder, loc, lastIter, newStepVal);
    Value newInitVal =
        arith::AddIOp::create(builder, loc, wideInitialValue, nStep);
    // `cc.cast signed` is only valid for widening (sext); a truncating
    // narrow back down to the induction's own type must use the plain
    // (mode-less) cast.
    if (newInitVal.getType() != origInductionTy)
      newInitVal =
          cudaq::cc::CastOp::create(builder, loc, origInductionTy, newInitVal);

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
      // Check that classical values do not have data-flow to subsequent ops.
      // `reverseTheOpsInTheBlock` moves every op in `invertedOps` to sit
      // just before `term`, processed last-to-first, so the whole set ends
      // up clustered immediately before `term` in reverse relative order.
      // Every *other* op in the block (classical "glue" — e.g. a running
      // index computed from several loops' results and fed to a later
      // loop's bound) never moves: it simply closes ranks in its original
      // relative order, and that entire group ends up positioned *before*
      // the whole moved cluster — regardless of whether, originally, it sat
      // before or after the op(s) it depends on. So a classical result of
      // any op-to-invert can only remain validly positioned if its sole
      // consumer is `term` itself (which — being the fixed point every move
      // targets — is guaranteed to end up after the entire cluster). Any
      // other consumer, whether another op-to-invert or plain glue, ends up
      // on the wrong side once its producer relocates.
      for (auto *inv : invertedOps) {
        if (inv->getNumResults() == 0)
          continue;
        for (auto res : inv->getResults())
          if (!cudaq::quake::isLinearType(res.getType()))
            for (auto *usr : res.getUsers())
              if (usr != term)
                return usr->emitOpError("control-flow def-use not reversible.");
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
            return newIfOp.emitOpError("then block not reversed.");
        if (failed(invert(newIfOp.getElseRegion())))
          if (!legacyClassical)
            return newIfOp.emitOpError("else block not reversed.");
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
            return newLoopOp.emitOpError("loop not reversed.");
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
            return newScopeOp.emitOpError("scope not reversed.");
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
      auto arrAttr = cast<DenseI32ArrayAttr>(
          op->getAttr(cudaq::runtime::operandSegmentSizes));
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
