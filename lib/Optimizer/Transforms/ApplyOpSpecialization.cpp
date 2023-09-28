/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "apply-op-specialization"

using namespace mlir;

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
      rv = !bit0 & bit1;
      bit0 = bit0 | bit1;
    };
    checkAndSet(needsControlVariant, that.needsControlVariant);
    checkAndSet(needsAdjointVariant, that.needsAdjointVariant);
    checkAndSet(needsAdjointControlVariant, that.needsAdjointControlVariant);
    return rv;
  }
};

/// Map from `func::FuncOp` to the variants to be created.
using ApplyOpAnalysisInfo = DenseMap<Operation *, ApplyVariants>;

/// This analysis scans the IR for `ApplyOp`s to see which ones need to have
/// variants created.
struct ApplyOpAnalysis {
  ApplyOpAnalysis(ModuleOp op) : module(op) {
    performAnalysis(op.getOperation());
  }

  const ApplyOpAnalysisInfo &getAnalysisInfo() const { return infoMap; }

private:
  void performAnalysis(Operation *op) {
    op->walk([&](quake::ApplyOp apply) {
      if (!apply.applyToVariant())
        return;
      ApplyVariants variant;
      auto callee = lookupCallee(apply);
      auto iter = infoMap.find(callee);
      if (iter != infoMap.end())
        variant = iter->second;
      if (apply.getIsAdj() && !apply.getControls().empty())
        variant.needsAdjointControlVariant = true;
      else if (apply.getIsAdj())
        variant.needsAdjointVariant = true;
      else if (!apply.getControls().empty())
        variant.needsControlVariant = true;
      infoMap.insert(std::make_pair(callee.getOperation(), variant));
    });

    // Propagate the transitive closure over the call tree.
    bool changed = true;
    while (changed) {
      changed = false;
      ApplyOpAnalysisInfo cloneMap(infoMap);
      for (auto pr : cloneMap) {
        auto &func = pr.first;
        auto &variant = pr.second;
        func->walk([&](quake::ApplyOp apply) {
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

  func::FuncOp lookupCallee(quake::ApplyOp apply) {
    auto callee = apply.getCallee();
    return module.lookupSymbol<func::FuncOp>(*callee);
  }

  ModuleOp module;
  ApplyOpAnalysisInfo infoMap;
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

static std::string getVariantFunctionName(quake::ApplyOp apply,
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
/// Replace an apply op with a call to the correct variant function.
struct ApplyOpPattern : public OpRewritePattern<quake::ApplyOp> {
  using Base = OpRewritePattern<quake::ApplyOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(quake::ApplyOp apply,
                                PatternRewriter &rewriter) const override {
    auto calleeName = getVariantFunctionName(
        apply, apply.getCallee()->getRootReference().str());
    auto *ctx = apply.getContext();
    auto consTy = quake::VeqType::getUnsized(ctx);
    SmallVector<Value> newArgs;
    if (!apply.getControls().empty()) {
      auto consOp = rewriter.create<quake::ConcatOp>(apply.getLoc(), consTy,
                                                     apply.getControls());
      newArgs.push_back(consOp);
    }
    newArgs.append(apply.getArgs().begin(), apply.getArgs().end());
    rewriter.replaceOpWithNewOp<func::CallOp>(apply, apply.getResultTypes(),
                                              calleeName, newArgs);
    return success();
  }
};

class ApplySpecializationPass
    : public cudaq::opt::ApplySpecializationBase<ApplySpecializationPass> {
public:
  ApplySpecializationPass() = default;
  ApplySpecializationPass(bool b) : optComputeActionOptim(b) {}

  void runOnOperation() override {
    ApplyOpAnalysis analysis(getOperation());
    const auto &applyVariants = analysis.getAnalysisInfo();
    step1(applyVariants);
    step2();
  }

  /// Step 1. Instantiate all the implied variants of functions from all
  /// quake.apply operations that were found.
  void step1(const ApplyOpAnalysisInfo &applyVariants) {
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

      if (variant.needsControlVariant)
        createControlVariantOf(func);
      if (variant.needsAdjointVariant) {
        auto fnName = func.getName().str();
        createAdjointVariantOf(func, getAdjVariantFunctionName(fnName));
      }
      if (variant.needsAdjointControlVariant)
        createAdjointControlVariantOf(func);
    }
  }

  /// Look for quake.compute_action operations or quake.apply triple patterns in
  /// the FuncOp \p func. In these cases, we do not want to add the controls to
  /// the compute and uncompute functions.
  DenseSet<Operation *> computeActionAnalysis(func::FuncOp func) {
    DenseSet<Operation *> controlNotNeeded;
    if (getComputeActionOptimization()) {
      func->walk([&](Operation *op) {
        if (auto compAct = dyn_cast<quake::ComputeActionOp>(op)) {
          // This is clearly a compute action. Mark the compute side.
          if (auto *defOp = compAct.getCompute().getDefiningOp()) {
            controlNotNeeded.insert(defOp);
          } else {
            compAct.emitError("compute value not determined");
            signalPassFailure();
          }
        } else if (auto app0 = dyn_cast<quake::ApplyOp>(op)) {
          auto next1 = ++app0->getIterator();
          Operation &op1 = *next1;
          if (auto app1 = dyn_cast<quake::ApplyOp>(op1)) {
            auto next2 = ++next1;
            Operation &op2 = *next2;
            if (auto app2 = dyn_cast<quake::ApplyOp>(op2);
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

  func::FuncOp createControlVariantOf(func::FuncOp func) {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();
    // Perform a pre-analysis to determine if func has any compute_action like
    // ops. If it does, then there is an exception case. Instead of applying the
    // controls to the compute kernel, just use the compute kernel (and
    // uncompute kernel) without the controls added.
    auto funcName = getCtrlVariantFunctionName(func.getName().str());
    auto funcTy = func.getFunctionType();
    auto veqTy = quake::VeqType::getUnsized(ctx);
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
        auto arrAttr = op->getAttr(segmentSizes).cast<DenseI32ArrayAttr>();
        SmallVector<Value> operands(op->getOperands().begin(),
                                    op->getOperands().begin() + arrAttr[0]);
        operands.push_back(newCond);
        operands.append(op->getOperands().begin() + arrAttr[0],
                        op->getOperands().end());
        auto newArrAttr = DenseI32ArrayAttr::get(
            ctx, {arrAttr[0], arrAttr[1] + 1, arrAttr[2]});
        NamedAttrList attrs(op->getAttrs());
        attrs.set(segmentSizes, newArrAttr);
        OperationState res(op->getLoc(), op->getName().getStringRef(), operands,
                           op->getResultTypes(), attrs);
        builder.create(res); // Quake quantum gates have no results
        op->erase();
      } else if (auto apply = dyn_cast<quake::ApplyOp>(op)) {
        // If op is an apply and in the set `controlNotNeeded`, then skip it.
        if (controlNotNeeded.count(apply))
          return;
        SmallVector<Value> newControls = {newCond};
        newControls.append(apply.getControls().begin(),
                           apply.getControls().end());
        auto newApply = builder.create<quake::ApplyOp>(
            apply.getLoc(), apply.getResultTypes(), apply.getCalleeAttr(),
            apply.getIsAdjAttr(), newControls, apply.getArgs());
        apply->replaceAllUsesWith(newApply.getResults());
        apply->erase();
      }
    });
    return newFunc;
  }

  /// The adjoint variant of the function is the "reverse" computation. We want
  /// to reverse the flow graph so the gates appear "upside down".
  func::FuncOp createAdjointVariantOf(func::FuncOp func,
                                      std::string &&funcName) {
    ModuleOp module = getOperation();
    auto loc = func.getLoc();
    auto &funcBody = func.getBody();

    // Check our restrictions.
    if (regionHasUnstructuredControlFlow(funcBody)) {
      emitError(loc,
                "cannot make adjoint of kernel with unstructured control flow");
      signalPassFailure();
      return {};
    }
    if (cudaq::opt::hasCallOp(func)) {
      emitError(loc, "cannot make adjoint of kernel with calls");
      signalPassFailure();
      return {};
    }
    if (cudaq::opt::internal::hasCharacteristic(
            [](Operation &op) {
              return isa<cudaq::cc::CreateLambdaOp,
                         cudaq::cc::InstantiateCallableOp>(op);
            },
            *func.getOperation())) {
      emitError(loc, "cannot make adjoint of kernel with callable expressions");
      signalPassFailure();
      return {};
    }
    if (cudaq::opt::hasMeasureOp(func)) {
      emitError(loc, "cannot make adjoint of kernel with a measurement");
      signalPassFailure();
      return {};
    }

    auto funcTy = func.getFunctionType();
    auto newFunc = cudaq::opt::factory::createFunction(
        funcName, funcTy.getResults(), funcTy.getInputs(), module);
    newFunc.setPrivate();
    IRMapping mapping;
    funcBody.cloneInto(&newFunc.getBody(), mapping);
    reverseTheOpsInTheBlock(loc, newFunc.getBody().front().getTerminator(),
                            getOpsToInvert(newFunc.getBody().front()));
    return newFunc;
  }

  static SmallVector<Operation *> getOpsToInvert(Block &block) {
    SmallVector<Operation *> ops;
    for (auto &op : block)
      if (cudaq::opt::hasQuantum(op))
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
    return builder.create<arith::ConstantOp>(loc, attr, ty);
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
      newStepVal = builder.create<arith::SubIOp>(loc, zero, newStepVal);
    }
    Value iters = builder.create<arith::SubIOp>(
        loc, newTermVal, loop.getInitialArgs()[loopComponents->induction]);
    auto cmpOp = cast<arith::CmpIOp>(loopComponents->compareOp);
    auto pred = cmpOp.getPredicate();
    auto one = createIntConstant(builder, loc, iters.getType(), 1);
    if (cudaq::opt::isSemiOpenPredicate(pred)) {
      Value negStepCond = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, newStepVal, zero);
      auto negOne = createIntConstant(builder, loc, iters.getType(), -1);
      Value adj = builder.create<arith::SelectOp>(loc, iters.getType(),
                                                  negStepCond, one, negOne);
      iters = builder.create<arith::AddIOp>(loc, iters, adj);
    }
    iters = builder.create<arith::AddIOp>(loc, iters, newStepVal);
    iters = builder.create<arith::DivSIOp>(loc, iters, newStepVal);
    Value noLoopCond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, iters, zero);
    iters = builder.create<arith::SelectOp>(loc, iters.getType(), noLoopCond,
                                            iters, zero);
    Value lastIter = builder.create<arith::SubIOp>(loc, iters, one);
    Value nStep = builder.create<arith::MulIOp>(loc, lastIter, newStepVal);
    Value newInitVal =
        builder.create<arith::AddIOp>(loc, loopComponents->initialValue, nStep);

    // Create the list of input arguments to loop. We're going to add an
    // argument to the end that is the number of iterations left to execute.
    SmallVector<Value> inputs = loop.getInitialArgs();
    assert(loopComponents->induction < inputs.size());
    inputs[loopComponents->induction] = newInitVal;
    inputs.push_back(iters);

    // Create the new LoopOp. This requires threading the new value that is the
    // number of iterations left to execute. In the whileRegion, update the
    // condition test to use the new argument. In the bodyRegion, update to pass
    // through the new argument. In the stepRegion, decrement the new argument
    // by 1 and convert the original step expression to be a negative step.
    IRRewriter rewriter(builder);
    return rewriter.create<cudaq::cc::LoopOp>(
        loc, ValueRange{inputs}.getTypes(), inputs, /*postCondition=*/false,
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
          auto newCond = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sgt, trip, zero);
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
          auto *stepOp = contOp.getOperand(0).getDefiningOp();
          auto newBump = [&]() -> Value {
            if (stepIsAnAddOp)
              return rewriter.create<arith::SubIOp>(
                  loc, stepOp->getOperand(commuteTheAddOp ? 1 : 0),
                  stepOp->getOperand(commuteTheAddOp ? 0 : 1));
            return rewriter.create<arith::AddIOp>(loc, stepOp->getOperands());
          }();
          args[loopComponents->induction] = newBump;
          auto one = createIntConstant(rewriter, loc, iters.getType(), 1);
          args.push_back(rewriter.create<arith::SubIOp>(
              loc, entry.getArguments().back(), one));
          rewriter.replaceOpWithNewOp<cudaq::cc::ContinueOp>(contOp, args);
        });
  }

  /// For each Op in \p invertedOps, visit them in reverse order and move each
  /// to just in front of \p term (the end of the function). This reversal of
  /// the order of quantum operations is done recursively.
  static void reverseTheOpsInTheBlock(Location loc, Operation *term,
                                      SmallVector<Operation *> &&invertedOps) {
    OpBuilder builder(term);
    for (auto *op : llvm::reverse(invertedOps)) {
      auto invert = [&](Region &reg) {
        if (reg.empty())
          return;
        auto &block = reg.front();
        reverseTheOpsInTheBlock(loc, block.getTerminator(),
                                getOpsToInvert(block));
      };
      if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "moving if: " << ifOp << ".\n");
        auto *newIf = builder.clone(*op);
        op->replaceAllUsesWith(newIf);
        op->erase();
        auto newIfOp = cast<cudaq::cc::IfOp>(newIf);
        invert(newIfOp.getThenRegion());
        invert(newIfOp.getElseRegion());
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "moving for: " << forOp << ".\n");
        TODO_loc(loc, "cannot make adjoint of kernel with scf.for");
        // should we convert to cc.loop and use code below?
      }
      if (auto loopOp = dyn_cast<cudaq::cc::LoopOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "moving loop: " << loopOp << ".\n");
        auto newLoopOp = cloneReversedLoop(builder, loopOp);
        LLVM_DEBUG(llvm::dbgs() << "  to: " << newLoopOp << ".\n");
        op->replaceAllUsesWith(newLoopOp->getResults().drop_back());
        op->erase();
        invert(newLoopOp.getBodyRegion());
        continue;
      }
      if (auto scopeOp = dyn_cast<cudaq::cc::ScopeOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "moving scope: " << scopeOp << ".\n");
        auto *newScope = builder.clone(*op);
        op->replaceAllUsesWith(newScope);
        op->erase();
        auto newScopeOp = cast<cudaq::cc::ScopeOp>(newScope);
        invert(newScopeOp.getInitRegion());
        continue;
      }

      bool opWasNegated = false;
      IRMapping mapper;
      LLVM_DEBUG(llvm::dbgs() << "moving quantum op: " << *op << ".\n");
      auto arrAttr = op->getAttr(segmentSizes).cast<DenseI32ArrayAttr>();
      // Walk over any floating-point parameters to `op` and negate them.
      for (auto iter = op->getOperands().begin(),
                endIter = op->getOperands().begin() + arrAttr[0];
           iter != endIter; ++iter) {
        Value val = *iter;
        Value neg = builder.create<arith::NegFOp>(loc, val.getType(), val);
        mapper.map(val, neg);
        opWasNegated = true;
      }

      // If this is a quantum op that is not self adjoint, we need
      // to adjoint it.
      if (auto quantumOp = dyn_cast_or_null<quake::OperatorInterface>(op);
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
  }

  /// This is the combination of adjoint and control transformations. We will
  /// create a control variant here, even if it wasn't needed to simplify
  /// things. The dead variant can be eliminated as unreferenced.
  func::FuncOp createAdjointControlVariantOf(func::FuncOp func) {
    ModuleOp module = getOperation();
    auto funcName = func.getName().str();
    auto ctrlFuncName = getCtrlVariantFunctionName(funcName);
    auto ctrlFunc = module.lookupSymbol<func::FuncOp>(ctrlFuncName);
    if (!ctrlFunc)
      ctrlFunc = createControlVariantOf(func);

    auto newFuncName = getAdjCtrlVariantFunctionName(funcName);
    return createAdjointVariantOf(ctrlFunc, std::move(newFuncName));
  }

  /// Step 2. Specialize all the quake.apply ops and convert them to calls.
  void step2() {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ApplyOpPattern>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<func::FuncDialect, quake::QuakeDialect>();
    target.addDynamicallyLegalOp<quake::ApplyOp>(
        [](quake::ApplyOp apply) { return apply->hasAttr("replaced"); });
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(module.getLoc(), "could not rewrite all apply ops.");
      signalPassFailure();
    }
  }

  bool getComputeActionOptimization() const {
    if (optComputeActionOptim)
      return *optComputeActionOptim;
    return computeActionOptimization;
  }
  std::optional<bool> optComputeActionOptim;

  // MLIR dependency: internal name used by tablegen.
  static constexpr char segmentSizes[] = "operand_segment_sizes";
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createApplyOpSpecializationPass() {
  return std::make_unique<ApplySpecializationPass>();
}

std::unique_ptr<mlir::Pass>
cudaq::opt::createApplyOpSpecializationPass(bool computeActionOpt) {
  return std::make_unique<ApplySpecializationPass>(computeActionOpt);
}
