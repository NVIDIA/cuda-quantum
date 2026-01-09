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

  const ApplyOpAnalysisInfo &getAnalysisInfo() const { return infoMap; }

private:
  void performAnalysis(Operation *op) {
    op->walk([&](quake::ApplyOp apply) {
      if (constProp) {
        // If some of the arguments in getArgs() are constants, then materialize
        // those constants in a clone of the variant. The specialized variant
        // will then be able to perform better constant propagation even if not
        // inlined.
        auto calleeName = apply.getCallee()->getRootReference().str();
        if (func::FuncOp genericFunc =
                module.lookupSymbol<func::FuncOp>(calleeName)) {
          SmallVector<Value> newArgs;
          newArgs.append(apply.getArgs().begin(), apply.getArgs().end());
          IRMapping mapper;
          SmallVector<Value> preservedArgs;
          SmallVector<Type> inputTys;
          SmallVector<arith::ConstantOp> moveConsts;
          bool updateSignature = false;
          for (auto [idx, v] : llvm::enumerate(newArgs)) {
            if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
              auto newConst = c.clone();
              moveConsts.push_back(newConst);
              mapper.map(genericFunc.getArgument(idx), newConst);
              LLVM_DEBUG(llvm::dbgs() << "apply has constant arguments.\n");
            } else {
              if (auto relax = v.getDefiningOp<quake::RelaxSizeOp>()) {
                // Also, specialize any relaxed veq types.
                v = relax.getInputVec();
                updateSignature = true;
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
            }
            newFunc.setPrivate();
            Block &entry = newFunc.front();
            for (auto c : moveConsts)
              entry.push_front(c);
            module.push_back(newFunc);
            OpBuilder builder(apply);
            auto newApply = builder.create<quake::ApplyOp>(
                apply.getLoc(), apply.getResultTypes(),
                SymbolRefAttr::get(ctx, calleeName), apply.getIndirectCallee(),
                apply.getIsAdj(), apply.getControls(), preservedArgs);
            apply->replaceAllUsesWith(newApply.getResults());
            apply->dropAllReferences();
            apply->erase();
            LLVM_DEBUG(llvm::dbgs()
                       << "apply specialization including constant "
                          "propagation of arguments\n"
                       << newFunc << '\n');
            apply = newApply;
          }
        }
      }

      if (!apply.applyToVariant())
        return;
      ApplyVariants variant;
      if (auto callee = lookupCallee(apply)) {
        auto iter = infoMap.find(callee);
        if (iter != infoMap.end())
          variant = iter->second;
        if (apply.getIsAdj() && !apply.getControls().empty())
          variant.needsAdjointControlVariant = true;
        else if (apply.getIsAdj())
          variant.needsAdjointVariant = true;
        else if (!apply.getControls().empty())
          variant.needsControlVariant = true;
        infoMap[callee.getOperation()] = variant;
      }
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

  explicit ApplyOpPattern(MLIRContext *ctx, bool constProp)
      : Base(ctx), constProp(constProp) {}

  LogicalResult matchAndRewrite(quake::ApplyOp apply,
                                PatternRewriter &rewriter) const override {
    std::string calleeOrigName;
    if (apply.getCallee()) {
      calleeOrigName = apply.getCallee()->getRootReference().str();
    } else {
      // Check if the first argument is a func.ConstantOp.
      auto calleeVals = apply.getIndirectCallee();
      if (calleeVals.empty())
        return failure();
      Value calleeVal = calleeVals.front();
      auto fc = calleeVal.getDefiningOp<func::ConstantOp>();
      if (!fc)
        return failure();
      calleeOrigName = fc.getValue().str();
    }
    auto calleeName = getVariantFunctionName(apply, calleeOrigName);
    auto *ctx = apply.getContext();
    auto consTy = quake::VeqType::getUnsized(ctx);
    SmallVector<Value> newArgs;
    if (!apply.getControls().empty()) {
      auto consOp = rewriter.create<quake::ConcatOp>(apply.getLoc(), consTy,
                                                     apply.getControls());
      newArgs.push_back(consOp);
    }
    if (constProp) {
      for (auto v : apply.getArgs()) {
        if (auto c = v.getDefiningOp<arith::ConstantOp>())
          continue;
        newArgs.emplace_back(v);
      }
    } else {
      newArgs.append(apply.getArgs().begin(), apply.getArgs().end());
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(apply, apply.getResultTypes(),
                                              calleeName, newArgs);
    return success();
  }

  const bool constProp;
};

struct FoldCallable : public OpRewritePattern<quake::ApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ApplyOp apply,
                                PatternRewriter &rewriter) const override {
    // If we already know the callee function, there's nothing to do.
    if (apply.getCallee())
      return failure();

    Value ind = apply.getIndirectCallee()[0];
    if (auto callee = ind.getDefiningOp<cudaq::cc::InstantiateCallableOp>()) {
      auto sym = callee.getCallee();
      SmallVector<Value> newArguments = {ind};
      newArguments.append(apply.getArgs().begin(), apply.getArgs().end());
      rewriter.replaceOpWithNewOp<quake::ApplyOp>(
          apply, apply.getResultTypes(), sym, apply.getIsAdj(),
          apply.getControls(), newArguments);
      return success();
    }
    return failure();
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
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();

    ApplyOpAnalysis analysis(module, constantPropagation);
    const auto &applyVariants = analysis.getAnalysisInfo();
    if (succeeded(step1(applyVariants)))
      step2();
  }

  /// Step 1. Instantiate all the implied variants of functions from all
  /// quake.apply operations that were found.
  [[nodiscard]] LogicalResult step1(const ApplyOpAnalysisInfo &applyVariants) {
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
        if (failed(createAdjointVariantOf(func,
                                          getAdjVariantFunctionName(fnName))))
          return failure();
      }
      if (variant.needsAdjointControlVariant)
        if (failed(createAdjointControlVariantOf(func)))
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
    // Helper to check if this is a call to a function taking quantum arguments.
    const auto isQuantumKernelCall = [](Operation *op) -> bool {
      if (auto callOp = dyn_cast<func::CallOp>(op))
        return !quake::getQuantumOperands(op).empty();
      return false;
    };

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
      } else if (isQuantumKernelCall(op)) {
        op->emitError("Unhandled controlled quantum kernel call in control "
                      "variant generation. This could be a result of not "
                      "calling inlining before the apply specialization pass.");
        signalPassFailure();
      }
    });
    return newFunc;
  }

  /// The adjoint variant of the function is the "reverse" computation. We want
  /// to reverse the flow graph so the gates appear "upside down".
  [[nodiscard]] LogicalResult createAdjointVariantOf(func::FuncOp func,
                                                     std::string &&funcName) {
    ModuleOp module = getOperation();
    auto loc = func.getLoc();
    auto &funcBody = func.getBody();

    // Check our restrictions.
    if (regionHasUnstructuredControlFlow(funcBody)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "cannot make adjoint of kernel: unstructured control flow\n");
      return failure();
    }
    if (cudaq::opt::hasCallOp(func)) {
      LLVM_DEBUG(llvm::dbgs() << "cannot make adjoint of kernel with calls\n");
      return failure();
    }
    if (cudaq::opt::internal::hasCharacteristic(
            [](Operation &op) {
              return isa<cudaq::cc::CreateLambdaOp,
                         cudaq::cc::InstantiateCallableOp>(op);
            },
            *func.getOperation())) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "cannot make adjoint of kernel with callable expressions\n");
      return failure();
    }
    if (cudaq::opt::hasMeasureOp(func)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "cannot make adjoint of kernel with a measurement\n");
      return failure();
    }

    auto funcTy = func.getFunctionType();
    auto newFunc = cudaq::opt::factory::createFunction(
        funcName, funcTy.getResults(), funcTy.getInputs(), module);
    newFunc.setPrivate();
    IRMapping mapping;
    funcBody.cloneInto(&newFunc.getBody(), mapping);
    reverseTheOpsInTheBlock(loc, newFunc.getBody().front().getTerminator(),
                            getOpsToInvert(newFunc.getBody().front()));
    return success();
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
  [[nodiscard]] LogicalResult createAdjointControlVariantOf(func::FuncOp func) {
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
    patterns.insert<ApplyOpPattern>(ctx, constantPropagation);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After apply specialization:\n"
                            << module << "\n\n");
  }

  // MLIR dependency: internal name used by tablegen.
  static constexpr char segmentSizes[] = "operand_segment_sizes";
};
} // namespace
