/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// \file
/// Data-flow alternative to unwind-lowering.  Instead of converting
/// structured ops to a primitive CFG, this pass threads flag variables through
/// the existing `cc.scope` / `cc.if` / `cc.loop` structure to carry unwind
/// signals and guards subsequent code behind `cc.if` checks.
///
/// Three unwind ops are handled:
///
///   - `cc.unwind_return`: exits the function. This op is rewritten to set the
///   `dfJump` flag to return and save its arguments in a set of preallocated
///   stack slots for the function.
///
///   - `cc.unwind_break`: exits the nearest enclosing `cc.loop`.  This op is
///   rewritten to set the `dfJump` flag to break and save its arguments in a
///   set of preallocated variables dominating the loop.
///
///   - `cc.unwind_continue`: continues the next iteration of the nearest
///   enclosing `cc.loop`. Shares implemented with `unwind_break` and set
///   `dfJump` to continue.

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_UNWINDBYDATAFLOW
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "unwind-by-dataflow"

using namespace mlir;

namespace {
/// Bit flags for the control-flow to be simulated.
enum class JumpKind { None = 0, Continue = 1, Break = 2, Return = 4 };

template <typename T>
JumpKind getJumpKind(T) {
  return JumpKind::None;
}
template <>
JumpKind getJumpKind(cudaq::cc::UnwindContinueOp) {
  return JumpKind::Continue;
}
template <>
JumpKind getJumpKind(cudaq::cc::UnwindBreakOp) {
  return JumpKind::Break;
}
template <>
JumpKind getJumpKind(cudaq::cc::UnwindReturnOp) {
  return JumpKind::Return;
}

/// Analyze the function.
///
///   - Find all the unwind ops.
///   - Find all the structured control-flow points from each unwind up to the
///   parent (destination) of the control-flow jump. Preserve the order.
class Analysis {
public:
  Analysis(func::FuncOp func) {
    doAnalysis(func);
    workingStack.clear();
  }

  void doAnalysis(func::FuncOp func) {
    workingStack.push_back(func);
    func->walk([&](Operation *op, const WalkStage &stage) {
      if (malformedIR)
        return;

      if (stage.isBeforeAllRegions()) {
        // Preorder steps.
        if (isa<cudaq::cc::IfOp, cudaq::cc::LoopOp, cudaq::cc::CreateLambdaOp,
                cudaq::cc::ScopeOp>(op))
          workingStack.push_back(op);
      }

      if (stage.isAfterAllRegions()) {
        // Postorder steps.
        if (isa<cudaq::cc::IfOp, cudaq::cc::LoopOp, cudaq::cc::CreateLambdaOp,
                cudaq::cc::ScopeOp>(op))
          workingStack.pop_back();
        if (isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
                cudaq::cc::UnwindReturnOp>(op)) {
          unwindOps.insert(op);
          auto scopeSet = getRelevantScopeSet(op);
          unwindStacks.insert({op, std::move(scopeSet)});
        }
      }
    });
  }

  /// Search for the nearest enclosing relevant control-flow structure per the
  /// semantics of what an unwinding goto operation does. See the semantics in
  /// the `tablegen` file. In the event the IR is malformed for some reason (in
  /// particular there is no enclosing loop), sets the malformed IR flag.
  SetVector<Operation *> getRelevantScopeSet(Operation *op) {
    // If this is a return, look for the nearest enclosing function.
    if (auto retOp = dyn_cast<cudaq::cc::UnwindReturnOp>(op)) {
      auto it = llvm::find_if(llvm::reverse(workingStack), [](Operation *op) {
        return isa<cudaq::cc::CreateLambdaOp, func::FuncOp>(op);
      });
      SetVector<Operation *> result;
      std::for_each(workingStack.rbegin(), ++it,
                    [&](Operation *op) { result.insert(op); });
      return result;
    }

    // Otherwise, look for the nearest enclosing loop. If we don't find it, this
    // IR is already malformed and we shouldn't be trying to analyze or
    // transform it.
    auto it = llvm::find_if(llvm::reverse(workingStack), [](Operation *op) {
      return isa<cudaq::cc::LoopOp>(op);
    });
    if (it == workingStack.rend()) {
      malformedIR = true;
      return {};
    }
    SetVector<Operation *> result;
    std::for_each(workingStack.rbegin(), ++it,
                  [&](Operation *op) { result.insert(op); });
    return result;
  }

  bool isMalformed() const { return malformedIR; }

  SmallVector<Operation *> workingStack;
  SetVector<Operation *> unwindOps;
  DenseMap<Operation *, SetVector<Operation *>> unwindStacks;
  bool malformedIR = false;
};

//===----------------------------------------------------------------------===//

class UnwindByDataFlowPass
    : public cudaq::opt::impl::UnwindByDataFlowBase<UnwindByDataFlowPass> {
public:
  using UnwindByDataFlowBase::UnwindByDataFlowBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.getBody().empty())
      return;

    // Check if there is anything to do here.
    bool hasUnwinds = false;
    bool hasCFG = false;
    [[maybe_unused]] auto unused = func.walk([&](Operation *op) {
      if (!hasUnwinds &&
          isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
              cudaq::cc::UnwindReturnOp>(op))
        hasUnwinds = true;
      if (isa<cf::ControlFlowDialect>(op->getDialect())) {
        hasCFG = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    // No uniwnd operations, nothing to process. Or if there is already
    // primitive CFG ops, the IR is not conducive.
    if (!hasUnwinds || hasCFG)
      return;

    // This is a high-level control-flow function with goto operations,
    // potentially across otherwise structured control-flow. We now convert
    // these to data-flow.
    Analysis analysis(func);
    if (analysis.isMalformed()) {
      func.emitWarning(
          "IR is malformed. It cannot be converted to a dataflow form.");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": unwind via dataflow on\n"
                            << func << '\n');
    // At this point, we have all the unwind operations and the entire stack of
    // control-flow operations that those `goto`s traverse. So, we can replace
    // the unwinds with setting the `dfJump` flags and insert exclusionary
    // conditions up the control-flow path. We need to pay heed to whether we've
    // already processed a particular enclosing op, so that we don't rewrite it
    // multiple times.
    auto loc = func.getLoc();
    DenseMap<Operation *, SmallVector<Value>> landingPadMap;
    SmallPtrSet<Operation *, 8> condDomOps;
    SmallPtrSet<Operation *, 8> scopeDone;
    Value dfJump;

    // We visit the unwinds in reverse textual order (bottom-to-top). Each
    // unwind's own scope stack is already innermost-to-outermost. Visiting
    // unwinds bottom-to-top ensures that by the time we process an earlier
    // (more outward) unwind and splice "everything after it" into a new
    // guard `cc.if`, any later unwind in that range has already been fully
    // rewritten into a self-contained subtree. This allows relocating an
    // opaque, already-correct chunk of IR rather than re-deriving anything
    // about it.

    // First, set up the variables and check for invalid IR types. Should this
    // fail, the only changes made were to introduce dead variables into the IR.
    {
      // Add the `dfJump` variable;
      IRRewriter rewriter(func);
      auto i8Ty = rewriter.getI8Type();
      rewriter.setInsertionPoint(&func.getBody().front().front());
      dfJump = cudaq::cc::AllocaOp::create(rewriter, loc, i8Ty);

      for (auto *unwind : analysis.unwindOps)
        if (failed(genDominatingVars(rewriter, unwind, landingPadMap,
                                     *analysis.unwindStacks[unwind].rbegin())))
          return;
    }

    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": dominating variables created\n"
                            << func << '\n');

    // Second, rewrite the IR.
    {
      IRRewriter rewriter(func);
      for (auto *unwind : llvm::reverse(analysis.unwindOps)) {
        replaceOpWithSetDFJumpAndIf(rewriter, unwind, dfJump,
                                    landingPadMap[unwind]);
        ArrayRef<Operation *> stack{
            std::next(analysis.unwindStacks[unwind].begin()),
            analysis.unwindStacks[unwind].end()};
        Operation *fromHere = analysis.unwindStacks[unwind].front();
        for (auto *scope : stack) {
          insertPostDominantIfs(rewriter, scope, fromHere, dfJump,
                                landingPadMap, condDomOps, scopeDone);
          fromHere = scope;
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": dataflow paths created\n"
                            << func << '\n');

    // Finally, erase the unwind operations.
    for (auto *unwind : analysis.unwindOps)
      unwind->erase();
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": final result\n"
                            << func << '\n');
  }

  LogicalResult
  genDominatingVars(IRRewriter &rewriter, Operation *unwind,
                    DenseMap<Operation *, SmallVector<Value>> &landingPadMap,
                    Operation *terminus) {
    if (landingPadMap.contains(terminus))
      return success();
    auto loc = unwind->getLoc();
    if (isa<cudaq::cc::CreateLambdaOp, func::FuncOp>(terminus)) {
      rewriter.setInsertionPoint(&terminus->getRegion(0).front().front());
    } else {
      rewriter.setInsertionPoint(terminus);
    }

    // Walk the arguments of unwind, create variables, and copy them into
    // landingPadMap.
    SmallVector<Value> vars;
    for (Value v : unwind->getOperands()) {
      if (cudaq::quake::isQuakeType(v.getType())) {
        unwind->emitWarning("unsupported types");
        return failure();
      }
      vars.push_back(cudaq::cc::AllocaOp::create(rewriter, loc, v.getType()));
    }
    landingPadMap.insert({unwind, std::move(vars)});
    return success();
  }

  template <bool noAdvance = false, JumpKind jk = JumpKind::None>
  Value placeDominatedUnderGuard(IRRewriter &rewriter, Location loc,
                                 Value dfJump, Operation *unwind,
                                 Value yieldValue = nullptr) {
    auto first = [&]() {
      if constexpr (noAdvance) {
        return unwind->getIterator();
      } else {
        return std::next(unwind->getIterator());
      }
    }();
    Block *source = unwind->getBlock();
    const bool hasTerm = source->mightHaveTerminator();
    auto last =
        hasTerm ? source->getTerminator()->getIterator() : source->end();

    // Insert the guard where the guarded code used to start, not before
    // `unwind` itself.
    rewriter.setInsertionPoint(source, first);
    auto jumpVal = cudaq::cc::LoadOp::create(rewriter, loc, dfJump);
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, 8);
    Value cond = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                       jumpVal, zero);
    if constexpr (jk != JumpKind::None) {
      auto one =
          arith::ConstantIntOp::create(rewriter, loc, static_cast<int>(jk), 8);
      auto cond2 = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, jumpVal, one);
      cond = arith::OrIOp::create(rewriter, loc, cond, cond2);
    }

    rewriter.setInsertionPoint(source, last);
    auto savedIP = rewriter.saveInsertionPoint();
    auto thenBuilder = [&](OpBuilder &builder, Location loc, Region &region) {
      region.push_back(new Block{});
      Block &newBlock = region.front();
      newBlock.getOperations().splice(newBlock.end(), source->getOperations(),
                                      first, last);
      builder.setInsertionPointToEnd(&newBlock);
      if (yieldValue)
        cudaq::cc::ContinueOp::create(builder, loc, ValueRange{yieldValue});
      else
        cudaq::cc::ContinueOp::create(builder, loc);
      builder.restoreInsertionPoint(savedIP);
    };
    cudaq::cc::IfOp ifOp;
    if (yieldValue) {
      auto elseBuilder = [&](OpBuilder &builder, Location loc, Region &region) {
        region.push_back(new Block{});
        Block &elseBlock = region.front();
        builder.setInsertionPointToStart(&elseBlock);
        Value falseVal = arith::ConstantIntOp::create(builder, loc, 0, 1);
        cudaq::cc::ContinueOp::create(builder, loc, ValueRange{falseVal});
        builder.restoreInsertionPoint(savedIP);
      };
      ifOp = cudaq::cc::IfOp::create(rewriter, loc,
                                     TypeRange{yieldValue.getType()}, cond,
                                     thenBuilder, elseBuilder);
    } else {
      ifOp = cudaq::cc::IfOp::create(rewriter, loc, TypeRange{}, cond,
                                     thenBuilder);
    }
    if (!hasTerm)
      cudaq::cc::ContinueOp::create(rewriter, loc);
    return yieldValue ? ifOp.getResult(0) : Value();
  }

  template <typename UnwindTy>
  void replaceWithSetDFJumpAndIf(IRRewriter &rewriter, UnwindTy unwind,
                                 Value dfJump, ValueRange vars) {
    rewriter.setInsertionPoint(unwind);
    auto jumpKind = getJumpKind(unwind);
    auto loc = unwind.getLoc();

    // Set the appropriate flag.
    auto jumpConst = arith::ConstantIntOp::create(
        rewriter, loc, static_cast<int>(jumpKind), 8);
    cudaq::cc::StoreOp::create(rewriter, loc, jumpConst, dfJump);

    // Stick the arguments into the variables.
    for (auto [val, var] : llvm::zip(unwind->getOperands(), vars))
      cudaq::cc::StoreOp::create(rewriter, loc, val, var);

    // Place all the dominated code from this point into an `if`
    placeDominatedUnderGuard(rewriter, loc, dfJump, unwind);

    // (Erase the unwind at the end.)
  }

  // Dispatcher boilerplate.
  void replaceOpWithSetDFJumpAndIf(IRRewriter &rewriter, Operation *unwind,
                                   Value dfJump, ValueRange vars) {
    if (auto _break = dyn_cast<cudaq::cc::UnwindBreakOp>(unwind))
      return replaceWithSetDFJumpAndIf(rewriter, _break, dfJump, vars);
    if (auto _continue = dyn_cast<cudaq::cc::UnwindContinueOp>(unwind))
      return replaceWithSetDFJumpAndIf(rewriter, _continue, dfJump, vars);
    if (auto _return = dyn_cast<cudaq::cc::UnwindReturnOp>(unwind))
      return replaceWithSetDFJumpAndIf(rewriter, _return, dfJump, vars);

    unwind->emitError("not an unwind operation?");
    signalPassFailure();
  }

  // Flat scan the func. The return op may be separated from its arguments by
  // a conditional that was inserted. Fix this up by storing the arguments in
  // the if body to the variables, reloading them before the return, and
  // replacing the return operation.
  template <typename RetTy, typename FuncLike>
  void fixupReturns(IRRewriter &rewriter, FuncLike func, ValueRange vars) {
    for (auto &region : func.getRegion())
      for (Operation &op : region) {
        auto ret = dyn_cast<RetTy>(op);
        if (!ret || ret.getOperands().empty())
          continue;
        auto loc = ret.getLoc();
        // Return candidate found.
        for (auto [w, var] : llvm::zip(ret.getOperands(), vars)) {
          // stash the arguments to this return in the variables.
          if (auto *dw = w.getDefiningOp()) {
            rewriter.setInsertionPointAfter(dw);
            cudaq::cc::StoreOp::create(rewriter, loc, w, var);
          } else {
            auto blockArg = cast<BlockArgument>(w);
            auto *owner = blockArg.getOwner();
            rewriter.setInsertionPoint(owner->getTerminator());
            cudaq::cc::StoreOp::create(rewriter, loc, w, var);
          }
        }
        // all values saved away. now we can go back to the return itself,
        // reload the values and replace the old return with a fresh one with
        // the proper values.
        rewriter.setInsertionPoint(ret);
        SmallVector<Value> results;
        for (auto w : vars)
          results.push_back(cudaq::cc::LoadOp::create(rewriter, loc, w));
        rewriter.replaceOpWithNewOp<RetTy>(ret, ValueRange{results});
      }
  }

  // Here scope contains an unwind inside \p unwindFrom. If we haven't already
  // done so, we thread the dfJump flag conditional logic for the dominated
  // range of code from that point.
  void insertPostDominantIfs(
      IRRewriter &rewriter, Operation *scope, Operation *unwindFrom,
      Value dfJump, DenseMap<Operation *, SmallVector<Value>> &landingPadMap,
      SmallPtrSet<Operation *, 8> &done,
      SmallPtrSet<Operation *, 8> &scopeDone) {
    if (done.contains(unwindFrom))
      return;
    auto loc = unwindFrom->getLoc();
    placeDominatedUnderGuard(rewriter, loc, dfJump, unwindFrom);
    done.insert(unwindFrom);

    // The generic guard above must run once per `unwindFrom`, but the
    // scope-specific rewrite below (loop while/step handling, or the return
    // fixup) must run at most once per `scope`, no matter how many distinct
    // unwind sites' stacks pass through it.
    if (!scopeDone.insert(scope).second)
      return;

    // Special handling for cc.loop and function-like \p scope is required.
    if (auto loop = dyn_cast<cudaq::cc::LoopOp>(scope)) {
      // Add the processing of break and continue to our loop. This is the most
      // complicated case.
      // - If the dfJump is continue, then we want to allow the step region (if
      // any) to execute. We also clear the dfJump flag if the while region.
      auto &stepRegion = loop.getStepRegion();
      placeDominatedUnderGuard</*noAdvance=*/true,
                               /*jumpKind=*/JumpKind::Continue>(
          rewriter, loc, dfJump, &stepRegion.front().front());
      // - Otherwise the dfJump is break or a return, then we skip the step
      // region, skip the while region, and update the cc.condition op to exit
      // if dfJump is non-zero.
      auto &whileRegion = loop.getWhileRegion();
      auto *origFront = &whileRegion.front().front();
      auto term =
          cast<cudaq::cc::ConditionOp>(whileRegion.front().getTerminator());
      rewriter.setInsertionPoint(origFront);
      Value jumpVal = cudaq::cc::LoadOp::create(rewriter, loc, dfJump);
      Value one = arith::ConstantIntOp::create(
          rewriter, loc, static_cast<int>(JumpKind::Continue), 8);
      Value cond = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, jumpVal, one);
      // Always clear the `continue` state first, if any.
      {
        auto savedIP = rewriter.saveInsertionPoint();
        cudaq::cc::IfOp::create(
            rewriter, loc, TypeRange{}, cond,
            [&](OpBuilder &builder, Location loc, Region &region) {
              Block *newBlock = new Block{};
              region.push_back(newBlock);
              builder.setInsertionPointToStart(newBlock);
              Value zero = arith::ConstantIntOp::create(
                  rewriter, loc, static_cast<int>(JumpKind::None), 8);
              cudaq::cc::StoreOp::create(builder, loc, zero, dfJump);
              cudaq::cc::ContinueOp::create(builder, loc);
              builder.restoreInsertionPoint(savedIP);
            });
      }

      // Move the body of the while under a guard.
      Value origCond = term.getCondition();
      Value guardedCond = placeDominatedUnderGuard</*noAdvance=*/true>(
          rewriter, loc, dfJump, origFront, origCond);

      Value jumpVal2 = cudaq::cc::LoadOp::create(rewriter, loc, dfJump);
      Value two = arith::ConstantIntOp::create(
          rewriter, loc, static_cast<int>(JumpKind::Break), 8);
      // And clear the `break` state, if any.
      Value cond3 = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, jumpVal2, two);
      {
        auto savedIP = rewriter.saveInsertionPoint();
        cudaq::cc::IfOp::create(
            rewriter, loc, TypeRange{}, cond3,
            [&](OpBuilder &builder, Location loc, Region &region) {
              Block *newBlock = new Block{};
              region.push_back(newBlock);
              builder.setInsertionPointToStart(newBlock);
              Value zero = arith::ConstantIntOp::create(
                  rewriter, loc, static_cast<int>(JumpKind::None), 8);
              cudaq::cc::StoreOp::create(builder, loc, zero, dfJump);
              cudaq::cc::ContinueOp::create(builder, loc);
              builder.restoreInsertionPoint(savedIP);
            });
      }
      // Thread the new condition, computed prior to any loop state clearing.
      rewriter.replaceOpWithNewOp<cudaq::cc::ConditionOp>(term, guardedCond,
                                                          term.getResults());
    } else if (auto func = dyn_cast<cudaq::cc::CreateLambdaOp>(scope)) {
      // Update the cc.return ops that we've now broken.
      fixupReturns<cudaq::cc::ReturnOp>(rewriter, func, landingPadMap[scope]);
    } else if (auto func = dyn_cast<func::FuncOp>(scope)) {
      // Update the func.return ops that we've now broken.
      fixupReturns<func::ReturnOp>(rewriter, func, landingPadMap[scope]);
    }
  }
};

} // namespace
