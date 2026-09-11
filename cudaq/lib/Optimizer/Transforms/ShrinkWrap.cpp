/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_SHRINKWRAP
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "shrink-wrap"

using namespace mlir;

/**
   \file

   Structural shrink-wrapping of classical stack allocations.

   See the `ShrinkWrap` description in `Passes.td` for the motivation and the
   relationship to `stack-frame-prealloc` and `variable-coalesce`.

   This pass reasons purely about IR structure, not dataflow: a
   `cc.alloca`'s only legal consumers are direct `cc.load`/`cc.store` and
   `cc.compute_ptr` chains rooted at it. Other uses are considered escaping and
   cancel any transformation from happening. Given that, the set of structured
   control-flow operations an `alloca`'s uses are nested in is exactly the set
   of places its allocation could legally move to; no `mlir::Liveness` or
   `mlir::DominanceInfo` is needed. A position at the front of a region
   structurally containing every use dominates all of them unconditionally.

   Sinking into a `cc.loop`'s body region is safe for the same structural
   reason, with no separate cross-iteration liveness analysis needed: a
   `cc.loop`'s `while`/`step` regions are where the loop's control condition
   and per-iteration update actually live, and those two regions are
   deliberately \e not recognized as valid nesting targets. A variable whose
   address is truly threaded across iterations is already excluded. The address
   itself never appears as a plain `load`/`store`/`compute_ptr` operand in that
   case, so `collectUses` already treats it as an escape. What remains eligible
   in the body is exactly a variable declared and fully consumed within a single
   pass through the body on every iteration, which is exactly what a fresh,
   scope-exited-and-reallocated-per-iteration `cc.scope` inside the body gives
   it.

   The `else` region of `cc.loop` is different from `while`/`step` as it runs at
   most once, which is structurally the same "executes zero or one times" shape
   as a `cc.if` arm, not a per-iteration region. It is therefore recognized as a
   sink target below just like a `cc.if` arm is.
 */

namespace {

// A structurally nested region a `cc.alloca` could legally be sunk into:
// one arm of a `cc.if`, or the body or else region of a `cc.loop`.
struct SinkTarget {
  enum class Kind { IfThen, IfElse, LoopBody, LoopElse };

  Operation *op = nullptr; // a cudaq::cc::IfOp or cudaq::cc::LoopOp
  Kind kind = Kind::IfThen;

  bool operator==(const SinkTarget &other) const {
    return op == other.op && kind == other.kind;
  }

  Region &region() const {
    if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op))
      return kind == Kind::IfThen ? ifOp.getThenRegion() : ifOp.getElseRegion();
    auto loopOp = cast<cudaq::cc::LoopOp>(op);
    return kind == Kind::LoopBody ? loopOp.getBodyRegion()
                                  : loopOp.getElseRegion();
  }
};

// The chain of sink targets structurally containing `op`, from outermost to
// innermost, stopping at (not including) `funcBody`. A `cc.loop`'s
// `while`/`step` regions are intentionally not recognized as targets. The
// `else` region is recognized, the same as a `cc.if` arm.
static SmallVector<SinkTarget> sinkTargetChain(Operation *op,
                                               Region &funcBody) {
  SmallVector<SinkTarget> chain;
  for (Region *region = op->getParentRegion(); region && region != &funcBody;
       region = region->getParentOp()->getParentRegion()) {
    Operation *parent = region->getParentOp();
    if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(parent)) {
      chain.push_back(SinkTarget{ifOp, region == &ifOp.getThenRegion()
                                           ? SinkTarget::Kind::IfThen
                                           : SinkTarget::Kind::IfElse});
    } else if (auto loopOp = dyn_cast<cudaq::cc::LoopOp>(parent)) {
      if (region == &loopOp.getBodyRegion())
        chain.push_back(SinkTarget{loopOp, SinkTarget::Kind::LoopBody});
      else if (region == &loopOp.getElseRegion())
        chain.push_back(SinkTarget{loopOp, SinkTarget::Kind::LoopElse});
    }
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

// Collect every direct/transitive use of `addr` (a `cc.alloca`'s result, or a
// `cc.compute_ptr` rooted at one), following `cc.compute_ptr` chains. Returns
// false if any use is not a plain load/store through `addr` or a further
// `cc.compute_ptr` off of it - i.e. if the address escapes in some way this
// purely structural analysis cannot reason about.
static bool collectUses(Value addr, SmallVectorImpl<Operation *> &uses) {
  for (Operation *user : addr.getUsers()) {
    if (isa<cudaq::cc::LoadOp>(user)) {
      uses.push_back(user);
    } else if (auto store = dyn_cast<cudaq::cc::StoreOp>(user)) {
      // `addr` must be the *pointer* operand; if it is the *value* being
      // stored, the address itself is escaping into memory.
      if (store.getPtrvalue() != addr)
        return false;
      uses.push_back(user);
    } else if (auto computePtr = dyn_cast<cudaq::cc::ComputePtrOp>(user)) {
      uses.push_back(user);
      if (!collectUses(computePtr.getResult(), uses))
        return false;
    } else {
      return false;
    }
  }
  return true;
}

// Move `alloca` into a `cc.scope` covering the entirety of `targetRegion`'s
// existing content: reusing an already-present wrapping `cc.scope` if the
// region is a single block whose only content already is one, otherwise
// relocating the entire region into a freshly built `cc.scope`.
//
// Every exit found is required to be a plain `cc.continue` or, if in a
// `cc.loop` body, `cc.break`. For every case of `cc.continue` the moved CFG
// must rethread through the `cc.scope`'s results and back through the enclosing
// control-flow operation to preserve proper semantics. If the region has a mix
// of `cc.continue` and `cc.break` (which we will discover here), then we fall
// back to converting those control-flow operations to a bit of extra dataflow
// to preserve the semantics and thread the values correctly. This is necessary,
// because `cc.break` is invalid within the body of the new `cc.scope` and
// sufficient because `cc.loop`'s semantics require all `cc.continue` and
// `cc.break` to properly conform to the `cc.loop`'s physical structure. The
// extra `bool` is merely used to select which goto semantics was originally
// present for the block in the body region, a `cc.continue` or a `cc.break`.
static void shrinkWrapInto(cudaq::cc::AllocaOp alloca, Region &targetRegion) {
  MLIRContext *ctx = alloca.getContext();
  OpBuilder builder(ctx);
  Location loc = alloca.getLoc();

  // Fast path: the region is a single block whose only content already is a
  // wrapping cc.scope - just add the alloca to it.
  if (targetRegion.hasOneBlock()) {
    Block &only = targetRegion.front();
    if (!only.empty())
      if (auto s = dyn_cast<cudaq::cc::ScopeOp>(only.front()))
        if (s->getNextNode() == only.getTerminator()) {
          builder.setInsertionPointToStart(&s.getInitRegion().front());
          auto newAlloca = cudaq::cc::AllocaOp::create(
              builder, loc, alloca.getElementType(), alloca.getSeqSize());
          alloca.getResult().replaceAllUsesWith(newAlloca.getResult());
          alloca.erase();
          return;
        }
  }

  // Collect every exit, and bail (leaving the region untouched) unless every
  // one is a plain cc.continue or cc.break.
  SmallVector<Operation *> exits;
  SmallVector<Type> payloadTypes;
  bool hasBreak = false;
  for (Block &b : targetRegion)
    if (b.hasNoSuccessors()) {
      Operation *term = b.getTerminator();
      if (!isa<cudaq::cc::ContinueOp, cudaq::cc::BreakOp>(term))
        return;
      exits.push_back(term);
      hasBreak |= isa<cudaq::cc::BreakOp>(term);
      if (payloadTypes.empty())
        payloadTypes.assign(term->getOperandTypes().begin(),
                            term->getOperandTypes().end());
    }
  if (exits.empty())
    return;

  SmallVector<Type> scopeResultTypes(payloadTypes);
  if (hasBreak)
    scopeResultTypes.push_back(builder.getI1Type());

  SmallVector<Type> carriedTypes(targetRegion.front().getArgumentTypes());

  // Build the new scope in a fresh, not-yet-attached block, so relocating
  // the region's existing blocks into the scope below cannot be circular.
  auto *newEntry = new Block;
  for (Type ty : carriedTypes)
    newEntry->addArgument(ty, loc);
  builder.setInsertionPointToStart(newEntry);
  cudaq::cc::ScopeOp::BodyBuilderFn noBody;
  auto scope =
      cudaq::cc::ScopeOp::create(builder, loc, scopeResultTypes, noBody);
  Region &scopeRegion = scope.getInitRegion();
  Block &placeholder = scopeRegion.front();
  scopeRegion.getBlocks().splice(scopeRegion.end(), targetRegion.getBlocks());
  placeholder.erase();

  Block &scopeEntry = scopeRegion.front();
  for (auto [oldArg, newArg] :
       llvm::zip(scopeEntry.getArguments(), newEntry->getArguments()))
    oldArg.replaceAllUsesWith(newArg);
  while (scopeEntry.getNumArguments() > 0)
    scopeEntry.eraseArgument(scopeEntry.getNumArguments() - 1);

  // Rewrite every exit's terminator: fold it into a `cc.continue`, adding
  // the break/continue discriminant only if some exit needs one.
  for (Operation *term : exits) {
    SmallVector<Value> operands(term->getOperands());
    builder.setInsertionPoint(term);
    if (hasBreak) {
      bool wasBreak = isa<cudaq::cc::BreakOp>(term);
      Value flag = arith::ConstantOp::create(builder, loc,
                                             builder.getBoolAttr(wasBreak));
      operands.push_back(flag);
    }
    cudaq::cc::ContinueOp::create(builder, loc, operands);
    term->erase();
  }

  // Sink the alloca to the front of the (relocated) entry block.
  builder.setInsertionPointToStart(&scopeEntry);
  auto newAlloca = cudaq::cc::AllocaOp::create(
      builder, loc, alloca.getElementType(), alloca.getSeqSize());
  alloca.getResult().replaceAllUsesWith(newAlloca.getResult());
  alloca.erase();

  SmallVector<Value> scopeResults(scope.getResults());
  targetRegion.getBlocks().push_back(newEntry);

  if (!hasBreak) {
    // Every exit agreed on one terminator kind (cc.continue): a plain
    // terminator after the scope suffices, no further reconstruction
    // needed.
    builder.setInsertionPointToEnd(newEntry);
    cudaq::cc::ContinueOp::create(builder, loc, scopeResults);
    return;
  }

  // Rebuild the region as three blocks: the scope, then a cf.cond_br on its
  // trailing i1 picking between a fresh cc.break and a fresh cc.continue.
  Value flagResult = scopeResults.back();
  SmallVector<Value> payload(scopeResults.begin(),
                             std::prev(scopeResults.end()));

  auto *trueBlock = new Block;
  auto *falseBlock = new Block;
  targetRegion.getBlocks().push_back(trueBlock);
  targetRegion.getBlocks().push_back(falseBlock);

  builder.setInsertionPointToEnd(newEntry);
  cf::CondBranchOp::create(builder, loc, flagResult, trueBlock, ValueRange{},
                           falseBlock, ValueRange{});

  builder.setInsertionPointToStart(trueBlock);
  cudaq::cc::BreakOp::create(builder, loc, payload);

  builder.setInsertionPointToStart(falseBlock);
  cudaq::cc::ContinueOp::create(builder, loc, payload);
}

class ShrinkWrapPass : public cudaq::opt::impl::ShrinkWrapBase<ShrinkWrapPass> {
public:
  using ShrinkWrapBase::ShrinkWrapBase;

  void runOnOperation() override {
    auto func = getOperation();
    if (func.getBody().empty())
      return;
    Region &funcBody = func.getBody();
    Block &entryBlock = funcBody.front();

    // Snapshot the candidates first: `shrinkWrapInto` erases the original
    // alloca, which would invalidate an in-flight walk of the same block. A
    // dynamically sized (`seqSize`) alloca is as eligible as a fixed-size one:
    // in the entry block, `seqSize` can only be a function argument or the
    // result of some earlier op in that same entry block, so it necessarily
    // dominates the entire function body - moving the alloca that uses it into
    // any nested scope is exactly as safe as moving one with no size operand at
    // all.
    auto allocaOps = entryBlock.getOps<cudaq::cc::AllocaOp>();
    SmallVector<cudaq::cc::AllocaOp> candidates(allocaOps.begin(),
                                                allocaOps.end());

    for (auto alloca : candidates)
      trySink(alloca, funcBody);
  }

private:
  void trySink(cudaq::cc::AllocaOp alloca, Region &funcBody) {
    SmallVector<Operation *> uses;
    if (!collectUses(alloca.getResult(), uses) || uses.empty())
      return;

    // The target is the deepest sink target (cc.if arm or cc.loop body)
    // structurally containing every use: the longest common prefix of each
    // use's root-to-leaf target chain.
    SmallVector<SinkTarget> common = sinkTargetChain(uses.front(), funcBody);
    for (Operation *use : llvm::drop_begin(uses)) {
      if (common.empty())
        return;
      SmallVector<SinkTarget> chain = sinkTargetChain(use, funcBody);
      size_t n = std::min(common.size(), chain.size());
      size_t i = 0;
      for (; i < n && common[i] == chain[i]; ++i)
        ;
      common.resize(i);
    }
    if (common.empty())
      return;

    SinkTarget target = common.back();
    shrinkWrapInto(alloca, target.region());
  }
};
} // namespace
