/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// The MemToReg pass converts the IR from memory-semantics to
/// register-semantics. This conversion takes values that are stored to and
/// loaded from memory locations (explicitly) to first-class SSA values in
/// virtual registers. It will convert either classical values, quantum values,
/// or (default) both.
///
/// Because memory dereferences are implicit in the Quake dialect (quantum), a
/// conversion to introduce explicit dereferences, conversion to the quantum
/// load/store form (QLS), is required and performed.

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <deque>

namespace cudaq::opt {
#define GEN_PASS_DEF_MEMTOREG
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "memtoreg"

using namespace mlir;

static bool isMemoryAlloc(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op))
    return iface.hasEffect<MemoryEffects::Allocate>();
  return false;
}

static bool isMemoryUse(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op))
    return iface.hasEffect<MemoryEffects::Read>();
  return false;
}

static bool isMemoryDef(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op))
    return iface.hasEffect<MemoryEffects::Write>();
  return false;
}

/// Returns true if and only if \p op is either a callable computation or an
/// inlined macro computation.
static bool isFunctionOp(Operation *op) {
  return isa<func::FuncOp, cudaq::cc::CreateLambdaOp>(op);
}

/// Is \p block immediately owned by a callable/function?
static bool isFunctionBlock(Block *block) {
  return isFunctionOp(block->getParentOp());
}

/// Is \p block both owned by a function and an entry block?
static bool isFunctionEntryBlock(Block *block) {
  return isFunctionBlock(block) && block->isEntryBlock();
}

namespace {
/// Determine the allocations in this scope (a function) whose lifetime is
/// limited to the scope and which do not escape the scope.
struct MemoryAnalysis {
  MemoryAnalysis(func::FuncOp f) { determineAllocSet(f); }

  bool isMember(Operation *op) const { return allocSet.count(op); }

  SmallVector<quake::AllocaOp> getAllQuantumAllocations() const {
    SmallVector<quake::AllocaOp> result;
    for (auto *op : allocSet)
      if (auto qalloc = dyn_cast<quake::AllocaOp>(op))
        result.push_back(qalloc);
    return result;
  }

private:
  void determineAllocSet(func::FuncOp func) {
    SmallVector<Operation *> allocations;
    auto qrefTy = quake::RefType::get(func.getContext());
    func->walk([&](Operation *op) {
      if (isMemoryAlloc(op)) {
        // Make sure this is stack here. Can we make use of an Interface?
        if (auto alloc = dyn_cast<quake::AllocaOp>(op)) {
          if (alloc.getType() == qrefTy)
            allocations.push_back(op);
        } else if (auto alloc = dyn_cast<cudaq::cc::AllocaOp>(op)) {
          if (!alloc.getSeqSize()) {
            LLVM_DEBUG(llvm::dbgs() << "adding: " << alloc << '\n');
            allocations.push_back(op);
          }
        }
      }
    });
    for (auto *a : allocations) {
      auto *add = a;
      for (auto *u : a->getUsers())
        if (!isMemoryUse(u) && !isMemoryDef(u)) {
          add = nullptr;
          break;
        }
      if (add)
        allocSet.insert(add);
    }
  }

  SmallPtrSet<Operation *, 4> allocSet;
};
} // namespace

/// Returns all the exiting blocks in region \p regionNumber of \p op.
static SmallVector<Block *> collectAllExits(Operation *op, int regionNumber) {
  SmallVector<Block *> blocks;
  for (auto &block : op->getRegion(regionNumber))
    if (block.hasNoSuccessors())
      blocks.push_back(&block);
  return blocks;
}

/// Generic traversal over an operation, \p op, that collects all its exit
/// blocks. If the operation does not have regions, an empty deque is returned.
/// Otherwise, the exit blocks are for all regions in \p op are returned.
///
/// For a high-level operation, an exit must be both an exiting block in a
/// region of the operation and the containing region must possibly conclude the
/// operation. (It is possible that some regions within an operation do not exit
/// the operation.)
static std::deque<Block *> collectAllExits(Operation *op) {
  std::deque<Block *> blocks;
  if (auto regionOp = dyn_cast<RegionBranchOpInterface>(op)) {
    SmallPtrSet<Region *, 4> regionSet;
    for (auto &region : op->getRegions()) {
      SmallVector<RegionSuccessor> successors;
      regionOp.getSuccessorRegions(region.getRegionNumber(), {}, successors);
      for (auto iter : successors) {
        auto *succReg = iter.getSuccessor();
        if (!succReg) {
          regionSet.insert(&region);
          break;
        }
      }
    }
    for (Region *region : regionSet) {
      auto blocksToAdd = collectAllExits(op, region->getRegionNumber());
      blocks.insert(blocks.end(), blocksToAdd.begin(), blocksToAdd.end());
    }
    return blocks;
  }
  for (auto &region : op->getRegions()) {
    auto blocksToAdd = collectAllExits(op, region.getRegionNumber());
    blocks.insert(blocks.end(), blocksToAdd.begin(), blocksToAdd.end());
  }
  return blocks;
}

/// Append the predecessors of \p block to \p worklist if and only if \p block
/// is not in \p blocksVisited.
static void
appendPredecessorsToWorklist(std::deque<Block *> &worklist, Block *block,
                             const SmallPtrSetImpl<Block *> &blocksVisited) {
  auto appendIfNotVisited = [&](auto preds) {
    for (auto *p : preds)
      if (!blocksVisited.count(p))
        worklist.push_back(p);
  };
  if (block->hasNoPredecessors()) {
    Region *r = block->getParent();
    auto rNum = r->getRegionNumber();
    // An entry block in a region may be a successor to the exit blocks of other
    // regions in the same Op.
    if (auto regionOp = dyn_cast<RegionBranchOpInterface>(r->getParentOp())) {
      SmallPtrSet<Region *, 4> regionSet;
      // Collect all the preceeding regions in a set.
      for (auto &region : regionOp->getRegions()) {
        SmallVector<RegionSuccessor> successors;
        regionOp.getSuccessorRegions(region.getRegionNumber(), {}, successors);
        for (auto iter : successors) {
          auto *succReg = iter.getSuccessor();
          if (succReg && succReg->getRegionNumber() == rNum) {
            regionSet.insert(&region);
            break;
          }
        }
      }
      // Add all the exit blocks of the preceeding regions to the worklist.
      for (Region *pred : regionSet) {
        auto blocksToAdd =
            collectAllExits(r->getParentOp(), pred->getRegionNumber());
        appendIfNotVisited(blocksToAdd);
      }
    }
  } else {
    appendIfNotVisited(block->getPredecessors());
  }
}

/// Append predecessor blocks of \p block unconditionally to \p worklist.
static void appendPredecessorsToWorklist(std::deque<Block *> &worklist,
                                         Block *block) {
  SmallPtrSet<Block *, 4> ignoreBlocksVisited;
  appendPredecessorsToWorklist(worklist, block, ignoreBlocksVisited);
  worklist.push_back(block);
}

static bool opResultOfType(Operation *op, Type ofTy) {
  auto results = op->getResults();
  for (auto r : results)
    if (r.getType() == ofTy)
      return true;
  return false;
}

/// Return true if and only if the value \p defVal is the result of an Operation
/// owned by the operation \p op.
static bool isDescendantOf(Operation *op, Value defVal) {
  if (auto *def = defVal.getDefiningOp())
    return op->isAncestor(def);
  for (auto *parent = cast<BlockArgument>(defVal).getOwner()->getParentOp();
       parent; parent = parent->getParentOp())
    if (parent == op)
      return true;
  return false;
}

/// Return the type after \p ty is dereferenced.
static Type dereferencedType(Type ty) {
  if (isa<quake::RefType>(ty))
    return quake::WireType::get(ty.getContext());
  return cast<cudaq::cc::PointerType>(ty).getElementType();
}

namespace {
/// For operations that contain Regions, a data-flow analysis is done over all
/// the Regions in the Op to determine the use-def information for scalar memory
/// reference. A scalar memory reference may be a classical variable (as
/// allocated with a cc.alloca) or a quantum reference (as allocated with a
/// `quake.alloca`). This class is used to track a map from memory references to
/// SSA virtual registers within blocks and maintain information on how to
/// stitch together blocks held by the Regions of the Op.
///
/// There are 3 basic cases.
///
///    -# High-level operations that take region arguments. In this case all
///       def information is passed as arguments between the blocks if it is
///       live. Use information, if only used, is passed as promoted loads,
///       otherwise it involves a def and is passed as an argument.
///    -# High-level operations that disallow region arguments. In this case
///       uses may have loads promoted to immediately before the operation.
///    -# Function operations. In this case, the body is a plain old CFG and
///       classical pruned SSA form (live SSA) with block arguments is used.
class RegionDataFlow {
public:
  // Typedefs to improve readability.
  using MemRef = Value; // A value that is a memory reference.
  using SSAReg = Value; // A value that is an SSA virtual register.
  using OrderedMemRegMap =
      llvm::MapVector<MemRef, SSAReg>; // A map that preserves insertion order.

  explicit RegionDataFlow(Operation *op) {
    if (isFunctionOp(op) || op->hasTrait<OpTrait::NoRegionArguments>())
      originalOpArgs = 0;
    else
      originalOpArgs = op->getNumOperands();
  }

  /// Add \p block to the data-flow map for processing. This will add arguments
  /// to the block for any region arguments not already appended.
  bool addBlock(Block *block) {
    assert(block);
    if (!rMap.count(block)) {
      rMap.insert({block, OrderedMemRegMap{}});
      originalBlockArgs[block] = block->getNumArguments();
    }
    return maybeAddEscapingBlockArguments(block);
  }

  bool updateBlock(Block *block) {
    assert(block && rMap.count(block));
    bool changed = false;
    for (Block *succ : block->getSuccessors())
      changed |= addBlock(succ);
    return changed;
  }

  /// Add a binding for memory reference \p mr to the virtual register \p sr in
  /// \p block. This binding is only valid within \p block. Once the block is
  /// fully processed, the set of bindings will reflect the live-out values from
  /// the basic block, \p block.
  ///
  /// Bindings are the mechanism for doing data-flow within a block.
  void addBinding(Block *block, MemRef mr, SSAReg sr) {
    assert(block && rMap.count(block) && mr);
    rMap[block][mr] = sr;
  }

  /// Used to cancel a binding when the value at a memory location is considered
  /// indeterminant because of an unknown operation that uses the memory
  /// location.
  void cancelBinding(Block *block, MemRef mr) {
    addBinding(block, mr, SSAReg{});
  }

  bool hasBinding(Block *block, MemRef mr) const {
    assert(block && rMap.count(block));
    return rMap.find(block)->second.count(mr);
  }

  /// Returns a binding. The binding must be present in the map.
  SSAReg getBinding(Block *block, MemRef mr) {
    assert(block && rMap.count(block) && mr && rMap[block].count(mr));
    return rMap[block][mr];
  }

  /// Add a (possibly) escaping binding for memory reference \p mr for the
  /// entire non-function operation. A memory reference may be used or live-out
  /// of \p block but not have a dominating definition in \p block. In these
  /// cases, the value will be passed as an argument to all blocks in the
  /// operation.
  std::pair<SSAReg, bool> addEscapingBinding(Block *block, MemRef mr) {
    assert(block && rMap.count(block) && mr && !isFunctionBlock(block));
    bool newlyAdded = false;
    if (!escapes.count(mr)) {
      auto off = escapes.size();
      escapes[mr] = off;
      newlyAdded = true;
    }
    bool changed = maybeAddEscapingBlockArguments(block);
    const auto blockArgNum = originalBlockArgs[block] + escapes[mr];
    auto ba = block->getArgument(blockArgNum);
    rMap[block][mr] = ba;
    if (newlyAdded && hasPromotedMemRef(mr)) {
      promoChange |= convertPromotedToEscapingDef(block, mr, blockArgNum);
      changed |= promoChange;
    }
    return {ba, changed};
  }

  /// Is \p mr a known escaping binding?
  bool hasEscape(MemRef mr) const {
    assert(mr);
    return escapes.count(mr);
  }

  SSAReg reloadMemoryReference(OpBuilder &builder, MemRef mr) {
    if (isa<quake::RefType>(mr.getType())) {
      auto wireTy = quake::WireType::get(builder.getContext());
      return builder.create<quake::UnwrapOp>(mr.getLoc(), wireTy, mr);
    }
    return builder.create<cudaq::cc::LoadOp>(mr.getLoc(), mr);
  }

  /// Update the terminator of \p block. All terminators must have operands
  /// added for any escapes that have been added to the op. Each block may have
  /// its own unique definitions for the list of escapes and those definitions
  /// must be threaded.
  ///
  /// If the parent operation is a function, this does \e not update the
  /// terminator. Terminators in functions are updated on-the-fly using CFG
  /// live-in information elsewhere.
  void updateTerminator(Block *block) {
    assert(block);
    if (isFunctionBlock(block)) {
      // The CFG of a function is updated in place and tracked by liveInMap.
      return;
    }

    auto *term = block->getTerminator();
    auto *ctx = term->getContext();
    auto *parent = block->getParentOp();

    auto reloadWhenInterference = [&](MemRef mr) {
      if (!rMap[block][mr]) {
        // The memory reference definition is unknown (interference by other
        // ops), so reload the value.
        OpBuilder builder(ctx);
        builder.setInsertionPoint(term);
        auto reg = reloadMemoryReference(builder, mr);
        rMap[block][mr] = reg;
      }
    };

    if (parent->hasTrait<OpTrait::NoRegionArguments>() &&
        block->hasNoSuccessors()) {
      if (hasLiveOutOfParent()) {
        auto liveOuts = getLiveOutOfParent();
        SmallVector<Value> args(term->getOperands());
        for (auto o : liveOuts) {
          auto a = rMap[block].count(o) ? rMap[block][o] : promotedMem[o];
          args.push_back(a);
        }
        term->setOperands(args);
      }
      return;
    }

    // Reverse the escapes map.
    DenseMap<unsigned, MemRef> revEscapes;
    for (auto [a, b] : escapes)
      revEscapes[b] = a;

    if (auto branch = dyn_cast<BranchOpInterface>(term)) {
      // This terminator can be handled via the BranchOpInterface and (likely)
      // represents low-level CFG branching within the parent op.
      for (auto iter : llvm::enumerate(block->getSuccessors())) {
        auto idx = iter.index();
        auto *succ = iter.value();
        const auto braArgSize = branch.getSuccessorOperands(idx).size();
        const auto succArgSize = succ->getNumArguments();
        if (braArgSize >= succArgSize)
          continue;
        auto off = braArgSize - originalBlockArgs[succ];
        SmallVector<Value> newArgs;
        for (unsigned i = off; i < succArgSize; ++i) {
          auto mr = revEscapes[i];
          reloadWhenInterference(mr);
          newArgs.push_back(rMap[block][mr]);
        }
        branch.getSuccessorOperands(idx).append(newArgs);
      }
      return;
    }

    // Otherwise the terminator is simpler (likely from the CC dialect), and can
    // also be handled in a common (but slightly different) way.
    const unsigned addend = isa<cudaq::cc::ConditionOp>(term) ? 1 : 0;
    SmallVector<Value> newArgs(term->getOperands());
    const unsigned offset = newArgs.size() - addend - originalOpArgs;
    for (unsigned i = offset; i < getNumEscapes(); ++i) {
      assert(revEscapes.count(i));
      auto mr = revEscapes[i];
      reloadWhenInterference(mr);
      newArgs.push_back(rMap[block][mr]);
    }
    term->setOperands(newArgs);
  }

  /// Get all the escaping bindings.
  SmallVector<Value> getAllEscapingBindingDefs() {
    SetVector<Value> results;
    for (auto &[memref, pos] : escapes)
      results.insert(memref);
    return {results.begin(), results.end()};
  }

  /// Add \p mr to the set of live-in definitions for \p block. This can only be
  /// used if the parent is a function.
  std::pair<SSAReg, bool> addLiveInToBlock(Block *block, MemRef mr) {
    assert(block && mr && isFunctionBlock(block));
    if (!liveInMap.count(block))
      liveInMap.insert({block, DenseMap<MemRef, SSAReg>{}});
    if (liveInMap[block].count(mr))
      return {liveInMap[block][mr], /*changed=*/false};

    // `mr` has not already been added.
    // Add it as an argument to `block`.
    auto ty = dereferencedType(mr.getType());
    SSAReg newArg = block->addArgument(ty, mr.getLoc());
    liveInMap[block][mr] = newArg;

    // For each predecessor, add a load and forward the value to `block`.
    for (auto *pred : block->getPredecessors()) {
      for (auto iter : llvm::enumerate(pred->getSuccessors())) {
        auto idx = iter.index();
        auto *succ = iter.value();
        if (succ != block)
          continue;
        // Create the re-load.
        OpBuilder builder(pred->getTerminator());
        auto sr = reloadMemoryReference(builder, mr);
        // Update the branch's successor operands list.
        auto branch = cast<BranchOpInterface>(pred->getTerminator());
        branch.getSuccessorOperands(idx).append(ArrayRef<Value>{sr});
        assert(branch.getSuccessorOperands(idx).size() ==
               block->getNumArguments());
      }
    }
    return {newArg, /*changed=*/true};
  }

  /// Promote the memory dereference \p memuse to immediately before the parent
  /// operation. This allows uses within the regions of the parent to use the
  /// new dominating dereference. Used when \p op does not allow region
  /// arguments.
  Value createPromotedValue(Value memuse, Operation *op) {
    if (hasPromotedMemRef(memuse))
      return getPromotedMemRef(memuse);
    Operation *parent = op->getParentOp();
    OpBuilder builder(parent);
    Value newUse = reloadMemoryReference(builder, memuse);
    return addPromotedMemRef(memuse, newUse);
  }

  SSAReg getPromotedMemRef(MemRef mr) const {
    assert(hasPromotedMemRef(mr));
    return promotedMem.find(mr)->second;
  }

  /// Track the memory reference \p mr as being live-out of the parent
  /// operation. (\p parent is passed for the assertion check only.)
  void addLiveOutOfParent(Operation *parent, MemRef mr) {
    assert(parent && mr && !isFunctionOp(parent));
    liveOutSet.insert(mr);
  }

  SmallVector<MemRef> getLiveOutOfParent() const {
    return SmallVector<MemRef>(liveOutSet.begin(), liveOutSet.end());
  }

  void cleanupIfPromoChanged(SmallPtrSetImpl<Block *> &visited, Block *block) {
    assert(block);
    if (promoChange) {
      // A promoted load was converted to an escaping definition. We have to
      // revisit all the blocks to thread the new block arguments and
      // terminators.
      visited.clear();
      visited.insert(block);
      promoChange = false;
    }
  }

private:
  // Delete all ctors that should never be used.
  RegionDataFlow() = delete;
  RegionDataFlow(const RegionDataFlow &) = delete;
  RegionDataFlow(RegionDataFlow &&) = delete;

  bool hasLiveOutOfParent() const { return !liveOutSet.empty(); }
  unsigned getNumEscapes() const { return escapes.size(); }

  bool hasPromotedMemRef(MemRef mr) const { return promotedMem.count(mr); }

  bool convertPromotedToEscapingDef(Block *block, MemRef mr,
                                    unsigned blockArgNum) {
    auto ssaReg = promotedMem[mr];
    SmallVector<Operation *> users(ssaReg.getUsers().begin(),
                                   ssaReg.getUsers().end());
    const bool result = !users.empty();
    for (auto *user : users) {
      Block *b = user->getBlock();
      if (b->getParentOp() != block->getParentOp()) {
        // Find the block in parent to add the escaping binding to.
        while (b->getParentOp() != block->getParentOp())
          b = b->getParentOp()->getBlock();
      }
      // Add an escaping binding to block `b` for the user to use.
      if (b != block)
        addEscapingBinding(b, mr);
      user->replaceUsesOfWith(ssaReg, b->getArgument(blockArgNum));
    }
    return result;
  }

  SSAReg addPromotedMemRef(MemRef mr, SSAReg sr) {
    assert(!hasPromotedMemRef(mr));
    promotedMem[mr] = sr;
    return sr;
  }

  bool maybeAddEscapingBlockArguments(Block *block) {
    if (isFunctionEntryBlock(block))
      return false;

    assert(block->getNumArguments() >= originalBlockArgs[block]);
    auto addedBlockArgs = block->getNumArguments() - originalBlockArgs[block];
    if (addedBlockArgs >= getNumEscapes())
      return false;

    // Make sure not to re-add arguments that were already added.
    auto dropCount = addedBlockArgs;
    [[maybe_unused]] unsigned counter = dropCount;
    for (auto [mr, off] : escapes) {
      if (dropCount) {
        --dropCount;
        continue;
      }
      assert(counter++ == off);
      auto ty = dereferencedType(mr.getType());
      SSAReg newArg = block->addArgument(ty, mr.getLoc());
      if (!rMap[block].count(mr))
        rMap[block][mr] = newArg;
    }
    return true;
  }

  /// The original number of operands to the parent op.
  unsigned originalOpArgs;
  /// A map for each block to its bindings from a memory reference to a virtual
  /// register value.
  DenseMap<Block *, OrderedMemRegMap> rMap;
  /// A map of memory references to offsets in the appended set of block
  /// arguments. The appended set starts at `originalBlockArgs[block]`.
  llvm::MapVector<MemRef, unsigned> escapes;
  /// Promotions of memory references to values immediately prior to the parent
  /// op. The exact promotion depends on uses/defs and scope.
  DenseMap<MemRef, SSAReg> promotedMem;
  /// This is the set of all definitions that are live-out of this op's regions
  /// and thus must be returned as results of the op. The op cannot be a
  /// function.
  SetVector<MemRef> liveOutSet;
  /// A map from a block to the original number of arguments for the block. Do
  /// not assume that every block in the original parent op has the same number
  /// of block arguments.
  DenseMap<Block *, unsigned> originalBlockArgs;
  /// For the body of a function, we maintain a distinct map for each block of
  /// the definitions that are live-in to each block.
  DenseMap<Block *, DenseMap<MemRef, SSAReg>> liveInMap;
  bool promoChange = false;
};
} // namespace

namespace {
/// The reset operation is a bit of an oddball and doesn't support the
/// QuakeOperator interface. Handle it special for now.
class ResetOpPattern : public OpRewritePattern<quake::ResetOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ResetOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto wireTy = quake::WireType::get(rewriter.getContext());
    auto opnd = op.getTargets();
    assert(opnd.getType() == quake::RefType::get(rewriter.getContext()));
    Value target = rewriter.create<quake::UnwrapOp>(loc, wireTy, opnd);
    auto newOp =
        rewriter.create<quake::ResetOp>(loc, TypeRange{wireTy}, target);
    rewriter.replaceOpWithNewOp<quake::WrapOp>(op, newOp.getResult(0), opnd);
    return success();
  }
};

class DeallocOpPattern : public OpRewritePattern<quake::DeallocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto wireTy = quake::WireType::get(rewriter.getContext());
    auto opnd = op.getReference();
    assert(isa<quake::RefType>(opnd.getType()));
    Value target = rewriter.create<quake::UnwrapOp>(loc, wireTy, opnd);
    rewriter.replaceOpWithNewOp<quake::SinkOp>(op, target);
    return success();
  }
};
} // namespace

template <typename OP>
class Wrapper : public OpRewritePattern<OP> {
public:
  using Base = OpRewritePattern<OP>;
  using Base::Base;

  LogicalResult matchAndRewrite(OP op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value> unwrapCtrls;
    auto wireTy = quake::WireType::get(rewriter.getContext());
    auto qrefTy = quake::RefType::get(rewriter.getContext());
    // Scan the control and target positions. Any that were not Wires will be
    // promoted to Wires via an unwrap operation. These unwrap ops become the
    // arguments to the quantum value form of the new quantum operation.
    if constexpr (!quake::isMeasure<OP>) {
      for (auto opnd : op.getControls()) {
        auto opndTy = opnd.getType();
        if (opndTy == qrefTy) {
          auto unwrap = rewriter.create<quake::UnwrapOp>(loc, wireTy, opnd);
          unwrapCtrls.push_back(unwrap);
        } else {
          unwrapCtrls.push_back(opnd);
        }
      }
    }
    SmallVector<Value> unwrapTargs;
    for (auto opnd : op.getTargets()) {
      auto opndTy = opnd.getType();
      if (opndTy == qrefTy) {
        auto unwrap = rewriter.create<quake::UnwrapOp>(loc, wireTy, opnd);
        unwrapTargs.push_back(unwrap);
      } else {
        unwrapTargs.push_back(opnd);
      }
    }

    auto threadWires = [&](const SmallVectorImpl<Value> &wireOperands,
                           auto newOp, unsigned addend) {
      unsigned count = 0;
      for (auto i : llvm::enumerate(wireOperands)) {
        auto opndTy = i.value().getType();
        auto offset = i.index() + addend;
        if (opndTy == qrefTy) {
          rewriter.create<quake::WrapOp>(loc, newOp.getResult(offset),
                                         i.value());
        } else if (opndTy == wireTy) {
          op.getResult(count++).replaceAllUsesWith(newOp.getResult(offset));
        }
      }
      rewriter.eraseOp(op);
    };

    if constexpr (quake::isMeasure<OP>) {
      // The result type of the bits is the same. Add the wire types.
      SmallVector<Type> newTy = {op.getBits().getType()};
      SmallVector<Type> wireTys(unwrapTargs.size(), wireTy);
      newTy.append(wireTys.begin(), wireTys.end());
      auto newOp = rewriter.create<OP>(loc, newTy, unwrapTargs,
                                       op.getRegisterNameAttr());
      SmallVector<Value> wireOperands = op.getTargets();
      op.getResult(0).replaceAllUsesWith(newOp.getResult(0));
      threadWires(wireOperands, newOp, 1);
    } else {
      // Scan the control and target positions. Any that were not wires
      // originally are now placed in the result vector. Those new results are
      // propagated to wrap operations.
      auto numberOfWires = unwrapCtrls.size() + unwrapTargs.size();
      SmallVector<Type> wireTys{numberOfWires, wireTy};
      auto newOp = rewriter.create<OP>(
          loc, wireTys, op.getIsAdjAttr(), op.getParameters(), unwrapCtrls,
          unwrapTargs, op.getNegatedQubitControlsAttr());
      SmallVector<Value> wireOperands = op.getControls();
      wireOperands.append(op.getTargets().begin(), op.getTargets().end());
      threadWires(wireOperands, newOp, 0);
    }
    return success();
  }
};

#define WRAPPER(OpClass) Wrapper<quake::OpClass>
#define WRAPPER_QUANTUM_OPS QUANTUM_OPS(WRAPPER)
#define RAW(OpClass) quake::OpClass
#define RAW_QUANTUM_OPS QUANTUM_OPS(RAW)

namespace {
class MemToRegPass : public cudaq::opt::impl::MemToRegBase<MemToRegPass> {
public:
  using MemToRegBase::MemToRegBase;
  using DefnMap = DenseMap<Value, Value>;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Function before memtoreg:\n" << func << "\n\n");

    if (!quantumValues && !classicalValues) {
      // nothing to do
      LLVM_DEBUG(llvm::dbgs() << "memtoreg: both quantum and classical "
                                 "transformations are disabled.\n");
      return;
    }

    // 1) Rewrite the quantum operations into the intermediate QLS form.
    if (failed(convertToQLS()))
      return;

    // 2) Convert load/store memory ops to value form.
    MemoryAnalysis memAnalysis(func);
    SmallPtrSet<Operation *, 4> cleanUps;
    processOpWithRegions(func, memAnalysis, cleanUps);

    // 3) Cleanup the dead ops.
    SmallVector<quake::WrapOp> wrapOps;
    for (auto *op : cleanUps) {
      if (auto wrap = dyn_cast<quake::WrapOp>(op)) {
        wrapOps.push_back(wrap);
        continue;
      }
      op->dropAllUses();
      op->erase();
    }
    for (auto wrap : wrapOps) {
      auto ref = wrap.getRefValue();
      auto wire = wrap.getWireValue();
      if (!ref || wire.getUses().empty()) {
        wrap->dropAllUses();
        wrap->erase();
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Finalized:\n" << func << "\n\n");
  }

  void handleSubRegions(Operation *parent, const MemoryAnalysis &memAnalysis,
                        SmallPtrSetImpl<Operation *> &cleanUps) {
    for (auto &region : parent->getRegions())
      for (auto &block : region)
        for (auto &op : block)
          if (op.getNumRegions())
            processOpWithRegions(&op, memAnalysis, cleanUps);
  }

  /// Process the operation \p parent, which must contain regions, and derive
  /// its use-def informations as an independent subgraph. Operations with
  /// regions are processed in a post-order traversal of the function.
  void processOpWithRegions(Operation *parent,
                            const MemoryAnalysis &memAnalysis,
                            SmallPtrSetImpl<Operation *> &cleanUps) {
    auto *ctx = &getContext();
    auto wireTy = quake::WireType::get(ctx);
    auto qrefTy = quake::RefType::get(ctx);

    if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(parent)) {
      // Special case: add an else region if it is absent from parent.
      auto &elseRegion = ifOp.getElseRegion();
      if (elseRegion.empty()) {
        auto block = new Block;
        elseRegion.push_back(block);
        OpBuilder builder(ctx);
        builder.setInsertionPointToEnd(block);
        builder.create<cudaq::cc::ContinueOp>(ifOp.getLoc());
      }
    }

    // First, if any operations held by the blocks of \p parent contain regions,
    // recursively process those operations. This establishes the value
    // semantics interface for these macro ops.
    handleSubRegions(parent, memAnalysis, cleanUps);

    SmallPtrSet<Block *, 4> blocksVisited;
    RegionDataFlow dataFlow(parent);

    // To produce a (semi-)pruned SSA graph, the Region's blocks are walked from
    // exits to entries to produce liveness information from predecessor to
    // successor blocks. (It is not possible to construct a fully pruned SSA IR
    // in the MLIR design of Ops with Regions as both exits and backedges must
    // have the exact same signatures regardless of liveness.)
    auto worklist = collectAllExits(parent);
    while (!worklist.empty()) {
      Block *block = worklist.front();
      worklist.pop_front();
      blocksVisited.insert(block);
      bool blockChanged = dataFlow.addBlock(block);

      // If this is the entry block and there are quantum reference arguments
      // into the function, promote them to wire values immediately.
      if (quantumValues && isFunctionEntryBlock(block)) {
        for (auto arg : block->getArguments()) {
          if (arg.getType() == qrefTy) {
            OpBuilder builder(ctx);
            builder.setInsertionPointToStart(block);
            Value v =
                builder.create<quake::UnwrapOp>(arg.getLoc(), wireTy, arg);
            dataFlow.addBinding(block, arg, v);
          }
        }
      }

      // Loop over all operations in the block.
      for (Operation &oper : *block) {
        Operation *op = &oper;

        // For any operation that creates a value of quantum reference type,
        // replace it with a null wire (if it is an AllocaOp) or unwrap the
        // reference to get the wire.
        if (opResultOfType(op, qrefTy)) {
          if (!quantumValues)
            continue;
          // If this op defines a quantum reference, record it in the maps.
          if (auto alloc = dyn_cast<quake::AllocaOp>(op);
              alloc && memAnalysis.isMember(alloc)) {
            // If it is a known non-escaping alloca, then replace it with a
            // null wire and record it for removal.
            if (!dataFlow.hasBinding(block, alloc)) {
              OpBuilder builder(alloc);
              Value v =
                  builder.create<quake::NullWireOp>(alloc.getLoc(), wireTy);
              cleanUps.insert(op);
              dataFlow.addBinding(block, alloc, v);
            }
          } else {
            OpBuilder builder(ctx);
            builder.setInsertionPointAfter(op);
            for (auto r : op->getResults()) {
              Value v =
                  builder.create<quake::UnwrapOp>(op->getLoc(), wireTy, r);
              dataFlow.addBinding(block, r, v);
            }
          }
          continue;
        }

        // If this is a classical stack slot allocation (and we're processing
        // classical values), promote the allocation to an undefined value.
        if (auto alloc = dyn_cast<cudaq::cc::AllocaOp>(op);
            alloc && memAnalysis.isMember(alloc)) {
          if (classicalValues) {
            if (!dataFlow.hasBinding(block, alloc)) {
              OpBuilder builder(alloc);
              Value v = builder.create<cudaq::cc::UndefOp>(
                  alloc.getLoc(), alloc.getElementType());
              cleanUps.insert(op);
              dataFlow.addBinding(block, alloc, v);
            }
          }
          continue;
        }

        // If this is a new value being created, add it to the map of values for
        // this block so it can be tracked and forwarded.
        if (auto nullWire = dyn_cast<quake::NullWireOp>(op)) {
          if (quantumValues)
            dataFlow.addBinding(block, nullWire, nullWire.getResult());
          continue;
        }
        if (auto undef = dyn_cast<cudaq::cc::UndefOp>(op)) {
          if (classicalValues)
            dataFlow.addBinding(block, undef, undef.getResult());
          continue;
        }

        // If op is a use of a memory ref, forward the last def if there is one.
        // If no def is known, then if this is a function entry raise an error,
        // or if this op does not have region arguments or this use is not also
        // being defined add a dominating def immediately before parent, or
        // (the default) add a block argument for the def.
        auto handleUse = [&]<typename T>(T useop, Value memuse) {
          if (!memuse)
            return;

          // If the use's def is already in the map, then use that def.
          if (dataFlow.hasBinding(block, memuse)) {
            auto memuseBinding = dataFlow.getBinding(block, memuse);
            if (!memuseBinding) {
              dataFlow.addBinding(block, memuse, useop);
            } else if (useop.getResult() != memuseBinding) {
              useop.replaceAllUsesWith(memuseBinding);
              cleanUps.insert(op);
            }
            return;
          }

          // The def isn't in the map.
          if (isFunctionEntryBlock(block)) {
            // This is a function's entry block. This use can't come before a
            // def in a valid program. Raise an error.
            oper.emitError("use before def in function");
            signalPassFailure();
            return;
          }

          // Is this an entry block and NOT a function?
          if (block->isEntryBlock()) {
            // Create a promoted value that dominates parent.
            auto newUseopVal = dataFlow.createPromotedValue(memuse, op);
            if (!dataFlow.hasEscape(memuse)) {
              // In this case, parent does not accept region arguments so the
              // reference values must already be defined to dominate parent.
              useop.replaceAllUsesWith(newUseopVal);
              dataFlow.addBinding(block, memuse, newUseopVal);
              cleanUps.insert(useop);
              return;
            }
            // Otherwise, parent requires region arguments, so dominating
            // values must be added and threaded through the block
            // arguments.
            auto numOperands = parent->getNumOperands();
            parent->insertOperands(numOperands, ValueRange{newUseopVal});
            for (auto &reg : parent->getRegions()) {
              if (reg.empty())
                continue;
              auto *entry = &reg.front();
              bool changes = dataFlow.addBlock(entry);
              auto [blockArg, changed] =
                  dataFlow.addEscapingBinding(entry, memuse);
              if (useop->getParentRegion() == &reg)
                useop.replaceAllUsesWith(blockArg);
              if (entry == block)
                dataFlow.addBinding(block, memuse, blockArg);
              dataFlow.cleanupIfPromoChanged(blocksVisited, block);
              if (changes || changed)
                appendPredecessorsToWorklist(worklist, entry);
            }
            cleanUps.insert(useop);
            return;
          } // end block is entry

          // The def is not in the map AND this is not an entry block.

          // Is parent a function?
          if (isFunctionOp(parent)) {
            // The parent is a function with a plain old CFG. In this case,
            // record the live-in use for `block` and generate a new block
            // argument. All the predecessor blocks will need to pass in the
            // value of this memory reference.
            auto [newUseArg, changed] =
                dataFlow.addLiveInToBlock(block, memuse);
            useop.replaceAllUsesWith(newUseArg);
            cleanUps.insert(useop);
            if (changed)
              for (auto *pred : block->getPredecessors())
                worklist.push_back(pred);
            return;
          }

          if (!dataFlow.hasEscape(memuse)) {
            // Create a promoted value that dominates parent. In this case, the
            // ref value must already be defined somewhere that dominates Op
            // `parent`, so we can just reload it.
            auto newUseopVal = dataFlow.createPromotedValue(memuse, op);
            useop.replaceAllUsesWith(newUseopVal);
            dataFlow.addBinding(block, memuse, newUseopVal);
            cleanUps.insert(useop);
          }
        };
        if (auto unwrap = dyn_cast<quake::UnwrapOp>(op)) {
          if (quantumValues)
            handleUse(unwrap, unwrap.getRefValue());
          continue;
        }
        if (auto load = dyn_cast<cudaq::cc::LoadOp>(op)) {
          if (classicalValues) {
            auto memuse = load.getPtrvalue();
            // Process only singleton classical scalars, no aggregates.
            if (auto *useOp = memuse.getDefiningOp())
              if (memAnalysis.isMember(useOp))
                handleUse(load, memuse);
          }
          continue;
        }

        // If op is a def of a memory ref, add a new binding to the data-flow
        // map for this def. If this def occurs in a non-function structured Op
        // and is defining a memory reference from above, and Op allows region
        // arguments, then add this definition as a region argument.
        auto handleDefinition = [&]<typename T>(T defop, Value val,
                                                Value memdef) {
          cleanUps.insert(defop);
          if (!isFunctionOp(parent) && !isDescendantOf(parent, memdef)) {
            if (parent->hasTrait<OpTrait::NoRegionArguments>()) {
              dataFlow.createPromotedValue(memdef, defop);
              dataFlow.addLiveOutOfParent(parent, memdef);
            } else {
              for (auto &reg : parent->getRegions()) {
                if (reg.empty())
                  continue;
                Block *entry = &reg.front();
                bool changes = dataFlow.addBlock(entry);
                auto pr = dataFlow.addEscapingBinding(entry, memdef);
                if (changes || pr.second)
                  appendPredecessorsToWorklist(worklist, entry);
              }
              dataFlow.cleanupIfPromoChanged(blocksVisited, block);
              auto pr = dataFlow.addEscapingBinding(block, memdef);
              if (pr.second)
                appendPredecessorsToWorklist(worklist, block);
            }
          }
          dataFlow.addBinding(block, memdef, val);
        };
        if (auto wrap = dyn_cast<quake::WrapOp>(op)) {
          if (quantumValues)
            handleDefinition(wrap, wrap.getWireValue(), wrap.getRefValue());
          continue;
        }
        if (auto store = dyn_cast<cudaq::cc::StoreOp>(op)) {
          if (classicalValues) {
            auto memdef = store.getPtrvalue();
            // Process only singleton classical scalars, no aggregates.
            if (auto *defOp = memdef.getDefiningOp())
              if (memAnalysis.isMember(defOp))
                handleDefinition(store, store.getValue(), store.getPtrvalue());
          }
          continue;
        }

        // If op uses a quantum reference, then halt forwarding the unwrap
        // use chain and leave a wrap dominating op.
        for (auto v : op->getOperands()) {
          if ((v.getType() == qrefTy) && dataFlow.hasBinding(block, v))
            if (auto vBinding = dataFlow.getBinding(block, v)) {
              OpBuilder builder(op);
              builder.create<quake::WrapOp>(op->getLoc(), vBinding, v);
              dataFlow.cancelBinding(block, v);
            }
        }
      } // end of loop over ops in block

      blockChanged |= dataFlow.updateBlock(block);
      if (blockChanged)
        appendPredecessorsToWorklist(worklist, block);
      else
        appendPredecessorsToWorklist(worklist, block, blocksVisited);

      dataFlow.updateTerminator(block);
    } // end of worklist loop

    if (!isFunctionOp(parent)) {
      // Determine all the unique definitions.
      SmallVector<Value> allDefs =
          parent->hasTrait<OpTrait::NoRegionArguments>()
              ? dataFlow.getLiveOutOfParent()
              : dataFlow.getAllEscapingBindingDefs();

      if (!allDefs.empty()) {
        // Replace parent with a copy.
        SmallVector<Type> resultTypes(parent->getResultTypes());
        for (auto d : allDefs)
          resultTypes.push_back(dereferencedType(d.getType()));
        ConversionPatternRewriter builder(ctx);
        builder.setInsertionPoint(parent);
        SmallVector<Value> operands;
        for (auto opndVal : parent->getOperands())
          operands.push_back(opndVal);
        if (!parent->hasTrait<OpTrait::NoRegionArguments>())
          for (auto d : allDefs)
            operands.push_back(dataFlow.getPromotedMemRef(d));
        Operation *np =
            Operation::create(parent->getLoc(), parent->getName(), resultTypes,
                              operands, parent->getAttrs(),
                              parent->getSuccessors(), parent->getNumRegions());
        builder.insert(np);
        for (unsigned i = 0; i < parent->getNumRegions(); ++i)
          builder.inlineRegionBefore(parent->getRegion(i), np->getRegion(i),
                                     np->getRegion(i).begin());
        for (unsigned i = 0; i < parent->getNumResults(); ++i)
          parent->getResult(i).replaceAllUsesWith(np->getResult(i));
        builder.setInsertionPointAfter(np);
        for (auto iter : llvm::enumerate(allDefs)) {
          auto i = iter.index() + parent->getNumResults();
          if (np->getResult(i).getType() == wireTy)
            builder.create<quake::WrapOp>(np->getLoc(), np->getResult(i),
                                          iter.value());
          else
            builder.create<cudaq::cc::StoreOp>(np->getLoc(), np->getResult(i),
                                               iter.value());
        }
        cleanUps.insert(parent);
        parent = np;
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "After threading values:\n"
                            << *parent << "\n\n");
  }

  // Convert the function to "quantum load/store" (QLS) format.
  LogicalResult convertToQLS() {
    if (!quantumValues)
      return success();
    auto func = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<WRAPPER_QUANTUM_OPS, ResetOpPattern, DeallocOpPattern>(ctx);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<RAW_QUANTUM_OPS, quake::ResetOp,
                                 quake::DeallocOp>(
        [](Operation *op) { return !quake::hasNonVectorReference(op); });
    target.addLegalOp<quake::UnwrapOp, quake::WrapOp, quake::NullWireOp,
                      quake::SinkOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      emitError(func.getLoc(), "error converting to QLS form\n");
      signalPassFailure();
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "After converting to QLS:\n" << func << "\n\n");
    return success();
  }
};
} // namespace
