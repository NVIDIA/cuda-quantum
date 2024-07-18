/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
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

static bool neverTakesRegionArguments(Operation *op) {
  return op->hasTrait<OpTrait::NoRegionArguments>();
}

static bool onlyTakesLinearTypeArguments(Operation *op) {
  return op->hasTrait<cudaq::cc::LinearTypeArgsTrait>();
}

static bool isLinearType(Value v) { return quake::isLinearType(v.getType()); }

template <typename T>
void appendToWorklist(std::deque<Block *> &d, T collection) {
  d.insert(d.end(), collection.begin(), collection.end());
}

static Block *findParentBlock(Operation *parent, Block *block) {
  Operation *p = block->getParentOp();
  while (p && p != parent) {
    block = p->getBlock();
    p = block->getParentOp();
  }
  return block;
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

  explicit RegionDataFlow(Operation *op) {
    // Stitch together the control-flow across op's regions.
    if (auto regionOp = dyn_cast<RegionBranchOpInterface>(op)) {
      SmallVector<RegionSuccessor> successors;
      regionOp.getSuccessorRegions(std::nullopt, {}, successors);
      for (auto iter : successors)
        if (iter.getSuccessor())
          entryCFG.insert(&iter.getSuccessor()->front());
      for (auto &region : op->getRegions()) {
        SmallVector<Block *> regionExitBlocks;
        for (auto &b : region)
          if (b.hasNoSuccessors())
            regionExitBlocks.push_back(&b);
        regionOp.getSuccessorRegions(region.getRegionNumber(), {}, successors);
        // Every region has exactly one entry and one or more exits.
        for (auto *b : regionExitBlocks)
          for (auto iter : successors) {
            auto *succ = iter.getSuccessor();
            if (succ) {
              auto *s = &succ->front();
              backwardCFG[s].insert(b);
            } else {
              exitCFG.insert(b);
            }
          }
      }
    } else {
      for (auto &region : op->getRegions())
        for (auto &b : region) {
          if (b.isEntryBlock())
            entryCFG.insert(&b);
          if (b.hasNoSuccessors())
            exitCFG.insert(&b);
        }
    }
  }

  //===--------------------------------------------------------------------===//
  // Cached CFG information.
  //
  // Since ops with regions can have a complex CFG structure that connects
  // blocks in different regions in non-trivial ways, we cache that CFG
  // structure here.
  //===--------------------------------------------------------------------===//

  bool isEntryBlock(Block *block) { return entryCFG.count(block); }

  SmallVector<Block *> getEntryBlocks() {
    return {entryCFG.begin(), entryCFG.end()};
  }

  bool isExitBlock(Block *block) { return exitCFG.count(block); }

  SmallVector<Block *> getExitBlocks() {
    return {exitCFG.begin(), exitCFG.end()};
  }

  SmallVector<Block *> getPredecessors(Block *block) {
    if (backwardCFG.count(block))
      return {backwardCFG[block].begin(), backwardCFG[block].end()};
    auto range = block->getPredecessors();
    return {range.begin(), range.end()};
  }

  /// Add \p block to the data-flow map for processing. This will add arguments
  /// to the block for any region arguments not already appended.
  void addBlock(Block *block) {
    assert(block);
    if (!rMap.count(block)) {
      rMap.insert({block, DenseMap<MemRef, SSAReg>{}});
      liveInMap.insert({block, llvm::MapVector<MemRef, SSAReg>{}});
    }
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

  /// Create a re-load of a memory reference. This can be used to place a
  /// dominating load operation immediately prior to an op with regions.
  SSAReg reloadMemoryReference(OpBuilder &builder, MemRef mr) {
    if (isa<quake::RefType>(mr.getType())) {
      auto wireTy = quake::WireType::get(builder.getContext());
      return builder.create<quake::UnwrapOp>(mr.getLoc(), wireTy, mr);
    }
    return builder.create<cudaq::cc::LoadOp>(mr.getLoc(), mr);
  }

  SSAReg unsafeAddLiveInToBlock(Block *block, MemRef mr) {
    auto ty = dereferencedType(mr.getType());
    SSAReg newReg = block->addArgument(ty, mr.getLoc());
    liveInMap[block][mr] = newReg;
    return newReg;
  }

  /// Record the memory reference \p mr as live-in to \p block. This creates a
  /// new argument to \p block that will correspond to the value loaded from
  /// memory reference, \p mr.
  SSAReg addLiveInToBlock(Block *block, MemRef mr) {
    assert(block && liveInMap.count(block) && mr &&
           !liveInMap[block].count(mr) && !isFunctionEntryBlock(block));
    return unsafeAddLiveInToBlock(block, mr);
  }

  SSAReg maybeAddLiveInToBlock(Block *block, MemRef mr) {
    assert(block && liveInMap.count(block) && mr);
    if (liveInMap[block].count(mr))
      return liveInMap[block][mr];
    return addLiveInToBlock(block, mr);
  }

  void maybeAddBalancedLiveInToBlock(Block *block, MemRef mr) {
    if (liveOutSet.count(mr))
      maybeAddLiveInToBlock(block, mr);
  }

  /// Record the memory reference \p mr as live-in to \p block. The live-in
  /// value is specified as \p val. Consequently, \p val \em{must dominate} \p
  /// block.
  void addLiveInToBlock(Block *block, MemRef mr, SSAReg val) {
    assert(block && liveInMap.count(block) && mr && val &&
           !liveInMap[block].count(mr) && !isFunctionEntryBlock(block));
    liveInMap[block][mr] = val;
  }

  /// Returns a vector of memory references. These memory references are the
  /// ordered list of arguments to \p block.
  SmallVector<MemRef> getLiveInToBlock(Block *block) {
    assert(block && liveInMap.count(block));
    std::map<unsigned, MemRef> sortedMap;
    for (auto [mr, val] : liveInMap[block])
      if (auto arg = dyn_cast<BlockArgument>(val))
        if (arg.getOwner() == block)
          sortedMap[arg.getArgNumber()] = mr;

#ifndef NDEBUG
    // Sanity check that these arguments are contiguous.
    if (!sortedMap.empty()) {
      auto iter = sortedMap.begin();
      unsigned index = iter->first;
      for (++iter; iter != sortedMap.end(); ++iter) {
        assert(iter->first == index + 1);
        index = iter->first;
      }
    }
#endif

    SmallVector<MemRef> result;
    for (auto [index, mr] : sortedMap)
      result.push_back(mr);
    return result;
  }

  /// Promote the memory dereference \p memuse to immediately before the parent
  /// operation. This allows uses within the regions of the parent to use the
  /// new dominating dereference. These will be converted to live-in arguments
  /// if the op takes region arguments.
  SSAReg createPromotedValue(Operation *parent, Value memref) {
    if (promotedDefs.count(memref))
      return promotedDefs[memref];
    OpBuilder builder(parent);
    Value newUse = reloadMemoryReference(builder, memref);
    promotedDefs[memref] = newUse;
    return newUse;
  }

  SSAReg getPromotedValue(Value memref) {
    assert(memref && promotedDefs.count(memref));
    return promotedDefs[memref];
  }

  SmallVector<SSAReg> getPromotedDefValues() {
    SmallVector<SSAReg> result;
    for (auto [mr, val] : promotedDefs)
      result.push_back(val);
    return result;
  }

  /// If \p parent takes region arguments, convert the live-out parent results
  /// to live-in parent arguments. Convert the promoted loads to parent op
  /// arguments. Replace any uses of the promoted loads to uses of block
  /// arguments and insert modified blocks and their preds on the worklist.
  void updatePromotedDefs(Operation *parent, std::deque<Block *> &worklist) {
    if (liveOutSet.empty() || neverTakesRegionArguments(parent))
      return;
    const bool onlyLinearTypes = onlyTakesLinearTypeArguments(parent);
    assert(liveInArgs.empty() && "parent's live-in args should not be set");
    for (auto liveOut : liveOutSet) {
      assert(promotedDefs.count(liveOut));
      if (onlyLinearTypes && !isLinearType(promotedDefs[liveOut]))
        continue;
      liveInArgs.push_back(promotedDefs[liveOut]);
    }
    SmallPtrSet<Block *, 4> blockSet;
    for (auto [mr, val] : promotedDefs) {
      if (onlyLinearTypes && !isLinearType(val))
        continue;
      if (liveOutSet.count(mr)) {
        SmallVector<Operation *> users(val.getUsers().begin(),
                                       val.getUsers().end());
        for (auto *user : users) {
          auto *block = findParentBlock(parent, user->getBlock());
          if (!blockSet.count(block)) {
            // Add the promoted defs to this block as arguments. Add all of them
            // in order so that the argument list doesn't get permuted. Use the
            // unsafe call here because liveInMap should already have a binding
            // for memref to the promoted load value. That binding will be
            // overwritten.
            for (auto memref : liveOutSet) {
              if (onlyLinearTypes && !isLinearType(promotedDefs[memref]))
                continue;
              unsafeAddLiveInToBlock(block, memref);
            }
            blockSet.insert(block);
            worklist.push_back(block);
            appendToWorklist(worklist, getPredecessors(block));
          }
          Value newReg = liveInMap[block][mr];
          if (!hasBinding(block, mr) || getBinding(block, mr) == val)
            addBinding(block, mr, newReg);
          user->replaceUsesOfWith(val, newReg);
        }
      }
    }
  }

  /// Track the memory reference \p mr as being live-out of the parent
  /// operation. (\p parent is passed for the assertion check only.)
  void addLiveOutOfParent(Operation *parent, MemRef mr) {
    assert(parent && mr && !isFunctionOp(parent));
    liveOutSet.insert(mr);
  }

  SmallVector<MemRef> getLiveOutOfParent() const {
    return {liveOutSet.begin(), liveOutSet.end()};
  }

  bool hasLiveOutOfParent() const { return !liveOutSet.empty(); }

  /// Get the live-in arguments to the parent operation. These values must
  /// dominate parent.
  SmallVector<SSAReg> &getLiveInArgs() { return liveInArgs; }

private:
  // Delete all ctors that should never be used.
  RegionDataFlow() = delete;
  RegionDataFlow(const RegionDataFlow &) = delete;
  RegionDataFlow(RegionDataFlow &&) = delete;

  /// A map for each block to its bindings from a memory reference to a
  /// virtual register value.
  DenseMap<Block *, DenseMap<MemRef, SSAReg>> rMap;
  /// For a CFG, maintain a distinct map for each block of the definitions
  /// that are live-in to each block.
  DenseMap<Block *, llvm::MapVector<MemRef, SSAReg>> liveInMap;
  DenseMap<MemRef, SSAReg> promotedDefs;

  /// The list of live-in arguments to the parent. The parent cannot be a
  /// function.
  SmallVector<SSAReg> liveInArgs;
  /// This is the set of all definitions that are live-out of the parent's
  /// regions and thus must be returned as results. The parent cannot be a
  /// function.
  SetVector<MemRef> liveOutSet;

  SmallPtrSet<Block *, 2> entryCFG;
  SmallPtrSet<Block *, 2> exitCFG;
  DenseMap<Block *, SmallPtrSet<Block *, 2>> backwardCFG;
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
      SmallVector<Type> newTy = {op.getMeasOut().getType()};
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

    // 3) Cleanup the dead ops. Make sure to delay erasing wrap ops since they
    // may still have uses.
    SmallVector<quake::WrapOp> wrapOps;
    for (auto *op : cleanUps) {
      if (auto wrap = dyn_cast<quake::WrapOp>(op)) {
        wrapOps.push_back(wrap);
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "erasing: "; op->dump(); llvm::dbgs() << '\n');
      op->dropAllUses();
      op->erase();
    }
    for (auto wrap : wrapOps) {
      auto ref = wrap.getRefValue();
      auto wire = wrap.getWireValue();
      if (!ref || !wire.hasOneUse()) {
        LLVM_DEBUG(llvm::dbgs() << "erasing: "; wrap->dump();
                   llvm::dbgs() << '\n');
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
  /// regions are processed in a post-order traversal of the function. To
  /// produce a (semi-)pruned SSA graph, the Region's blocks are walked from
  /// exits to entries to produce liveness information from predecessor to
  /// successor blocks. (It is not possible to construct a \em fully pruned SSA
  /// IR in the MLIR design of Ops with Regions as both exits and backedges must
  /// have the exact same signatures regardless of liveness.)
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

    // 1. If any operations held by the blocks of \p parent contain regions,
    // recursively process those operations. This establishes the value
    // semantics interface for these macro ops.
    handleSubRegions(parent, memAnalysis, cleanUps);

    // 2. Traverse each basic block threading the defs to their uses. This will
    // construct the liveIn and liveOut maps for each block. If parent is not a
    // function, all references to memory from outside scopes are promoted to
    // dominating loads and if the reference is a definition it is recorded as
    // live-out of parent.
    RegionDataFlow dataFlow(parent);
    for (auto &region : parent->getRegions()) {
      for (auto &blockRef : region) {
        Block *block = &blockRef;
        dataFlow.addBlock(block);

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
        for (Operation &operRef : *block) {
          Operation *op = &operRef;

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
                cleanUps.insert(alloc);
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
          if (auto alloc = dyn_cast<cudaq::cc::AllocaOp>(op))
            if (memAnalysis.isMember(alloc)) {
              if (classicalValues && !dataFlow.hasBinding(block, alloc)) {
                OpBuilder builder(alloc);
                Value v = builder.create<cudaq::cc::UndefOp>(
                    alloc.getLoc(), alloc.getElementType());
                cleanUps.insert(alloc);
                dataFlow.addBinding(block, alloc, v);
              }
              continue;
            }

          // If this is a new value being created, add it to the map of values
          // for this block so it can be tracked and forwarded.
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

          // If op is a use of a memory ref, forward the last def if there is
          // one. If no def is known, then if this is a function entry raise an
          // error, or if this op does not have region arguments or this use is
          // not also being defined add a dominating def immediately before
          // parent, or (the default) add a block argument for the def.
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
                cleanUps.insert(useop);
              }
              return;
            }

            // At this point, the def isn't in the map.
            if (isFunctionEntryBlock(block)) {
              // This is a function's entry block. This use can't come before a
              // def in a valid program. Raise an error.
              operRef.emitError("use before def in function");
              signalPassFailure();
              return;
            }

            // Parent is not a function.
            if (!isDescendantOf(parent, memuse)) {
              // `block` is using a value from another scope.
              // Create a promoted value that dominates parent.
              auto newUseopVal = dataFlow.createPromotedValue(parent, memuse);
              dataFlow.addBinding(block, memuse, newUseopVal);
              dataFlow.addLiveInToBlock(block, memuse, newUseopVal);
              useop.replaceAllUsesWith(newUseopVal);
              cleanUps.insert(useop);
              return;
            }

            // The def is not in the map AND this is not an entry block.
            auto newDef = dataFlow.addLiveInToBlock(block, memuse);
            dataFlow.addBinding(block, memuse, newDef);
            useop.replaceAllUsesWith(newDef);
            cleanUps.insert(useop);
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
          // map for this def. If this def occurs in a non-function structured
          // Op and is defining a memory reference from above, and Op allows
          // region arguments, then add this definition as a region argument.
          auto handleDefinition = [&]<typename T>(T defop, Value val,
                                                  Value memdef) {
            dataFlow.addBinding(block, memdef, val);
            if (!isFunctionOp(parent)) {
              if (!isDescendantOf(parent, memdef)) {
                dataFlow.addLiveOutOfParent(parent, memdef);
                dataFlow.createPromotedValue(parent, memdef);
              }
            }
            cleanUps.insert(defop);
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
                  handleDefinition(store, store.getValue(),
                                   store.getPtrvalue());
            }
            continue;
          }

          // If op uses a quantum reference, then halt forwarding the unwrap
          // use chain and leave a wrap dominating op.
          for (auto v : op->getOperands())
            if ((v.getType() == qrefTy) && dataFlow.hasBinding(block, v))
              if (auto vBinding = dataFlow.getBinding(block, v)) {
                OpBuilder builder(op);
                builder.create<quake::WrapOp>(op->getLoc(), vBinding, v);
                dataFlow.cancelBinding(block, v);
              }

        } // end loop over ops
      }   // end loop over blocks
    }     // end loop over regions

    LLVM_DEBUG(llvm::dbgs() << "After threading intra-block:\n"
                            << *parent << "\n\n");

    std::deque<Block *> worklist;
    appendToWorklist(worklist, dataFlow.getExitBlocks());

    // 3. If there are defs that are live-out for parent and parent takes region
    // arguments, construct a list of live-in region arguments to add to the new
    // parent and replace uses of promoted defs with block arguments.
    dataFlow.updatePromotedDefs(parent, worklist);

    LLVM_DEBUG({
      llvm::dbgs() << "After fixing up promoted loads:\n"
                   << *parent << "\nPromotions:\n";
      for (auto v : dataFlow.getPromotedDefValues())
        v.dump();
      llvm::dbgs() << '\n';
    });

    // 4. Update the block arguments and terminators to thread the values
    // between the blocks in the CFG.
    // If there are defs that are live-out for parent, then they need to be
    // added to each terminator.
    // Update each pred's terminator to pass all the live-in values to a
    // successor.
    auto liveOutParent = dataFlow.getLiveOutOfParent();

    auto addTerminatorArgument = [&](Operation *term, Block *target,
                                     Value val) {
      if (auto branch = dyn_cast<BranchOpInterface>(term)) {
        unsigned numSuccs = branch->getNumSuccessors();
        bool changes = false;
        for (unsigned i = 0; i < numSuccs; ++i) {
          if (target && branch->getSuccessor(i) != target)
            continue;
          auto newArgs = branch.getSuccessorOperands(i).getForwardedOperands();
          if (std::find(newArgs.begin(), newArgs.end(), val) != newArgs.end())
            continue;
          branch.getSuccessorOperands(i).append(val);
          changes = true;
        }
        if (changes)
          worklist.push_back(term->getBlock());
        return;
      }
      SmallVector<Value> newArgs(term->getOperands());
      if (std::find(newArgs.begin(), newArgs.end(), val) != newArgs.end())
        return;
      newArgs.push_back(val);
      term->setOperands(newArgs);
      worklist.push_back(term->getBlock());
    };

    const bool usePromo = neverTakesRegionArguments(parent);
    const bool onlyLinear = onlyTakesLinearTypeArguments(parent);
    auto updateTerminator = [&](Operation *term, Block *target, auto bindings) {
      auto *block = term->getBlock();
      for (auto liveOut : bindings) {
        if (dataFlow.hasBinding(block, liveOut)) {
          if (!isFunctionBlock(block) && !usePromo && !onlyLinear)
            dataFlow.maybeAddBalancedLiveInToBlock(block, liveOut);
          auto oldVal = dataFlow.getBinding(block, liveOut);
          addTerminatorArgument(term, target, oldVal);
        } else if ((usePromo ||
                    (onlyLinear && !isa<quake::RefType>(liveOut.getType()))) &&
                   dataFlow.isEntryBlock(block)) {
          auto newVal = dataFlow.getPromotedValue(liveOut);
          dataFlow.addBinding(block, liveOut, newVal);
          addTerminatorArgument(term, target, newVal);
        } else {
          auto newArg = dataFlow.maybeAddLiveInToBlock(block, liveOut);
          addTerminatorArgument(term, target, newArg);
        }
      }
    };

    auto updateExitTerminator = [&](Block *block, auto bindings) {
      return updateTerminator(block->getTerminator(), nullptr, bindings);
    };

    SmallPtrSet<Block *, 8> blocksVisited;
    while (!worklist.empty()) {
      Block *block = worklist.front();
      worklist.pop_front();
      // Check terminator is threading live-out of parent values.
      if (!liveOutParent.empty() && dataFlow.isExitBlock(block))
        updateExitTerminator(block, liveOutParent);

      // Check that preds are threading all live-in values.
      auto liveInBlock = dataFlow.getLiveInToBlock(block);
      if (!liveInBlock.empty()) {
        auto preds = dataFlow.getPredecessors(block);
        for (auto *pred : preds)
          updateTerminator(pred->getTerminator(), block, liveInBlock);
      }

      // We should visit all the blocks at least once.
      blocksVisited.insert(block);
      auto preds = dataFlow.getPredecessors(block);
      for (auto *pred : preds)
        if (!blocksVisited.count(pred))
          worklist.push_back(pred);
    } // end of worklist loop

    if (dataFlow.hasLiveOutOfParent()) {
      // Get all the new results to append.
      auto allDefs = dataFlow.getLiveOutOfParent();

      // Replace parent with a copy.
      SmallVector<Type> resultTypes(parent->getResultTypes());
      for (auto d : allDefs)
        resultTypes.push_back(dereferencedType(d.getType()));
      ConversionPatternRewriter builder(ctx);
      builder.setInsertionPoint(parent);
      SmallVector<Value> operands(parent->getOperands());
      operands.insert(operands.end(), dataFlow.getLiveInArgs().begin(),
                      dataFlow.getLiveInArgs().end());
      Operation *np = Operation::create(
          parent->getLoc(), parent->getName(), resultTypes, operands,
          parent->getAttrs(), parent->getSuccessors(), parent->getNumRegions());
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

    LLVM_DEBUG(llvm::dbgs() << "After threading inter-block:\n"
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
