/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_DEADSTOREREMOVAL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "dead-store-removal"

using namespace mlir;

STATISTIC(numLoadsRemoved, "Number of loads replaced by a known value");
STATISTIC(numStoresRemoved, "Number of overwritten stores removed");

namespace {

//===----------------------------------------------------------------------===//
// Store/load forwarding, redundant load removal and removal of overwritten
// stores.
//
// The analysis tracks the *contents* of memory locations: a map from address to
// the SSA value known to be stored there. Two addresses built from the same
// SSA operands are the same location only as long as no merge point has been
// crossed. A block argument of a block with several predecessors (or of a
// loop's back-edge) is a phi node: the same SSA name is a different value
// depending on how control got there. The rules are therefore:
//
//   - Within a block every SSA value, including block arguments, is a single
//     value. Facts are freely created and compared within a block.
//   - A fact flows into a block only if it holds on *every* path that can reach
//     it: from the sole predecessor, from the intersection of all predecessors
//     (none of which is a back-edge), or into the regions of a cc.if, cc.scope
//     or cc.loop from the state in front of the operation.
//   - Facts that flow into a cc.loop are first pruned of everything the loop
//     might write, so that they hold on every iteration.
//   - A fact never flows back out of a region or a loop body. After a region
//     operation the facts that survive are those that nothing in the region
//     may have written.
//   - The header of a natural loop in CFG form (found with CFGLoopInfo) is
//     treated like a cc.loop: what holds on entry to the loop is pruned of
//     everything that any block of the loop may write. Back-edges of anything
//     that is not a natural loop leave nothing known.
//   - A stack slot is private, and so safe from calls and from stores through
//     other pointers, until the program point where its address may first be
//     handed out. An operation that leaks the address only affects the code
//     that may execute after it. This is only worked out for code that is
//     executed at most once, in order: in a region of several blocks, or in a
//     loop, any operation that leaks the address counts as having done so from
//     the start.
//   - Operations that are not loads or stores are assumed to read and write
//     only the memory that they declare, through MemoryEffectOpInterface, to
//     touch. Anything that reads or writes memory therefore must declare it.
//   - A store that was seen before a branch, a region or any other join point
//     is never removed as being overwritten, since the overwriting store is
//     not executed on every path (control may also leave the function).
//===----------------------------------------------------------------------===//

/// The bounds that keep the work done per operation constant. These are the
/// options of the pass.
struct Limits {
  /// Maximum number of memory locations whose contents are tracked at once.
  unsigned trackedLocations;
  /// Maximum number of levels of operations that are compared when deciding
  /// that two addresses are equal. Zero means only identical values are.
  unsigned addressDepth;
  /// Maximum number of operations that leak the address of a stack slot that
  /// are tracked. A slot that is leaked more often is treated as leaked from
  /// the start of the function.
  unsigned escapingUses;
};

/// Is \p a certainly the same address as \p b? Both must live in the same
/// block or in blocks that dominate the use, so that SSA identity is
/// meaningful.
bool sameAddress(Value a, Value b, unsigned maxDepth, unsigned depth = 0) {
  if (a == b)
    return true;
  if (depth >= maxDepth)
    return false;
  auto *da = a.getDefiningOp();
  auto *db = b.getDefiningOp();
  if (!da || !db || da->getName() != db->getName() || !isPure(da) ||
      !isPure(db))
    return false;
  if (cast<OpResult>(a).getResultNumber() !=
      cast<OpResult>(b).getResultNumber())
    return false;
  return OperationEquivalence::isEquivalentTo(
      da, db,
      [&](Value x, Value y) -> LogicalResult {
        return success(sameAddress(x, y, maxDepth, depth + 1));
      },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

/// Strip the pointer arithmetic from \p ptr to find the object it points into.
Value getRoot(Value ptr) {
  while (true) {
    if (auto cast = ptr.getDefiningOp<cudaq::cc::CastOp>()) {
      if (!isa<cudaq::cc::PointerType>(cast.getValue().getType()))
        return ptr;
      ptr = cast.getValue();
    } else if (auto cp = ptr.getDefiningOp<cudaq::cc::ComputePtrOp>()) {
      ptr = cp.getBase();
    } else {
      return ptr;
    }
  }
}

class Forwarder {
public:
  Forwarder(DominanceInfo &dom, const Limits &limits)
      : dom(dom), limits(limits) {}

  /// Process the body of a function. Nothing is known on entry.
  void processFunction(Region &body) { processRegion(body, {}); }

private:
  /// A memory location whose contents are known: the value that is in memory
  /// and, if it came from a store that may still be removed, that store.
  ///
  /// The state is copied freely (into the exit state of a block, into the state
  /// inside a region, along each edge). A store is erased when it is found to
  /// be overwritten, so a copy of the state must not hold on to a store unless
  /// that copy is the only one that is ever going to see it overwritten.
  /// Whenever the state is copied to more than one place, or control may leave
  /// without reaching the next operation, the stores are cleared.
  struct MemoryFact {
    Value address;
    Value value;
    cudaq::cc::StoreOp store; // null when the store may not be removed.
    bool observed = false;    // might the stored value have been read?
  };
  using State = SmallVector<MemoryFact>;

  enum class Relation { Same, Disjoint, MayAlias };

  /// What a region operation might do to memory, in total.
  struct Summary {
    /// Something that is not a plain load or store may read or write any memory
    /// that is not a private stack slot.
    bool clobbersAll = false;
    /// Too many different stores to list. Any memory may have been written.
    bool writesAny = false;
    /// The addresses of the stores.
    SmallVector<Value> writes;
  };

  //===--------------------------------------------------------------------===//
  // Regions and blocks.
  //===--------------------------------------------------------------------===//

  /// Process all of the blocks of \p region. \p entry is what is known at the
  /// start of the entry block.
  void processRegion(Region &region, State entry) {
    if (region.empty())
      return;
    if (region.hasOneBlock()) {
      processBlock(region.front(), entry);
      return;
    }
    DenseMap<Block *, State> exits;
    CFGLoopInfo loops(dom.getDomTree(&region));
    llvm::ReversePostOrderTraversal<Region *> order(&region);
    for (Block *block : order) {
      State state =
          block == &region.front() ? entry : join(*block, exits, loops);
      processBlock(*block, state);
      exits[block] = std::move(state);
    }
    // Blocks that cannot be reached know nothing.
    for (Block &block : region)
      if (!exits.count(&block)) {
        State state;
        processBlock(block, state);
      } // The loops are about to go away, and others may be put at their
        // addresses.
    loopSummaries.clear();
  }

  /// What is known on entry to \p block, which is not the entry block of its
  /// region, given what was known at the end of the blocks that precede it.
  State join(Block &block, DenseMap<Block *, State> &exits,
             CFGLoopInfo &loops) {
    SmallVector<Block *> preds;
    for (Block *pred : block.getPredecessors())
      if (!llvm::is_contained(preds, pred))
        preds.push_back(pred);
    // If this block is the header of a natural loop, then the predecessors that
    // are in the loop are back-edges. The header's block arguments merge the
    // values from the loop's entry with those of the previous iteration.
    CFGLoop *loop = loops.getLoopFor(&block);
    if (loop && loop->getHeader() != &block)
      loop = nullptr;
    SmallVector<Block *> incoming;
    for (Block *pred : preds) {
      if (loop && loop->contains(pred))
        continue;
      // A predecessor that has not been processed yet is a back-edge of a loop
      // that is not natural (or is not reducible). Know nothing.
      if (!exits.count(pred))
        return {};
      incoming.push_back(pred);
    }
    if (incoming.empty())
      return {};
    auto edgeState = [&](Block *pred) {
      State state = exits[pred];
      // If the predecessor can branch elsewhere then the store in it is not
      // overwritten on all paths.
      if (pred->getTerminator()->getNumSuccessors() != 1)
        for (auto &fact : state)
          fact.store = cudaq::cc::StoreOp{};
      return state;
    };
    State state = edgeState(incoming.front());
    for (Block *pred : llvm::drop_begin(incoming))
      state = intersect(state, edgeState(pred), block);
    if (incoming.size() > 1 || loop)
      for (auto &fact : state)
        fact.store = cudaq::cc::StoreOp{};
    if (loop) {
      // What holds on entry to the loop holds on every iteration only if
      // nothing in the loop may change it.
      apply(summarizeLoop(loop, loops), state, &block.front());
    }
    return state;
  }

  /// Keep what is known along both \p a and \p b, which are the states on
  /// entry to \p block from two of its predecessors. A block may have any
  /// number of predecessors; the states of all of them are combined by folding
  /// this over them (see join), which is sound as the intersection is
  /// associative.
  State intersect(const State &a, const State &b, Block &block) {
    State result;
    Operation *start = &block.front();
    for (const MemoryFact &x : a)
      for (const MemoryFact &y : b)
        if (x.value == y.value &&
            compare(x.address, y.address, start) == Relation::Same &&
            dom.properlyDominates(x.address, start) &&
            dom.properlyDominates(x.value, start)) {
          result.push_back({x.address, x.value, cudaq::cc::StoreOp{}, false});
          break;
        }
    return result;
  }

  /// Process the operations of \p block. \p state is what is known on entry,
  /// and is what is known on exit when this returns.
  void processBlock(Block &block, State &state) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (auto load = dyn_cast<cudaq::cc::LoadOp>(&op))
        visitLoad(state, load);
      else if (auto store = dyn_cast<cudaq::cc::StoreOp>(&op))
        visitStore(state, store);
      else if (isa<FunctionOpInterface>(&op))
        continue; // Declaring a function does not execute it.
      else if (op.getNumRegions() != 0)
        visitRegionOp(state, &op);
      else if (mayAccessMemory(&op))
        clobberAll(state, &op);
    }
  }

  //===--------------------------------------------------------------------===//
  // Operations.
  //===--------------------------------------------------------------------===//

  void visitLoad(State &state, cudaq::cc::LoadOp load) {
    Value addr = load.getPtrvalue();
    for (auto &fact : state) {
      Relation rel = compare(fact.address, addr, load);
      if (rel == Relation::Same && fact.value.getType() == load.getType()) {
        // Store-load forwarding or redundant load. This load does not read
        // memory any more, so it does not observe the store.
        LLVM_DEBUG(llvm::dbgs()
                   << "forwarding " << fact.value << " to " << load << '\n');
        load.getResult().replaceAllUsesWith(fact.value);
        load.erase();
        ++numLoadsRemoved;
        return;
      }
    }
    // The load reads memory. Any store that may feed it is observed.
    for (auto &fact : state)
      if (compare(fact.address, addr, load) != Relation::Disjoint)
        fact.observed = true;
    remember(state, {addr, load.getResult(), cudaq::cc::StoreOp{}, false});
  }

  void visitStore(State &state, cudaq::cc::StoreOp store) {
    Value addr = store.getPtrvalue();
    Value val = store.getValue();
    llvm::erase_if(state, [&](MemoryFact &fact) {
      Relation rel = compare(fact.address, addr, store);
      // An earlier store to exactly this location, with no intervening read,
      // is completely overwritten (the types must agree so that all of the
      // bytes are overwritten).
      if (rel == Relation::Same && fact.store && !fact.observed &&
          fact.value.getType() == val.getType()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "removing overwritten " << fact.store << '\n');
        fact.store.erase();
        ++numStoresRemoved;
      }
      return rel != Relation::Disjoint;
    });
    remember(state, {addr, val, store, false});
  }

  /// An operation with regions. What is known inside of it, and after it, is
  /// derived from what is known in front of it.
  void visitRegionOp(State &state, Operation *op) {
    const Summary &summary = summarize(op);
    State inner;
    if (isa<cudaq::cc::IfOp, cudaq::cc::ScopeOp>(op)) {
      inner = state;
    } else if (isa<cudaq::cc::LoopOp>(op)) {
      // The regions of a loop are executed repeatedly. What is known inside is
      // only what no iteration may change.
      inner = state;
      apply(summary, inner, op);
    } // else: the semantics are unknown, so nothing is known inside.
    for (auto &fact : inner)
      fact.store = cudaq::cc::StoreOp{};
    for (Region &region : op->getRegions())
      processRegion(region, inner);
    // Nothing that is known inside a region is known afterwards. What is known
    // in front of the operation still is, unless the operation changed it. The
    // operation may also be left early, so a store before it is not
    // necessarily overwritten by a store after it.
    apply(summary, state, op);
    for (auto &fact : state)
      fact.store = cudaq::cc::StoreOp{};
  }

  /// Add \p write to the stores that \p summary lists.
  void addWrite(Summary &summary, Value write) {
    if (summary.writesAny || llvm::is_contained(summary.writes, write))
      return;
    if (summary.writes.size() >= maxSummaryWrites()) {
      summary.writesAny = true;
      summary.writes.clear();
      return;
    }
    summary.writes.push_back(write);
  }

  void merge(Summary &into, const Summary &from) {
    into.clobbersAll |= from.clobbersAll;
    if (from.writesAny) {
      into.writesAny = true;
      into.writes.clear();
      return;
    }
    for (Value w : from.writes)
      addWrite(into, w);
  }

  /// Add what \p op, and anything nested in it, might do to memory.
  void summarizeInto(Operation *op, Summary &summary) {
    if (isa<FunctionOpInterface>(op))
      return; // Declaring a function does not execute it.
    if (auto store = dyn_cast<cudaq::cc::StoreOp>(op)) {
      addWrite(summary, store.getPtrvalue());
      return;
    }
    if (isa<cudaq::cc::LoadOp>(op))
      return;
    if (op->getNumRegions() != 0)
      merge(summary, summarize(op));
    else if (mayAccessMemory(op))
      summary.clobbersAll = true;
  }

  /// What \p op, and anything nested in it, might do to memory. Every
  /// operation is summarized once, from the summaries of the operations nested
  /// in it, which are kept.
  const Summary &summarize(Operation *op) {
    auto it = summaries.find(op);
    if (it != summaries.end())
      return it->second;
    Summary summary;
    if (mayAccessMemory(op))
      summary.clobbersAll = true;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          summarizeInto(&nested, summary);
    // Summarizing the nested operations may have rehashed the map.
    return summaries[op] = std::move(summary);
  }

  /// What the blocks of \p loop, including those of the loops nested in it,
  /// might do to memory.
  const Summary &summarizeLoop(CFGLoop *loop, CFGLoopInfo &loops) {
    auto it = loopSummaries.find(loop);
    if (it != loopSummaries.end())
      return it->second;
    Summary summary;
    for (CFGLoop *sub : loop->getSubLoops())
      merge(summary, summarizeLoop(sub, loops));
    for (Block *member : loop->getBlocks())
      if (loops.getLoopFor(member) == loop)
        for (Operation &op : *member)
          summarizeInto(&op, summary);
    return loopSummaries[loop] = std::move(summary);
  }

  /// A summary that lists more stores than this says that anything may have
  /// been written, which keeps what is done with a summary constant.
  unsigned maxSummaryWrites() const { return 2 * limits.trackedLocations; }

  /// Forget everything that \p summary says might have changed.
  void apply(const Summary &summary, State &state, Operation *at) {
    if (summary.writesAny) {
      state.clear();
      return;
    }
    llvm::erase_if(state, [&](MemoryFact &fact) {
      if (summary.clobbersAll && !isPrivateSlot(fact.address, at))
        return true;
      return llvm::any_of(summary.writes, [&](Value w) {
        return compare(fact.address, w, at) != Relation::Disjoint;
      });
    });
  }

  /// Forget everything that something which is not a load or a store may have
  /// changed. Stack slots whose address is not available to it are safe.
  void clobberAll(State &state, Operation *at) {
    llvm::erase_if(state, [&](MemoryFact &fact) {
      return !isPrivateSlot(fact.address, at);
    });
  }

  void remember(State &state, MemoryFact fact) {
    if (limits.trackedLocations == 0)
      return;
    if (state.size() >= limits.trackedLocations)
      state.erase(state.begin());
    state.push_back(fact);
  }

  //===--------------------------------------------------------------------===//
  // Memory.
  //===--------------------------------------------------------------------===//

  /// Might \p op, ignoring the operations nested in it, read or write memory
  /// that is visible to loads and stores?
  static bool mayAccessMemory(Operation *op) {
    if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      iface.getEffects(effects);
      for (auto &e : effects)
        if (isa<MemoryEffects::Read, MemoryEffects::Write>(e.getEffect()))
          return true;
      return false;
    }
    // Operations that only have effects through their regions are examined
    // through those regions.
    return !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  }

  /// Is \p address in a stack slot whose address is not available, when \p at
  /// executes, to anything but the loads and stores that use it directly?
  bool isPrivateSlot(Value address, Operation *at) {
    auto alloca = getRoot(address).getDefiningOp<cudaq::cc::AllocaOp>();
    return alloca && !escapedAt(alloca, at);
  }

  /// The ways that the address of a stack slot may be made available to code
  /// that this analysis cannot see.
  struct Escapes {
    /// The address escapes in a way that cannot be placed in the program.
    bool always = false;
    /// Operations that use the address (or a pointer derived from it) for
    /// something other than loading from it or storing to it.
    SmallVector<Operation *> uses;
  };

  const Escapes &getEscapes(cudaq::cc::AllocaOp alloca) {
    auto [iter, inserted] = escapes.try_emplace(alloca.getOperation());
    Escapes &info = iter->second;
    if (!inserted)
      return info;
    SmallVector<Value> work = {alloca.getResult()};
    while (!work.empty() && !info.always) {
      Value v = work.pop_back_val();
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();
        if (isa<cudaq::cc::LoadOp>(user))
          continue;
        if (auto store = dyn_cast<cudaq::cc::StoreOp>(user)) {
          // Storing the address itself, rather than storing to it, escapes. The
          // store may be erased by this analysis, so don't keep track of it.
          if (store.getValue() == v)
            info.always = true;
          continue;
        }
        if (isa<cudaq::cc::CastOp, cudaq::cc::ComputePtrOp>(user) &&
            isa<cudaq::cc::PointerType>(user->getResult(0).getType())) {
          work.push_back(user->getResult(0));
          continue;
        }
        info.uses.push_back(user);
        if (info.uses.size() > limits.escapingUses)
          info.always = true;
      }
    }
    return info;
  }

  /// Might the address of \p alloca be available to something other than the
  /// loads and stores that use it directly, by the time \p at executes? That is
  /// the case when an operation that makes it available may have executed
  /// before \p at, otherwise nothing can have gotten hold of the address yet.
  bool escapedAt(cudaq::cc::AllocaOp alloca, Operation *at) {
    const Escapes &info = getEscapes(alloca);
    if (info.always)
      return true;
    return llvm::any_of(info.uses,
                        [&](Operation *use) { return mayPrecede(use, at); });
  }

  /// Might \p first execute before \p second?
  static bool mayPrecede(Operation *first, Operation *second) {
    // Find the innermost block that holds both, or ancestors of both.
    for (Block *block = second->getBlock(); block;) {
      if (Operation *f = block->findAncestorOpInBlock(*first)) {
        Operation *s = block->findAncestorOpInBlock(*second);
        // If one is nested in the other, or the code may be executed again,
        // then the order in the block does not say anything.
        if (f == s || !isExecutedOnce(block))
          return true;
        return f->isBeforeInBlock(s);
      }
      Operation *parent = block->getParentOp();
      block = parent ? parent->getBlock() : nullptr;
    }
    return true;
  }

  /// Is every operation in \p block executed at most once per call of the
  /// function, and in the order that they appear?
  static bool isExecutedOnce(Block *block) {
    for (Region *region = block->getParent(); region;) {
      if (!region->hasOneBlock())
        return false;
      Operation *parent = region->getParentOp();
      if (!parent || isa<FunctionOpInterface>(parent))
        return true;
      if (!isa<cudaq::cc::IfOp, cudaq::cc::ScopeOp>(parent))
        return false;
      region = parent->getParentRegion();
    }
    return true;
  }

  /// How are the locations that \p a and \p b point to related?
  Relation compare(Value a, Value b, Operation *at) {
    if (a == b)
      return Relation::Same;
    auto ca = a.getDefiningOp<cudaq::cc::ComputePtrOp>();
    auto cb = b.getDefiningOp<cudaq::cc::ComputePtrOp>();
    if (ca && cb && ca.getBase() == cb.getBase() &&
        ca.getDynamicIndices().empty() && cb.getDynamicIndices().empty()) {
      // Constant subobject indices off the very same base. This is by far the
      // most common case, so don't do any more work. Indices that are out of
      // range are not valid, so different indices are different locations.
      auto ia = ca.getRawConstantIndices();
      auto ib = cb.getRawConstantIndices();
      if (ia == ib)
        return Relation::Same;
      return ia.size() == ib.size() ? Relation::Disjoint : Relation::MayAlias;
    }
    Value ra = getRoot(a);
    Value rb = getRoot(b);
    auto aa = ra.getDefiningOp<cudaq::cc::AllocaOp>();
    auto ab = rb.getDefiningOp<cudaq::cc::AllocaOp>();
    // Distinct stack objects. This is the most common question, so answer it
    // before doing any more work.
    if (aa && ab && aa != ab)
      return Relation::Disjoint;
    if (sameAddress(a, b, limits.addressDepth))
      return Relation::Same;
    if (ra == rb)
      return Relation::MayAlias;
    if ((aa && !escapedAt(aa, at)) || (ab && !escapedAt(ab, at)))
      return Relation::Disjoint;
    return Relation::MayAlias;
  }

  DominanceInfo &dom;
  Limits limits;
  /// Operations that leak a stack slot are never erased by this analysis, so
  /// these may be kept.
  DenseMap<Operation *, Escapes> escapes;
  DenseMap<Operation *, Summary> summaries;
  /// The summaries of the loops of the region being processed.
  DenseMap<CFGLoop *, Summary> loopSummaries;
};

class DSRPattern : public OpRewritePattern<cudaq::cc::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // If we have a cc.alloca and all of its uses are cc.store ops, then these are
  // dead stores. Anchoring on the alloca rather than on each store means the
  // alloca's users are scanned once, instead of once per store.
  LogicalResult matchAndRewrite(cudaq::cc::AllocaOp alloca,
                                PatternRewriter &rewriter) const override {
    SmallVector<cudaq::cc::StoreOp> stores;
    // At least one store must be directly to the alloca, through a cast, or
    // through a compute_ptr with constant offsets.
    bool hasCandidate = false;
    auto testAllStoreUsers = [&](Operation *c, bool isCandidate) {
      for (auto v : c->getUsers()) {
        if (auto s = dyn_cast<cudaq::cc::StoreOp>(v)) {
          // Make sure this stores *to* the address rather stores the address.
          if (s.getPtrvalue() == c->getResult(0)) {
            stores.push_back(s);
            hasCandidate |= isCandidate;
            continue;
          }
        }
        return false;
      }
      return true;
    };

    for (auto u : alloca->getUsers()) {
      if (auto c = dyn_cast<cudaq::cc::CastOp>(u)) {
        if (!testAllStoreUsers(c, /*isCandidate=*/true)) {
          LLVM_DEBUG(llvm::dbgs() << "store not from cast of alloca.\n");
          return failure();
        }
        continue;
      }
      if (auto c = dyn_cast<cudaq::cc::ComputePtrOp>(u)) {
        if (!testAllStoreUsers(c, c.getNumOperands() == 1)) {
          LLVM_DEBUG(llvm::dbgs() << "store not from compute_ptr of alloca.\n");
          return failure();
        }
        continue;
      }

      if (auto s = dyn_cast<cudaq::cc::StoreOp>(u))
        if (s.getPtrvalue() == alloca.getResult()) {
          stores.push_back(s);
          hasCandidate = true;
          continue;
        }
      LLVM_DEBUG(llvm::dbgs() << "alloca use is not store/cast/compute_ptr.\n");
      return failure();
    }
    if (!hasCandidate)
      return failure();
    for (auto s : stores)
      rewriter.eraseOp(s);
    return success();
  }
};

class DSRPass : public cudaq::opt::impl::DeadStoreRemovalBase<DSRPass> {
public:
  using DeadStoreRemovalBase::DeadStoreRemovalBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before erasure:\n" << *op << "\n\n");
    // Forward stores to loads and remove overwritten stores.
    DominanceInfo domInfo(op);
    Forwarder forwarder(domInfo, {.trackedLocations = maxTrackedLocations,
                                  .addressDepth = maxAddressDepth,
                                  .escapingUses = maxEscapingUses});
    op->walk([&](FunctionOpInterface func) {
      if (!func.isExternal())
        forwarder.processFunction(func.getFunctionBody());
    });
    // Then remove the stores to stack slots that are never read.
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<DSRPattern>(ctx);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After erasure:\n" << *op << "\n\n");
  }
};
} // namespace
