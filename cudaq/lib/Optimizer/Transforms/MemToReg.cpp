/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Dominance.h"
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

static bool isLinearType(Value v) {
  return cudaq::quake::isLinearType(v.getType());
}

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

private:
  bool nonEscapingDef(Operation *use, Value result) {
    // Return false if not a def.
    if (!isMemoryDef(use))
      return false;
    // Return true if not classical.
    if (!result)
      return true;
    // Check that the address doesn't escape by storing it to a variable.
    if (auto st = dyn_cast<cudaq::cc::StoreOp>(use))
      return st.getValue() != result;
    // Default assume this one escapes.
    return false;
  }

  void determineAllocSet(func::FuncOp func) {
    SmallVector<Operation *> allocations;
    auto qrefTy = cudaq::quake::RefType::get(func.getContext());
    func->walk([&](Operation *op) {
      if (isMemoryAlloc(op)) {
        // Make sure this is stack here. Can we make use of an Interface?
        if (auto alloc = dyn_cast<cudaq::quake::AllocaOp>(op)) {
          if (!alloc.hasInitializedState() && alloc.getType() == qrefTy)
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
      Value v;
      if (auto alloc = dyn_cast<cudaq::cc::AllocaOp>(a))
        v = alloc.getResult();
      for (auto *u : a->getUsers()) {
        // Don't convert quake.custom unitary ops as they have ambiguous
        // semantics.
        //
        // cc.instantiate_callable always escapes any pointer-typed operand it
        // captures: the closure holds onto that raw address for later
        // dereference from a completely different function, not a load-like
        // use here. isMemoryUse's op-level MemoryEffectOpInterface check
        // can't tell escaping capture operands apart from ordinary ones.
        // InstantiateCallableOp::getEffects reports a blanket Read (to keep
        // CSE from merging distinct instantiations of a closure that
        // captures a quantum reference; see its definition in CCOps.cpp),
        // which makes isMemoryUse return true for the whole op regardless of
        // which operand is being examined, so a classical alloca captured
        // alongside an unrelated quantum capture would otherwise look like a
        // harmless load and get promoted out from under the closure.
        if (isa<cudaq::quake::CustomUnitaryCallOp,
                cudaq::quake::CustomUnitaryConstantOp,
                cudaq::cc::InstantiateCallableOp>(u) ||
            (!isMemoryUse(u) && !nonEscapingDef(u, v))) {
          add = nullptr;
          break;
        }
      }
      if (add)
        allocSet.insert(add);
    }
  }

  SmallPtrSet<Operation *, 4> allocSet;
};
} // namespace

static bool opResultOfType(Operation *op, Type ofTy) {
  if (op->getNumResults() == 0)
    return false;
  return llvm::any_of(op->getResultTypes(),
                      [ofTy](Type t) { return t == ofTy; });
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
  if (isa<cudaq::quake::RefType>(ty))
    return cudaq::quake::WireType::get(ty.getContext());
  return cast<cudaq::cc::PointerType>(ty).getElementType();
}

/// Returns the element index of \p ext, if statically known. This inspects both
/// the trivial attribute case as well as the case when the SSA value is itself
/// a constant operation to decouple from canonicalization.
static std::optional<std::size_t>
constantExtractIndex(cudaq::quake::ExtractRefOp ext) {
  if (ext.hasConstantIndex())
    return ext.getConstantIndex();
  if (auto v = cudaq::opt::factory::getIntIfConstant(ext.getIndex()))
    if (*v >= 0)
      return static_cast<std::size_t>(*v);
  return std::nullopt;
}

/// Return the lower bound of \p sub, if statically known. Same rationale as
/// constantExtractIndex.
static std::optional<std::size_t>
constantSubVeqLower(cudaq::quake::SubVeqOp sub) {
  if (sub.hasConstantLowerBound())
    return sub.getConstantLowerBound();
  if (auto v = cudaq::opt::factory::getIntIfConstant(sub.getLower()))
    if (*v >= 0)
      return static_cast<std::size_t>(*v);
  return std::nullopt;
}

/// Peel a chain of `quake.subveq`/`quake.relax_size` views off \p veq to find
/// the underlying veq it is ultimately a view of (an alloca, init_state
/// result, function/block argument, or any other op result that isn't itself
/// a further view), accumulating the element offset of the view within that
/// root. Returns nullopt when a view's offset is not a compile-time constant,
/// i.e. the view's position within the root is unknown.
static std::optional<std::pair<Value, std::size_t>> resolveVeqBase(Value veq) {
  std::size_t offset = 0;
  while (true) {
    if (auto sub = veq.getDefiningOp<cudaq::quake::SubVeqOp>()) {
      auto lo = constantSubVeqLower(sub);
      if (!lo)
        return std::nullopt;
      offset += *lo;
      veq = sub.getVeq();
      continue;
    }
    if (auto relax = veq.getDefiningOp<cudaq::quake::RelaxSizeOp>()) {
      veq = relax.getInputVec();
      continue;
    }
    return std::make_pair(veq, offset);
  }
}

/// The number of qubits \p veq spans, if statically known.
static std::optional<std::size_t> veqExtent(Value veq) {
  return cudaq::quake::getVeqSize(veq);
}

namespace {
/// A qubit's abstract location: a storage root plus an element index within
/// it. Two distinct SSA `!quake.ref` values that resolve to the same location
/// name the same physical qubit.
using QubitLoc = std::pair<Value, std::size_t>;

/// Resolve the abstract location \p ref names, if it has one. Returns nullopt
/// when the position is not statically knowable, which callers must treat as
/// "may be any qubit in the root".
static std::optional<QubitLoc> resolveRefLocation(Value ref) {
  if (auto *def = ref.getDefiningOp()) {
    if (auto alloc = dyn_cast<cudaq::quake::AllocaOp>(def))
      if (isa<cudaq::quake::RefType>(alloc.getType()))
        return QubitLoc{alloc, 0}; // a standalone qubit is its own root
    if (auto ext = dyn_cast<cudaq::quake::ExtractRefOp>(def)) {
      auto idx = constantExtractIndex(ext);
      if (!idx)
        return std::nullopt;
      auto base = resolveVeqBase(ext.getVeq());
      if (!base)
        return std::nullopt;
      return QubitLoc{base->first, base->second + *idx};
    }
    return std::nullopt;
  }
  // A ref-typed block argument (notably a function parameter) is its own
  // single-qubit root.
  return QubitLoc{ref, 0};
}

/// Merge `quake.extract_ref` ops that name the same qubit.
///
/// A `!quake.ref` is a reference-to-a-wire and this pass keys its bindings on
/// the SSA ref value, so two distinct refs naming one qubit would be tracked as
/// two independent memory locations, issuing both loads before either store and
/// silently dropping the first gate. `extract_ref` is Pure, so CSE merges them,
/// but depending on that makes correctness a property of pass ordering rather
/// than of this pass. Do it here instead, so memtoreg is correct on whatever IR
/// it is handed.
///
/// Only merges into a reference that dominates the ones it replaces. Where no
/// such reference exists the duplicates are left alone and the analysis below
/// blacklists the location.
static void dedupQuantumRefs(func::FuncOp func) {
  llvm::MapVector<QubitLoc, SmallVector<Value, 2>> byLoc;
  func.walk([&](cudaq::quake::ExtractRefOp ext) {
    if (auto loc = resolveRefLocation(ext.getResult()))
      byLoc[*loc].push_back(ext.getResult());
  });
  if (llvm::none_of(byLoc, [](auto &e) { return e.second.size() > 1; }))
    return;

  DominanceInfo dom(func);
  for (auto &[loc, refs] : byLoc) {
    if (refs.size() < 2)
      continue;
    Value rep;
    for (Value cand : refs)
      if (llvm::all_of(refs, [&](Value other) {
            return other == cand ||
                   dom.properlyDominates(cand, other.getDefiningOp());
          })) {
        rep = cand;
        break;
      }
    if (!rep)
      continue;
    for (Value r : refs)
      if (r != rep) {
        LLVM_DEBUG(llvm::dbgs()
                   << "memtoreg: merging duplicate reference " << r << '\n');
        r.replaceAllUsesWith(rep);
        r.getDefiningOp()->erase();
      }
  }
}

/// Determines, for every `!quake.ref` value in a function, the abstract qubit
/// location it names and whether that location may be promoted to a wire.
///
/// A `!quake.ref` behaves as a reference-to-a-wire: `quake.unwrap` loads and
/// `quake.wrap` stores. Promoting a location to SSA wires is only valid while
/// nothing *else* reaches that qubit through memory, because such an access
/// would read (or write) the cell while the live value sits in a register.
///
/// So each root is scanned for accesses that touch its elements in memory
/// form, and each contributes the element range it spans:
///
///   - a whole-veq operand to a gate/measure/call/init_state/closure spans
///     that view's extent (a `quake.subveq %q, 0, 1` used as a control spans
///     [0,2) of %q, not all of %q -- this is compact notation for "every
///     element of this view", not an aliasing event);
///   - a dynamic-index `quake.extract_ref`, or any view whose offset or
///     extent is not statically known, spans the whole root;
///   - a constant-index `quake.extract_ref` spans nothing: it names one
///     location, which is what we are trying to promote.
///
/// A location is promotable if and only if no such range covers it.
/// Transparency must be proven here. An op this analysis does not recognize is
/// treated as spanning the whole root, so an unmodelled operation makes the
/// output less optimized, never wrong.
class QuantumRefAnalysis {
public:
  explicit QuantumRefAnalysis(func::FuncOp f) { compute(f); }

  /// True if \p ref must be left in memory (reference) form.
  bool isBlacklisted(Value ref) const { return blacklist.count(ref); }

  /// The location \p ref names, or nullopt if it could not be resolved.
  std::optional<QubitLoc> locationOf(Value ref) const {
    auto it = locs.find(ref);
    if (it == locs.end())
      return std::nullopt;
    return it->second;
  }

  /// Every ref value that resolves to \p loc, in program order.
  ArrayRef<Value> refsAt(QubitLoc loc) const {
    auto it = refsAtLoc.find(loc);
    return it == refsAtLoc.end() ? ArrayRef<Value>{}
                                 : ArrayRef<Value>(it->second);
  }

  /// Counts for the pass statistics.
  std::size_t numBlacklisted() const { return blacklist.size(); }
  std::size_t numLocations() const { return refsAtLoc.size(); }

  /// True if every element of veq-typed \p root is promotable, so the root
  /// itself can be replaced by per-element wires.
  bool rootFullyPromotable(Value root) const {
    auto extent = veqExtent(root);
    if (!extent)
      return false;
    auto it = opaque.find(root);
    if (it == opaque.end())
      return true;
    return it->second.empty();
  }

private:
  /// Record that [lo, lo+len) of \p root is accessed in memory form.
  void markOpaque(Value root, std::size_t lo, std::optional<std::size_t> len) {
    auto &ranges = opaque[root];
    if (!len) {
      // Unknown extent: the access may touch anything in the root.
      ranges.assign(1, std::make_pair(std::size_t{0},
                                      std::numeric_limits<std::size_t>::max()));
      return;
    }
    ranges.emplace_back(lo, lo + *len);
  }

  bool isCovered(Value root, std::size_t index) const {
    auto it = opaque.find(root);
    if (it == opaque.end())
      return false;
    for (auto [lo, hi] : it->second)
      if (index >= lo && index < hi)
        return true;
    return false;
  }

  /// Walk every use of \p veq (a view at \p offset within \p root) and record
  /// the memory-form accesses it exposes.
  void scanVeqUses(Value root, Value veq, std::size_t offset,
                   SmallPtrSetImpl<Operation *> &visited) {
    for (Operation *user : veq.getUsers()) {
      if (isa<cudaq::quake::DeallocOp>(user))
        continue;
      if (auto ext = dyn_cast<cudaq::quake::ExtractRefOp>(user)) {
        if (constantExtractIndex(ext))
          continue; // names a single location; not a range
        markOpaque(root, 0, std::nullopt);
        continue;
      }
      if (auto sub = dyn_cast<cudaq::quake::SubVeqOp>(user)) {
        auto lo = constantSubVeqLower(sub);
        if (!lo) {
          markOpaque(root, 0, std::nullopt);
          continue;
        }
        if (visited.insert(sub).second)
          scanVeqUses(root, sub.getResult(), offset + *lo, visited);
        continue;
      }
      if (auto relax = dyn_cast<cudaq::quake::RelaxSizeOp>(user)) {
        if (visited.insert(relax).second)
          scanVeqUses(root, relax.getResult(), offset, visited);
        continue;
      }
      // Anything else -- a gate with a veq control or broadcast target, a
      // measure, a call, init_state, a concat feeding elsewhere, a closure
      // capture, or an op we simply do not model -- reaches these qubits
      // through memory. It spans this view's extent.
      LLVM_DEBUG({
        llvm::dbgs() << "memtoreg: memory-form access forces ";
        if (auto n = veqExtent(veq))
          llvm::dbgs() << "[" << offset << ", " << (offset + *n) << ")";
        else
          llvm::dbgs() << "all";
        llvm::dbgs() << " of ";
        root.printAsOperand(llvm::dbgs(), OpPrintingFlags());
        llvm::dbgs() << " into memory form, due to: " << *user << '\n';
      });
      markOpaque(root, offset, veqExtent(veq));
    }
  }

  void compute(func::FuncOp f) {
    // 1. gather every ref and veq typed SSA value. Op results and block
    // arguments are exhaustive.  An SSA value has no other origin.
    // Set-vectors: `walk` visits the function op itself as well as its nested
    // ops, so a plain vector would collect the entry block's arguments twice
    // and the duplicate-location rule below would then blacklist every
    // reference-typed function parameter.
    SetVector<Value> refs;
    SetVector<Value> veqs;
    auto note = [&](Value v) {
      if (isa<cudaq::quake::RefType>(v.getType()))
        refs.insert(v);
      else if (isa<cudaq::quake::VeqType>(v.getType()))
        veqs.insert(v);
    };
    for (auto arg : f.getArguments())
      note(arg);
    f.walk([&](Operation *op) {
      for (Value r : op->getResults())
        note(r);
      for (auto &region : op->getRegions())
        for (auto &b : region)
          for (auto arg : b.getArguments())
            note(arg);
    });

    // 2. for each veq that is a root (not itself a view), scan its uses to
    // build the set of memory-form access ranges.
    for (Value veq : veqs) {
      auto base = resolveVeqBase(veq);
      if (!base || base->first != veq)
        continue; // a view; scanned via its root
      SmallPtrSet<Operation *, 8> visited;
      scanVeqUses(veq, veq, 0, visited);
    }

    // 3. classify each ref. An unresolvable location, or a location covered by
    // a memory-form access, means the ref stays in memory form.
    for (Value ref : refs) {
      auto loc = resolveRefLocation(ref);
      if (!loc || isCovered(loc->first, loc->second)) {
        blacklist.insert(ref);
        continue;
      }
      locs[ref] = *loc;
      refsAtLoc[*loc].push_back(ref);
    }

    // A location still named by more than one reference is one that
    // deduplication could not merge (no single dominating reference). Tracking
    // either of them independently would reintroduce the split-state bug, so
    // leave the qubit in memory form.
    for (auto &[loc, refsHere] : refsAtLoc)
      if (refsHere.size() > 1)
        for (Value r : refsHere) {
          blacklist.insert(r);
          locs.erase(r);
        }
  }

  DenseSet<Value> blacklist;
  DenseMap<Value, QubitLoc> locs;
  DenseMap<QubitLoc, SmallVector<Value, 2>> refsAtLoc;
  /// Per root, the element ranges reached in memory form.
  DenseMap<Value, SmallVector<std::pair<std::size_t, std::size_t>, 2>> opaque;
};
} // namespace

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
    // Snapshot every block's argument count as it stood before this function
    // does anything. This may run any number of times, so whatever arguments
    // preexisted must stay exactly where they are; only arguments we add here
    // are free to be reordered for cross-region consistency (see
    // canonicalizeArgumentOrder).
    for (auto &region : op->getRegions())
      for (auto &b : region)
        originalArgCount[&b] = b.getNumArguments();

    // Stitch together the control-flow across op's regions.
    SmallPtrSet<Block *, 2> entryBlocks;
    SmallPtrSet<Block *, 2> exitBlocks;
    DenseMap<Block *, SmallPtrSet<Block *, 2>> reverseCFG;
    if (auto regionOp = dyn_cast<RegionBranchOpInterface>(op)) {
      SmallVector<RegionSuccessor> successors;
      regionOp.getSuccessorRegions(RegionBranchPoint::parent(), successors);
      for (auto iter : successors)
        if (iter.getSuccessor() && !iter.getSuccessor()->empty())
          entryBlocks.insert(&iter.getSuccessor()->front());
      for (auto &region : op->getRegions()) {
        if (region.empty())
          continue;
        SmallVector<Block *> regionExitBlocks;
        for (auto &b : region)
          if (b.hasNoSuccessors())
            regionExitBlocks.push_back(&b);
        for (auto *b : regionExitBlocks) {
          auto *terminator = b->getTerminator();
          SmallVector<RegionSuccessor> blockSuccessors;
          if (auto terminatorOp =
                  dyn_cast<RegionBranchTerminatorOpInterface>(terminator))
            regionOp.getSuccessorRegions(terminatorOp, blockSuccessors);
          if (blockSuccessors.empty()) {
            exitBlocks.insert(b);
            continue;
          }
          for (auto iter : blockSuccessors) {
            auto *succ = iter.getSuccessor();
            if (succ) {
              auto *s = &succ->front();
              reverseCFG[s].insert(b);
            } else {
              exitBlocks.insert(b);
            }
          }
        }
      }
    } else {
      for (auto &region : op->getRegions())
        for (auto &b : region) {
          if (b.isEntryBlock())
            entryBlocks.insert(&b);
          if (b.hasNoSuccessors())
            exitBlocks.insert(&b);
        }
    }
    entryCFG.append(entryBlocks.begin(), entryBlocks.end());
    exitCFG.append(exitBlocks.begin(), exitBlocks.end());
    for (auto [succBlk, predBlks] : reverseCFG) {
      auto &preds = backwardCFG[succBlk];
      for (Block *p : predBlks)
        if (!llvm::is_contained(preds, p))
          preds.push_back(p);
    }
  }

  //===--------------------------------------------------------------------===//
  // Cached CFG information.
  //
  // Since ops with regions can have a complex CFG structure that connects
  // blocks in different regions in non-trivial ways, we cache that CFG
  // structure here.
  //===--------------------------------------------------------------------===//

  bool isEntryBlock(Block *block) {
    return llvm::is_contained(entryCFG, block);
  }

  SmallVector<Block *> &getEntryBlocks() { return entryCFG; }

  bool isExitBlock(Block *block) { return llvm::is_contained(exitCFG, block); }

  SmallVector<Block *> &getExitBlocks() { return exitCFG; }

  SmallVector<Block *> &getPredecessors(Block *block) {
    if (backwardCFG.count(block))
      return backwardCFG[block];
    // The CFG is constant, so cache it for efficiency.
    if (!cachedPredCFG.count(block)) {
      auto range = block->getPredecessors();
      cachedPredCFG[block].append(range.begin(), range.end());
    }
    return cachedPredCFG[block];
  }

  /// Add \p block to the data-flow map for processing. This will add arguments
  /// to the block for any region arguments not already appended.
  void addBlock(Block *block) {
    assert(block);
    if (!rMap.count(block)) {
      rMap.insert({block, llvm::MapVector<MemRef, SSAReg>{}});
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
    assert(block && mr);
    auto blockIt = rMap.find(block);
    assert(blockIt != rMap.end());
    auto mrIt = blockIt->second.find(mr);
    assert(mrIt != blockIt->second.end());
    return mrIt->second;
  }

  /// Returns the binding for \p mr in \p block, or a null Value if not
  /// present or if the binding was cancelled.
  SSAReg lookupBinding(Block *block, MemRef mr) {
    assert(block && mr);
    auto blockIt = rMap.find(block);
    assert(blockIt != rMap.end());
    auto mrIt = blockIt->second.find(mr);
    return mrIt != blockIt->second.end() ? mrIt->second : SSAReg{};
  }

  /// Create a re-load of a memory reference. This can be used to place a
  /// dominating load operation immediately prior to an op with regions.
  SSAReg reloadMemoryReference(OpBuilder &builder, MemRef mr) {
    if (isa<cudaq::quake::RefType>(mr.getType())) {
      auto wireTy = cudaq::quake::WireType::get(builder.getContext());
      return cudaq::quake::UnwrapOp::create(builder, mr.getLoc(), wireTy, mr);
    }
    return cudaq::cc::LoadOp::create(builder, mr.getLoc(), mr);
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
    auto &blockMap = liveInMap[block];
    auto it = blockMap.find(mr);
    if (it != blockMap.end())
      return it->second;
    return addLiveInToBlock(block, mr);
  }

  void maybeAddBalancedLiveInToBlock(Block *block, MemRef mr) {
    if (liveOutSet.count(mr)) {
      if (block->getPredecessors().empty()) {
        if (liveInMap[block].count(mr))
          if (isa<BlockArgument>(liveInMap[block][mr]))
            return;
        auto ty = dereferencedType(mr.getType());
        SSAReg newReg = block->addArgument(ty, mr.getLoc());
        liveInMap[block][mr] = newReg;
        return;
      }
      maybeAddLiveInToBlock(block, mr);
    }
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
  unsigned getLiveInToBlock(SmallVectorImpl<MemRef> &result, Block *block) {
    assert(block && liveInMap.count(block));
    unsigned offset = std::numeric_limits<unsigned>::max();
    for (auto [mr, val] : liveInMap[block])
      if (auto arg = dyn_cast<BlockArgument>(val);
          arg && arg.getOwner() == block) {
        auto argNum = arg.getArgNumber();
        result[argNum] = mr;
        if (argNum < offset)
          offset = argNum;
      }

    LLVM_DEBUG(
        if (std::distance(result.begin(), result.end()) > offset)
            std::for_each(result.begin() + offset, result.end(), [](MemRef mr) {
              if (!mr)
                llvm::dbgs() << "block argument value must be present\n";
            }));
    return offset;
  }

  // After the initial per-region scan, each region of a multi-region parent
  // has independently assigned its own block arguments to whatever memrefs it
  // locally references. A region-branch terminator forwards a single operand
  // list across regions that must agree on slot order. Just select a region as
  // the canonical and physically permute every other region's block arguments
  // to match it, remapping their uses accordingly.
  void canonicalizeArgumentOrder(Operation *parent) {
    if (entryCFG.empty())
      return;
    const bool noRegionArguments = neverTakesRegionArguments(parent);
    if (noRegionArguments)
      return;
    const bool onlyLinearTypes = onlyTakesLinearTypeArguments(parent);

    // Build a canonical order for the whole of the op's regions.
    SmallVector<MemRef> canonicalOrder;
    DenseSet<MemRef> seen;
    auto addFromBlock = [&](Block *block) {
      SmallVector<MemRef> order(block->getNumArguments(), MemRef{});
      getLiveInToBlock(order, block);
      for (auto mr : order)
        if (mr && seen.insert(mr).second)
          canonicalOrder.push_back(mr);
    };
    addFromBlock(entryCFG.front());
    for (auto &region : parent->getRegions()) {
      if (region.empty())
        continue;
      Block *block = &region.front();
      if (block != entryCFG.front())
        addFromBlock(block);
    }
    for (auto mr : liveOutSet)
      if (seen.insert(mr).second)
        canonicalOrder.push_back(mr);
    if (onlyLinearTypes)
      llvm::erase_if(canonicalOrder, [](MemRef mr) {
        return !cudaq::quake::isQuantumReferenceType(mr.getType());
      });
    if (canonicalOrder.empty())
      return;

    LLVM_DEBUG(llvm::dbgs() << "canonicalizeArgumentOrder: canonicalOrder=[";
               for (auto mr : canonicalOrder) mr.dump(); llvm::dbgs() << "]\n");
    for (auto &region : parent->getRegions()) {
      if (!region.empty())
        unifyBlockArguments(&region.front(), canonicalOrder);
    }
    reorderLiveOut(canonicalOrder, onlyLinearTypes);
  }

  // The live-out set fixes the parent's appended result order and the operand
  // order of every exit terminator, both of which must agree with the block
  // arguments just canonicalized. Permute it to match canonicalOrder. memrefs
  // with no block argument (classical ones, under LinearTypeArgs) keep their
  // relative order and follow.
  void reorderLiveOut(ArrayRef<MemRef> canonicalOrder, bool onlyLinearTypes) {
    if (liveOutSet.empty())
      return;
    DenseMap<MemRef, unsigned> canonicalPos;
    for (auto [i, mr] : llvm::enumerate(canonicalOrder))
      canonicalPos[mr] = i;
    SmallVector<MemRef> ordered(liveOutSet.begin(), liveOutSet.end());
    llvm::stable_sort(ordered, [&](MemRef a, MemRef b) {
      auto ai = canonicalPos.find(a);
      auto bi = canonicalPos.find(b);
      bool aKnown = ai != canonicalPos.end();
      bool bKnown = bi != canonicalPos.end();
      if (aKnown != bKnown)
        return aKnown;
      return aKnown && ai->second < bi->second;
    });
    liveOutSet.clear();
    for (auto mr : ordered)
      liveOutSet.insert(mr);

    // Rebuild liveInArgs, which was built from liveOutSet's previous order.
    liveInArgs.clear();
    for (auto liveOut : liveOutSet) {
      assert(promotedDefs.count(liveOut));
      if (onlyLinearTypes && !isLinearType(promotedDefs[liveOut]))
        continue;
      liveInArgs.push_back(promotedDefs[liveOut]);
    }
  }

  // Ensure block arguments are in a canonicalOrder.
  void unifyBlockArguments(Block *block, ArrayRef<MemRef> canonicalOrder) {
    unsigned prefix = originalArgCount.lookup(block);
    SmallVector<MemRef> order(block->getNumArguments(), MemRef{});
    getLiveInToBlock(order, block);

    if (order.size() - prefix == canonicalOrder.size()) {
      bool identity = true;
      for (unsigned i = 0; i < canonicalOrder.size(); ++i)
        if (order[prefix + i] != canonicalOrder[i]) {
          identity = false;
          break;
        }
      if (identity)
        return;
    }

    DenseMap<MemRef, unsigned> canonicalPos;
    for (auto [i, mr] : llvm::enumerate(canonicalOrder))
      canonicalPos[mr] = i;

    unsigned oldCount = block->getNumArguments();
    SmallVector<BlockArgument> newArgs;
    for (auto mr : canonicalOrder)
      newArgs.push_back(
          block->addArgument(dereferencedType(mr.getType()), mr.getLoc()));

    // Map each of block's current suffix arguments (added before this pass
    // touched anything vs. this pass's own earlier additions -- either way,
    // about to be erased) to the new argument replacing it.
    DenseMap<Value, Value> oldToNew;
    for (unsigned i = prefix; i < oldCount; ++i) {
      MemRef mr = order[i];
      if (!mr)
        continue;
      auto it = canonicalPos.find(mr);
      assert(it != canonicalPos.end() &&
             "canonicalOrder must be a superset of every region's own order");
      oldToNew[block->getArgument(i)] = newArgs[it->second];
    }

    // RAUW every real IR use of an old suffix argument.
    for (auto [oldVal, newVal] : oldToNew)
      oldVal.replaceAllUsesWith(newVal);

    // A different memref's binding may itself alias one of these old arguments,
    // because its genuinely correct value happens to equal one.
    if (rMap.count(block))
      for (auto &[mr, val] : rMap[block])
        if (auto it = oldToNew.find(val); it != oldToNew.end())
          val = it->second;
    if (liveInMap.count(block))
      for (auto &[mr, val] : liveInMap[block])
        if (auto it = oldToNew.find(val); it != oldToNew.end())
          val = it->second;

    block->eraseArguments(prefix, oldCount - prefix);

    // Establish liveInMap/rMap for every canonical memref at its new
    // position, including ones block never referenced before (the
    // pass-through case: its own value, for the duration of this block, is
    // simply whatever comes in).
    for (unsigned i = 0; i < canonicalOrder.size(); ++i) {
      MemRef mr = canonicalOrder[i];
      liveInMap[block][mr] = newArgs[i];
      if (!rMap.count(block) || !rMap[block].count(mr))
        rMap[block][mr] = newArgs[i];
    }
  }

  std::optional<SSAReg> hasLiveInToBlock(Block *block, MemRef mr) {
    assert(block && mr);
    auto iter = liveInMap.find(block);
    if (iter == liveInMap.end())
      return {};
    for (auto [mrk, val] : iter->second)
      if (mrk == mr)
        return {val};
    return {};
  }

  // Promote the memory dereference \p memuse to immediately before the parent
  // operation. This allows uses within the regions of the parent to use the
  // new dominating dereference. These will be converted to live-in arguments
  // if the op takes region arguments.
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

  /// Convert the promoted loads for live-out values to block arguments and
  /// insert modified blocks and their predecessors on the worklist. If \p
  /// parent takes region arguments, also pass those loads as parent operands.
  /// Otherwise its region entries capture the promoted loads directly while
  /// internal CFG blocks continue to receive block arguments.
  void updatePromotedDefs(Operation *parent, std::deque<Block *> &worklist) {
    if (liveOutSet.empty())
      return;
    const bool noRegionArguments = neverTakesRegionArguments(parent);
    const bool onlyLinearTypes = onlyTakesLinearTypeArguments(parent);
    assert(liveInArgs.empty() && "parent's live-in args should not be set");
    if (!noRegionArguments)
      for (auto liveOut : liveOutSet) {
        assert(promotedDefs.count(liveOut));
        if (onlyLinearTypes && !isLinearType(promotedDefs[liveOut]))
          continue;
        liveInArgs.push_back(promotedDefs[liveOut]);
      }
    // Phase 1: In one pass, collect unique blocks and snapshot (user, block)
    // pairs per def. Snapshotting here avoids re-traversing promotedDefs and
    // re-calling findParentBlock in phase 3.
    using UserBlocksType = SmallVector<std::pair<Operation *, Block *>>;
    using DefInfo = std::tuple<MemRef, SSAReg, UserBlocksType>;
    SmallVector<DefInfo> defInfos;
    SmallPtrSet<Block *, 4> blockSet;
    for (auto [mr, val] : promotedDefs) {
      if (onlyLinearTypes && !isLinearType(val))
        continue;
      if (!liveOutSet.count(mr))
        continue;
      auto &info = defInfos.emplace_back(mr, val, UserBlocksType{});
      for (auto *user : val.getUsers()) {
        auto *block = findParentBlock(parent, user->getBlock());
        blockSet.insert(block);
        std::get<UserBlocksType>(info).emplace_back(user, block);
      }
    }
    // Phase 2: For each unique block, add live-in block args and queue preds.
    // Add all promoted defs in order so that the argument list doesn't get
    // permuted. Use the unsafe call here because liveInMap should already have
    // a binding for memref to the promoted load value. That binding will be
    // overwritten.
    for (auto *block : blockSet) {
      // NoRegionArguments applies to physical region entry blocks, not to
      // internal CFG blocks. Entry blocks capture the dominating promoted
      // values directly; internal blocks still need arguments to thread defs
      // from their predecessors.
      if (noRegionArguments && block->isEntryBlock())
        continue;
      for (auto memref : liveOutSet) {
        if (onlyLinearTypes && !isLinearType(promotedDefs[memref]))
          continue;
        // block landed in blockSet because some memref's promoted value has
        // a user here. If this memref was already threaded into block as a
        // genuine live-in, it must be left alone: the "unsafe" call below
        // always appends a fresh argument, so calling it again here would leave
        // that first, already-live argument orphaned (unused, uncounted)
        // instead of overwriting it.
        if (auto it = liveInMap[block].find(memref);
            it != liveInMap[block].end() && it->second != promotedDefs[memref])
          continue;
        unsafeAddLiveInToBlock(block, memref);
      }
      worklist.push_back(block);
      appendToWorklist(worklist, getPredecessors(block));
    }
    // Phase 3: Update bindings and replace uses with the new block args.
    for (auto &info : defInfos) {
      for (auto [user, block] : std::get<UserBlocksType>(info)) {
        Value newReg = liveInMap[block][std::get<0>(info)];
        if (!hasBinding(block, std::get<0>(info)) ||
            getBinding(block, std::get<0>(info)) == std::get<1>(info))
          addBinding(block, std::get<0>(info), newReg);
        user->replaceUsesOfWith(std::get<1>(info), newReg);
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

  void incBindingsAdded(Operation *op, Block *target) {
    if (isPreciseEdgeTerminator(op))
      ++preciseEdgeBindingsAdded[op][target];
    else
      ++bindingsAdded[op];
  }

  unsigned numBindingsAdded(Operation *op, Block *target) {
    if (isPreciseEdgeTerminator(op))
      return preciseEdgeBindingsAdded[op][target];
    return bindingsAdded[op];
  }

private:
  // Delete all ctors that should never be used.
  RegionDataFlow() = delete;
  RegionDataFlow(const RegionDataFlow &) = delete;
  RegionDataFlow(RegionDataFlow &&) = delete;

  /// Does this terminator naturally have multiple targets and also support
  /// precise CFG edge semantics?
  bool isPreciseEdgeTerminator(Operation *op) {
    return isa<cf::CondBranchOp>(op);
  }

  /// A map for each block to its bindings from a memory reference to a
  /// virtual register value. Insertion-order-preserving so that ops emitted
  /// while iterating a block's bindings come out in a deterministic order
  /// instead of DenseMap's pointer-hash
  /// bucket order, which varies run to run.
  DenseMap<Block *, llvm::MapVector<MemRef, SSAReg>> rMap;
  /// For a CFG, maintain a distinct map for each block of the definitions
  /// that are live-in to each block.
  DenseMap<Block *, llvm::MapVector<MemRef, SSAReg>> liveInMap;
  /// Map from a memory reference to its promoted value.
  DenseMap<MemRef, SSAReg> promotedDefs;
  /// Map for each imprecise terminator to track the number of bindings added.
  DenseMap<Operation *, unsigned> bindingsAdded;
  /// Map for each precise terminator to track the number of bindings added.
  DenseMap<Operation *, DenseMap<Block *, unsigned>> preciseEdgeBindingsAdded;

  /// The list of live-in arguments to the parent. The parent cannot be a
  /// function.
  SmallVector<SSAReg> liveInArgs;
  /// This is the set of all definitions that are live-out of the parent's
  /// regions and thus must be returned as results. The parent cannot be a
  /// function.
  SetVector<MemRef> liveOutSet;

  /// Each block's argument count as it stood before this pass added
  /// anything -- see the RegionDataFlow constructor.
  DenseMap<Block *, unsigned> originalArgCount;

  SmallVector<Block *> entryCFG;
  SmallVector<Block *> exitCFG;
  DenseMap<Block *, SmallVector<Block *>> backwardCFG;
  DenseMap<Block *, SmallVector<Block *>> cachedPredCFG;
};
} // namespace

namespace {
/// The reset operation is a bit of an oddball and doesn't support the
/// QuakeOperator interface. Handle it special for now.
class ResetOpPattern : public OpRewritePattern<cudaq::quake::ResetOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ResetOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto wireTy = cudaq::quake::WireType::get(rewriter.getContext());
    auto opnd = op.getTargets();
    assert(opnd.getType() == cudaq::quake::RefType::get(rewriter.getContext()));
    Value target = cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, opnd);
    auto newOp =
        cudaq::quake::ResetOp::create(rewriter, loc, TypeRange{wireTy}, target);
    rewriter.replaceOpWithNewOp<cudaq::quake::WrapOp>(op, newOp.getResult(0),
                                                      opnd);
    return success();
  }
};

class DeallocOpPattern : public OpRewritePattern<cudaq::quake::DeallocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::DeallocOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto wireTy = cudaq::quake::WireType::get(rewriter.getContext());
    auto opnd = op.getReference();
    assert(isa<cudaq::quake::RefType>(opnd.getType()));
    Value target = cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, opnd);
    rewriter.replaceOpWithNewOp<cudaq::quake::SinkOp>(op, target);
    return success();
  }
};

/// The evince operation is also an oddball like the reset operation.
class EvinceOpPattern : public OpRewritePattern<cudaq::quake::EvinceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::EvinceOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto wireTy = cudaq::quake::WireType::get(rewriter.getContext());
    auto qrefTy = cudaq::quake::RefType::get(rewriter.getContext());

    SmallVector<Value> refArgs;
    SmallVector<Value> newArgs;
    for (Value arg : op.getArgs()) {
      if (arg.getType() == qrefTy) {
        Value wire = cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, arg);
        newArgs.push_back(wire);
        refArgs.push_back(arg);
      } else {
        newArgs.push_back(arg);
      }
    }

    auto newOp = cudaq::quake::EvinceOp::create(rewriter, loc, newArgs);
    for (auto namedAttr : op->getAttrs())
      newOp->setAttr(namedAttr.getName(), namedAttr.getValue());

    for (auto [ref, wireResult] : llvm::zip(refArgs, newOp.getOuts()))
      cudaq::quake::WrapOp::create(rewriter, loc, wireResult, ref);

    rewriter.eraseOp(op);
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
    auto wireTy = cudaq::quake::WireType::get(rewriter.getContext());
    auto qrefTy = cudaq::quake::RefType::get(rewriter.getContext());
    // Scan the control and target positions. Any that were not Wires will be
    // promoted to Wires via an unwrap operation. These unwrap ops become the
    // arguments to the quantum value form of the new quantum operation.
    if constexpr (!cudaq::quake::isMeasure<OP>) {
      for (auto opnd : op.getControls()) {
        auto opndTy = opnd.getType();
        if (opndTy == qrefTy) {
          auto unwrap =
              cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, opnd);
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
        auto unwrap =
            cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, opnd);
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
          cudaq::quake::WrapOp::create(rewriter, loc, newOp.getResult(offset),
                                       i.value());
        } else if (opndTy == wireTy) {
          op.getResult(count++).replaceAllUsesWith(newOp.getResult(offset));
        }
      }
      rewriter.eraseOp(op);
    };

    if constexpr (cudaq::quake::isMeasure<OP>) {
      // The result type of the bits is the same. Add the wire types.
      SmallVector<Type> newTy = {op.getMeasOut().getType()};
      SmallVector<Type> wireTys(unwrapTargs.size(), wireTy);
      newTy.append(wireTys.begin(), wireTys.end());
      auto newOp = OP::create(rewriter, loc, newTy, unwrapTargs,
                              op.getRegisterNameAttr());
      SmallVector<Value> wireOperands = op.getTargets();
      op.getResult(0).replaceAllUsesWith(newOp.getResult(0));
      threadWires(wireOperands, newOp, 1);
    } else {
      // Scan the control and target positions. Any that were not wires
      // originally are now placed in the result vector. Those new results are
      // propagated to wrap operations.
      auto numberOfWires = wireCount(unwrapCtrls, unwrapTargs);
      SmallVector<Type> wireTys{numberOfWires, wireTy};
      auto newOp = OP::create(rewriter, loc, wireTys, op.getIsAdjAttr(),
                              op.getParameters(), unwrapCtrls, unwrapTargs,
                              op.getNegatedQubitControlsAttr());
      auto wireOperands =
          filteredByType(qrefTy, op.getControls(), op.getTargets());
      threadWires(wireOperands, newOp, 0);
    }
    return success();
  }

  static SmallVector<Value> filteredByType(Type qrefTy, ValueRange ctls,
                                           ValueRange trgs) {
    SmallVector<Value> result;
    for (Value v : ctls)
      if (v.getType() == qrefTy)
        result.push_back(v);
    for (Value v : trgs)
      if (v.getType() == qrefTy)
        result.push_back(v);
    return result;
  }

  static std::size_t wireCount(ArrayRef<Value> ctls, ArrayRef<Value> trgs) {
    std::size_t result = 0;
    for (Value v : ctls)
      if (cudaq::quake::isQuantumValueType(v.getType()))
        result++;
    for (Value v : trgs)
      if (cudaq::quake::isQuantumValueType(v.getType()))
        result++;
    return result;
  }
};

#define WRAPPER(OpClass) Wrapper<cudaq::quake::OpClass>
#define WRAPPER_QUANTUM_OPS QUANTUM_OPS(WRAPPER)
#define RAW(OpClass) cudaq::quake::OpClass
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

    // 0) Check that the IR doesn't have high-level control flow present.
    if (failed(preconditionChecks()))
      return;

    // 1) Rewrite the quantum operations into the intermediate QLS form.
    if (failed(convertToQLS()))
      return;

    // 2) Convert load/store memory ops to value form.
    // Merge references naming the same qubit before anything keys a binding
    // on one of them. This is what makes the pass independent of whether CSE
    // ran ahead of it.
    if (quantumValues)
      dedupQuantumRefs(func);

    MemoryAnalysis memAnalysis(func);
    // Decide up front which qubits may be promoted. Everything the analysis
    // cannot prove is reached through exactly one reference is left in memory
    // form.
    QuantumRefAnalysis refAnalysis(func);
    numBlacklistedRefs += refAnalysis.numBlacklisted();
    numPromotedLocations += refAnalysis.numLocations();
    SmallPtrSet<Operation *, 4> cleanUps;
    std::optional<DominanceInfo> domOpt;
    processOpWithRegions(func, memAnalysis, refAnalysis, cleanUps, domOpt);

    // 3) Cleanup the dead ops. Make sure to delay erasing wrap ops since they
    // may still have uses.
    SmallVector<cudaq::quake::WrapOp> wrapOps;
    for (auto *op : cleanUps) {
      if (auto wrap = dyn_cast<cudaq::quake::WrapOp>(op)) {
        wrapOps.push_back(wrap);
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "erasing: "; op->dump(); llvm::dbgs() << '\n');
      op->dropAllUses();
      op->erase();
    }
    for (auto wrap : wrapOps) {
      // In LLVM 22, the typed accessors (getRefValue/getWireValue) perform
      // llvm::cast<TypedValue<T>> which crashes on null operands. After
      // erasing other ops above (with dropAllUses), WrapOp operands may be
      // null. Use raw getOperand() to safely check for null.
      Value ref = wrap->getOperand(1);  // ref_value is operand 1
      Value wire = wrap->getOperand(0); // wire_value is operand 0
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
                        const QuantumRefAnalysis &refAnalysis,
                        SmallPtrSetImpl<Operation *> &cleanUps,
                        std::optional<DominanceInfo> &domOpt) {
    for (auto &region : parent->getRegions())
      for (auto &block : region)
        for (auto &op : block)
          if (op.getNumRegions())
            processOpWithRegions(&op, memAnalysis, refAnalysis, cleanUps,
                                 domOpt);
  }

  /// Process the operation \p parent, which must contain regions, and derive
  /// its use-def informations as an independent subgraph. Operations with
  /// regions are processed in a post-order traversal of the function. To
  /// produce a (semi-)pruned SSA graph, the Region's blocks are walked from
  /// exits to entries to produce liveness information from predecessor to
  /// successor blocks. (It is not possible to construct a \em fully pruned SSA
  /// IR in the MLIR design of Ops with Regions as both exits and backedges must
  /// have the exact same signatures regardless of liveness.)
  ///
  /// Returns the operation that should be treated as \p parent's identity
  /// from here on: \p parent itself, unless it had live-outs that required
  /// rebuilding it with extra results, in which case the newly built
  /// replacement is returned instead.
  Operation *processOpWithRegions(Operation *parent,
                                  const MemoryAnalysis &memAnalysis,
                                  const QuantumRefAnalysis &refAnalysis,
                                  SmallPtrSetImpl<Operation *> &cleanUps,
                                  std::optional<DominanceInfo> &domOpt) {
    ++numProcessOpWithRegionsCalls;
    auto *ctx = &getContext();
    auto wireTy = cudaq::quake::WireType::get(ctx);
    auto qrefTy = cudaq::quake::RefType::get(ctx);

    if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(parent)) {
      // Special case: add an else region if it is absent from parent.
      auto &elseRegion = ifOp.getElseRegion();
      if (elseRegion.empty()) {
        auto block = new Block;
        elseRegion.push_back(block);
        OpBuilder builder(ctx);
        builder.setInsertionPointToEnd(block);
        cudaq::cc::ContinueOp::create(builder, ifOp.getLoc());
      }
    }

    // Precompute, once, every memref that's the target of a cc.store
    // anywhere within `parent`'s regions. `handleUse` (below) needs this to
    // decide whether an external-scope load requires cross-region live-in
    // threading.
    DenseSet<Value> writtenWithinParent;
    parent->walk([&](cudaq::cc::StoreOp store) {
      writtenWithinParent.insert(store.getPtrvalue());
    });

    // 1. If any operations held by the blocks of \p parent contain regions,
    // recursively process those operations. This establishes the value
    // semantics interface for these macro ops.
    handleSubRegions(parent, memAnalysis, refAnalysis, cleanUps, domOpt);

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
            if (arg.getType() == qrefTy && !refAnalysis.isBlacklisted(arg)) {
              OpBuilder builder(ctx);
              builder.setInsertionPointToStart(block);
              Value v = cudaq::quake::UnwrapOp::create(builder, arg.getLoc(),
                                                       wireTy, arg);
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
            // A blacklisted result names a qubit that is also reached through
            // memory, so it must stay in reference form: leave the op and its
            // unwrap/wrap pairs exactly as they are.
            if (llvm::any_of(op->getResults(), [&](Value r) {
                  return r.getType() == qrefTy && refAnalysis.isBlacklisted(r);
                }))
              continue;
            // If this op defines a quantum reference, record it in the maps.
            if (auto alloc = dyn_cast<cudaq::quake::AllocaOp>(op);
                alloc && memAnalysis.isMember(alloc)) {
              // If it is a known non-escaping alloca, then replace it with a
              // null wire and record it for removal.
              if (!dataFlow.hasBinding(block, alloc)) {
                OpBuilder builder(alloc);
                Value v = cudaq::quake::NullWireOp::create(
                    builder, alloc.getLoc(), wireTy);
                cleanUps.insert(alloc);
                dataFlow.addBinding(block, alloc, v);
              }
            } else if (auto alloc = dyn_cast<cudaq::quake::AllocaOp>(op);
                       alloc && alloc.hasInitializedState()) {
              // If this is an quake.alloca followed by a quake.init_state,
              // just skip this op. It has to remain in reference form and
              // there can't be any other ops between this pairing.
            } else {
              OpBuilder builder(ctx);
              builder.setInsertionPoint(op);
              // Track (memref, freshRef) pairs so the wires captured by op's
              // ref-typed operands can be reclaimed with an unwrap placed
              // after op.
              SmallVector<std::pair<Value, Value>> toReclaim;
              for (auto v : op->getOperands())
                if (v.getType() == qrefTy)
                  if (auto vBinding = dataFlow.lookupBinding(block, v)) {
                    // v may be an alloca-promoted ref that is about to be
                    // erased, so op cannot keep using it directly. Bind the
                    // current wire to a fresh reference for op to consume
                    // (see quake.wrap_new) and reclaim the wire with an
                    // unwrap placed after op.
                    auto newRef = cudaq::quake::WrapNewOp::create(
                        builder, op->getLoc(), qrefTy, vBinding);
                    op->replaceUsesOfWith(v, newRef);
                    toReclaim.emplace_back(v, newRef);
                  }
              builder.setInsertionPointAfter(op);
              for (auto [v, newRef] : toReclaim) {
                Value newWire = cudaq::quake::UnwrapOp::create(
                    builder, op->getLoc(), wireTy, newRef);
                dataFlow.addBinding(block, v, newWire);
                // This unwrap's own ref operand is newRef, not v: bind it
                // too so the walk loop's own re-visit of this synthetic
                // unwrap (below) resolves to a known def instead of
                // reporting a spurious "use before def".
                dataFlow.addBinding(block, newRef, newWire);
              }
              for (auto r : op->getResults())
                if (r.getType() == qrefTy) {
                  Value v = cudaq::quake::UnwrapOp::create(
                      builder, op->getLoc(), wireTy, r);
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
                Value v = cudaq::cc::UndefOp::create(builder, alloc.getLoc(),
                                                     alloc.getElementType());
                cleanUps.insert(alloc);
                dataFlow.addBinding(block, alloc, v);
              }
              continue;
            }

          // If this is a new value being created, add it to the map of values
          // for this block so it can be tracked and forwarded.
          if (auto nullWire = dyn_cast<cudaq::quake::NullWireOp>(op)) {
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
              operRef.emitError(DEBUG_TYPE ": use before def in function");
              signalPassFailure();
              return;
            }

            // Parent is not a function.
            if (!isDescendantOf(parent, memuse)) {
              // `block` is using a value from another scope.
              //
              // Normal path: memuse is live-in to `parent` from outside. If
              // `parent` doesn't support real region arguments, or `memuse` is
              // never written anywhere inside `parent`, fall back to a single
              // promoted value reused everywhere. Otherwise thread it as a
              // genuine live-in block argument instead, with no fixed value,
              // and let the live-in/worklist machinery (steps 3-4) resolve the
              // correct per-block value, including revisiting sibling regions
              // when a later-discovered live-in requires it.
              if (neverTakesRegionArguments(parent) ||
                  (onlyTakesLinearTypeArguments(parent) &&
                   !cudaq::quake::isQuantumReferenceType(memuse.getType())) ||
                  !writtenWithinParent.contains(memuse)) {
                auto newUseopVal = dataFlow.createPromotedValue(parent, memuse);
                dataFlow.addBinding(block, memuse, newUseopVal);
                dataFlow.addLiveInToBlock(block, memuse, newUseopVal);
                useop.replaceAllUsesWith(newUseopVal);
              } else {
                auto newDef = dataFlow.addLiveInToBlock(block, memuse);
                dataFlow.addBinding(block, memuse, newDef);
                useop.replaceAllUsesWith(newDef);
              }
              cleanUps.insert(useop);
              return;
            }

            // The def is not in the map AND this is not an entry block.
            auto newDef = dataFlow.addLiveInToBlock(block, memuse);
            dataFlow.addBinding(block, memuse, newDef);
            useop.replaceAllUsesWith(newDef);
            cleanUps.insert(useop);
          };
          if (auto unwrap = dyn_cast<cudaq::quake::UnwrapOp>(op)) {
            // A load from a blacklisted reference stays a real load.
            if (quantumValues &&
                !refAnalysis.isBlacklisted(unwrap.getRefValue()))
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
          if (auto wrap = dyn_cast<cudaq::quake::WrapOp>(op)) {
            // A store to a blacklisted reference stays a real store.
            if (quantumValues && !refAnalysis.isBlacklisted(wrap.getRefValue()))
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
          // use chain and leave a wrap dominating op. Since v may be an
          // alloca-promoted ref that is about to be erased, op cannot keep
          // using it directly: bind the current wire to a fresh reference
          // for op to consume (see quake.wrap_new) and reclaim the wire
          // with an unwrap placed after op.
          {
            SmallVector<std::pair<Value, Value>> toReclaim;
            OpBuilder builder(op);
            for (auto v : op->getOperands())
              if (v.getType() == qrefTy)
                if (auto vBinding = dataFlow.lookupBinding(block, v)) {
                  auto newRef = cudaq::quake::WrapNewOp::create(
                      builder, op->getLoc(), qrefTy, vBinding);
                  op->replaceUsesOfWith(v, newRef);
                  toReclaim.emplace_back(v, newRef);
                }
            if (!toReclaim.empty()) {
              builder.setInsertionPointAfter(op);
              for (auto [v, newRef] : toReclaim) {
                Value newWire = cudaq::quake::UnwrapOp::create(
                    builder, op->getLoc(), wireTy, newRef);
                dataFlow.addBinding(block, v, newWire);
                // This unwrap's own ref operand is newRef, not v: bind it
                // too so the walk loop's own re-visit of this synthetic
                // unwrap (below) resolves to a known def instead of
                // reporting a spurious "use before def".
                dataFlow.addBinding(block, newRef, newWire);
              }
            }
          }

        } // end loop over ops
      } // end loop over blocks
    } // end loop over regions

    LLVM_DEBUG(llvm::dbgs() << "After threading intra-block:\n"
                            << *parent << "\n\n");

    std::deque<Block *> worklist;
    appendToWorklist(worklist, dataFlow.getExitBlocks());

    // 3. If there are defs that are live-out for parent and parent takes region
    // arguments, construct a list of live-in region arguments to add to the new
    // parent and replace uses of promoted defs with block arguments.
    dataFlow.updatePromotedDefs(parent, worklist);

    // 3.5. Steps 2 and 3 each independently assigned block arguments to
    // whatever regions needed them (region-local uses in step 2; the
    // liveOutSet-ordered sweep across every block that references a
    // promoted def in step 3) -- with no coordination between regions. But a
    // region-branch terminator forwards a single operand list across regions
    // that must agree on slot order. Bring every region's argument order in
    // line with the entry region's order now that all such arguments have been
    // created.
    dataFlow.canonicalizeArgumentOrder(parent);

    LLVM_DEBUG({
      llvm::dbgs() << "After fixing up promoted loads:\n"
                   << *parent << "\nPromotions:\n";
      for (auto v : dataFlow.getPromotedDefValues())
        v.dump();
      llvm::dbgs() << '\n';
    });

    // 4. Update the block arguments and terminators to thread the values
    // between the blocks in the CFG. If there are defs that are live-out for
    // parent, then they need to be added to each terminator. Update each pred's
    // terminator to pass all the live-in values to a successor.
    // To maintain SSI properly and form proper sigma nodes, values of linear
    // type must propagate to each successor block.
    auto liveOutParent = dataFlow.getLiveOutOfParent();

    auto addTerminatorArgument = [&](Operation *term, Block *target, Value val,
                                     Value liveOut) {
      if (auto branch = dyn_cast<BranchOpInterface>(term)) {
        unsigned numSuccs = branch->getNumSuccessors();
        // Forward val to the target successor. For SSI (linear) types also
        // form a sigma node: add a block argument and WrapOp to non-target
        // successors that lack a lazy mechanism to receive the wire. For SSA
        // types non-target successors are skipped (handled by the back-edge
        // or outgoing branch processing via maybeAddLiveInToBlock).
        const bool isLinear = cudaq::quake::isLinearType(val.getType());
        bool changes = false;
        for (unsigned i = 0; i < numSuccs; ++i) {
          Block *succ = branch->getSuccessor(i);
          if (target && succ == target) {
            branch.getSuccessorOperands(i).append(val);
            changes = true;
            continue;
          }
          if (!isLinear)
            continue;
          // Non-target SSI successor: insert a block argument and WrapOp only
          // when no lazy path will create them:
          //   - isExitBlock: the return terminator never processes liveOut, so
          //     maybeAddLiveInToBlock is never called lazily.
          //   - hasBinding: a local def causes updateTerminator to use the
          //     binding value directly, bypassing maybeAddLiveInToBlock and
          //     leaving the incoming wire without a block argument to land in.
          // Otherwise the back-edge or a later outgoing branch lazily adds the
          // block argument, and the target path appends branch operands in the
          // correct block-argument order when that block is the target.
          if (liveOut && !dataFlow.hasLiveInToBlock(succ, liveOut) &&
              (dataFlow.isExitBlock(succ) ||
               dataFlow.hasBinding(succ, liveOut))) {
            if (!domOpt) {
              DominanceInfo dom(parent->getParentOfType<func::FuncOp>());
              domOpt = std::move(dom);
            }
            if (domOpt->properlyDominates(liveOut, &succ->front())) {
              worklist.push_back(succ);
              auto sigma = dataFlow.maybeAddLiveInToBlock(succ, liveOut);
              OpBuilder builder(&succ->front());
              auto sigmaWrap = cudaq::quake::WrapOp::create(
                  builder, term->getLoc(), sigma, liveOut);
              // liveOut (the ref) may itself be dead and already destined
              // for cleanUps
              cleanUps.insert(sigmaWrap);
            }
          }
        }
        if (changes)
          worklist.push_back(term->getBlock());
      } else {
        SmallVector<Value> newArgs(term->getOperands());
        newArgs.push_back(val);
        term->setOperands(newArgs);
        worklist.push_back(term->getBlock());
      }
      dataFlow.incBindingsAdded(term, target);
    };

    const bool usePromo = neverTakesRegionArguments(parent);
    const bool onlyLinear = onlyTakesLinearTypeArguments(parent);
    auto updateTerminator = [&](Operation *term, Block *target,
                                ValueRange bindings) {
      Block *block = term->getBlock();
      auto numAddedBindings = dataFlow.numBindingsAdded(term, target);
      if (bindings.size() <= numAddedBindings)
        return;
      for (Value liveOut : bindings.drop_front(numAddedBindings)) {
        if (dataFlow.hasBinding(block, liveOut)) {
          if (!isFunctionBlock(block) && !usePromo && !onlyLinear)
            dataFlow.maybeAddBalancedLiveInToBlock(block, liveOut);
          auto oldVal = dataFlow.getBinding(block, liveOut);
          if (!oldVal) {
            OpBuilder builder(term);
            oldVal = cudaq::quake::UnwrapOp::create(
                builder, term->getLoc(),
                cudaq::quake::WireType::get(builder.getContext()), liveOut);
          }
          addTerminatorArgument(term, target, oldVal, liveOut);
        } else if ((usePromo || (onlyLinear && !isa<cudaq::quake::RefType>(
                                                   liveOut.getType()))) &&
                   dataFlow.isEntryBlock(block)) {
          auto newVal = dataFlow.getPromotedValue(liveOut);
          dataFlow.addBinding(block, liveOut, newVal);
          addTerminatorArgument(term, target, newVal, liveOut);
        } else {
          // Cannot live-in a value from the ether. Give an error.
          if (isFunctionEntryBlock(block) &&
              !dataFlow.hasLiveInToBlock(block, liveOut)) {
            emitError(term->getLoc(), DEBUG_TYPE
                      ": cannot promote a value that would require threading a "
                      "new live-in past the function entry block");
            signalPassFailure();
            return;
          }
          auto newArg = dataFlow.maybeAddLiveInToBlock(block, liveOut);
          addTerminatorArgument(term, target, newArg, liveOut);
        }
      }
    };

    auto updateExitTerminator = [&](Block *block, auto &bindings) {
      return updateTerminator(
          block->getTerminator(), nullptr,
          llvm::make_range(bindings.begin(), bindings.end()));
    };

    SmallPtrSet<Block *, 8> blocksVisited;
    SmallVector<Value> liveInBlock;
    while (!worklist.empty()) {
      ++numWorklistIterations;
      Block *block = worklist.front();
      worklist.pop_front();
      // Check terminator is threading live-out of parent values.
      if (!liveOutParent.empty() && dataFlow.isExitBlock(block))
        updateExitTerminator(block, liveOutParent);

      // Check that preds are threading all live-in values.
      liveInBlock.assign(block->getNumArguments(), Value{});
      auto offset = dataFlow.getLiveInToBlock(liveInBlock, block);
      auto preds = dataFlow.getPredecessors(block);
      if (offset != std::numeric_limits<decltype(offset)>::max()) {
        // Block arguments were added. Update the terminator(s). It's possible
        // that some terminators were already updated from other successor
        // blocks, so we must check each predecessor individually.
        for (auto *pred : preds)
          updateTerminator(pred->getTerminator(), block,
                           llvm::make_range(liveInBlock.begin() + offset,
                                            liveInBlock.end()));
      }

      // We should visit all the predecessor blocks at least once. Add any
      // blocks not yet visited to the worklist.
      blocksVisited.insert(block);
      for (auto *pred : preds)
        if (!blocksVisited.count(pred)) {
          blocksVisited.insert(pred);
          worklist.push_back(pred);
        }
    } // end of worklist loop

    if (dataFlow.hasLiveOutOfParent()) {
      // Get all the new results to append.
      auto allDefs = dataFlow.getLiveOutOfParent();

      // Replace parent with a copy.
      SmallVector<Type> resultTypes(parent->getResultTypes());
      for (auto d : allDefs)
        resultTypes.push_back(dereferencedType(d.getType()));
      IRRewriter builder(ctx);
      builder.setInsertionPoint(parent);
      SmallVector<Value> operands(parent->getOperands());
      operands.insert(operands.end(), dataFlow.getLiveInArgs().begin(),
                      dataFlow.getLiveInArgs().end());
      Operation *np = Operation::create(
          parent->getLoc(), parent->getName(), resultTypes, operands,
          parent->getAttrs(), OpaqueProperties{nullptr},
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
          cudaq::quake::WrapOp::create(builder, np->getLoc(), np->getResult(i),
                                       iter.value());
        else
          cudaq::cc::StoreOp::create(builder, np->getLoc(), np->getResult(i),
                                     iter.value());
      }
      cleanUps.insert(parent);
      parent = np;
    }

    LLVM_DEBUG(llvm::dbgs() << "After threading inter-block:\n"
                            << *parent << "\n\n");
    return parent;
  }

  LogicalResult preconditionChecks() {
    if (getOperation()
            .walk([](Operation *op) {
              if (isa<cudaq::cc::CreateLambdaOp, cudaq::cc::UnwindBreakOp,
                      cudaq::cc::UnwindContinueOp, cudaq::cc::UnwindReturnOp>(
                      op))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted())
      return failure();
    return success();
  }

  // Convert the function to "quantum load/store" (QLS) format.
  LogicalResult convertToQLS() {
    if (!quantumValues)
      return success();
    auto func = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<WRAPPER_QUANTUM_OPS, ResetOpPattern, DeallocOpPattern,
                    EvinceOpPattern>(ctx);
    ConversionTarget target(*ctx);
    target
        .addDynamicallyLegalOp<RAW_QUANTUM_OPS, cudaq::quake::ResetOp,
                               cudaq::quake::DeallocOp, cudaq::quake::EvinceOp>(
            [](Operation *op) {
              return !cudaq::quake::hasNonVectorReference(op);
            });
    target.addLegalOp<cudaq::quake::UnwrapOp, cudaq::quake::WrapOp,
                      cudaq::quake::NullWireOp, cudaq::quake::SinkOp>();
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      emitError(func.getLoc(), DEBUG_TYPE ": error converting to QLS form\n");
      signalPassFailure();
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "After converting to QLS:\n" << func << "\n\n");
    return success();
  }
};
} // namespace
