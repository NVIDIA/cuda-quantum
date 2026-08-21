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
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
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

/// Returns true if and only if \p op behaves like a quantum-gate operator:
/// it implicitly dereferences a `!quake.ref` operand and modifies the wire
/// it refers to. All quake ops with the `QuantumGate` trait (which includes
/// `quake.reset` and `quake.exp_pauli`) act this way, as do the quantum
/// measurement ops (`quake.mx`/`my`/`mz`, tagged `QuantumMeasure`) and ops
/// tagged `QuantumSideEffects` that are not themselves quantum operators
/// but macro-expand to (or invoke) one: `quake.apply`,
/// `quake.compute_action`, `quake.apply_noise`, and `quake.call_by_ref`.
/// Every other op only manages references/veqs — e.g. `quake.concat`,
/// `quake.dealloc`, `quake.relax_size`, `cc.instantiate_callable`,
/// `cc.callable_closure`, and (despite appearances) `quake.log_output` — and
/// cannot alias or mutate a qubit we are tracking. `quake.log_output` merely
/// marks a value as observable for later codegen; it does not dereference
/// or modify anything, so it must not trigger the conservative
/// wrap-and-cancel-everything path below.
static bool actsLikeQuantumOperator(Operation *op) {
  return op->hasTrait<cudaq::QuantumGate>() ||
         op->hasTrait<cudaq::QuantumMeasure>() ||
         op->hasTrait<cudaq::QuantumSideEffects>();
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
        // can't tell escaping capture operands apart from ordinary ones —
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

/// Peel a chain of `quake.subveq`/`quake.relax_size` views off \p veq to find
/// the underlying veq it is ultimately a view of (an alloca, init_state
/// result, function/block argument, or any other op result that isn't itself
/// a further view). \p veq itself is returned unchanged if it isn't a view.
static Value resolveVeqRoot(Value veq) {
  while (true) {
    if (auto sub = veq.getDefiningOp<cudaq::quake::SubVeqOp>()) {
      veq = sub.getVeq();
      continue;
    }
    if (auto relax = veq.getDefiningOp<cudaq::quake::RelaxSizeOp>()) {
      veq = relax.getInputVec();
      continue;
    }
    return veq;
  }
}

/// Walk \p ref's provenance backward — through the `quake.extract_ref` that
/// produced it and that op's source-veq chain of `quake.subveq`/
/// `quake.relax_size` views — and compare its ultimate root veq against \p
/// v's own ultimate root (resolved the same way) to determine whether \p ref
/// can be proven independent of \p v's qubit range.
///
/// Both provenance chains must be resolved to their roots and compared —
/// not just \p ref's chain searched for the literal value \p v — because
/// \p ref and \p v may be *siblings* that share a common root without either
/// being derived from the other (e.g. \p ref extracted directly from a veq
/// %q, and \p v a `quake.subveq` of that same %q taken independently): \p
/// v never appears verbatim in \p ref's chain in that case, even though they
/// plainly alias.
///
/// This is intentionally driven backward from a single, already-tracked ref
/// binding rather than forward from \p v across every use in the function:
/// the only refs that matter are ones that are currently live bindings in
/// the block being processed (i.e. already properly scoped to the region
/// being threaded), and a ref's own provenance chain is unambiguous — unlike
/// enumerating every downstream use of \p v, which can spuriously implicate
/// (or, on an unrelated unclassifiable use anywhere in the function, force a
/// blanket bailout for) refs that have nothing to do with the current
/// aliasing site.
///
/// Returns true only when \p ref's provenance is fully resolved and its root
/// is a distinct SSA value from \p v's own resolved root — i.e. \p ref is
/// definitely not derived from \p v. Returns false (can't rule it out, so
/// the caller must treat \p ref as a possible alias) whenever the roots
/// match or \p ref bottoms out at something that isn't itself further
/// resolvable (e.g. \p ref is a quake.wrap_new stand-in — see
/// reclaimAliasedRef — whose provenance can't be traced back at all).
static bool definitelyNotDerivedFrom(Value ref, Value v) {
  // A standalone ref alloca owns an independent qubit: it was never
  // extracted from any veq, so it can't be derived from v regardless of v's
  // identity.
  if (auto *defOp = ref.getDefiningOp())
    if (isa<cudaq::quake::AllocaOp>(defOp))
      return true;
  auto extract = ref.getDefiningOp<cudaq::quake::ExtractRefOp>();
  if (!extract)
    return false;
  return resolveVeqRoot(extract.getVeq()) != resolveVeqRoot(v);
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
        auto *terminator = region.back().getTerminator();
        if (auto terminatorOp =
                dyn_cast<RegionBranchTerminatorOpInterface>(terminator))
          regionOp.getSuccessorRegions(terminatorOp, successors);
        // Every region has exactly one entry and one or more exits.
        for (auto *b : regionExitBlocks)
          for (auto iter : successors) {
            auto *succ = iter.getSuccessor();
            if (succ) {
              auto *s = &succ->front();
              reverseCFG[s].insert(b);
            } else {
              exitBlocks.insert(b);
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

  /// \p ref is an alloca-promoted ref that is about to be erased (its
  /// defining op is in \p cleanUps), but a downstream op not yet visited may
  /// still hold \p ref as a raw operand (e.g. a synthetic UnwrapOp inserted
  /// by convertToQLS). Cancelling \p ref's binding without doing anything
  /// else would leave that later use dangling once the alloca is erased —
  /// give \p ref a fresh, live stand-in and redirect every remaining use to
  /// it before cancelling.
  ///
  /// \p wire is consumed by the mint (its qubit identity is handed off to
  /// the fresh ref), so the fresh ref's own binding must be cancelled too —
  /// not bound back to \p wire — otherwise a later use of the fresh ref
  /// would be optimized straight through to \p wire, giving it a second use
  /// and violating wire linearity. This mirrors the fact that whatever
  /// mutated the aliased qubit did so entirely in ref-space: \p wire no
  /// longer represents the qubit's current state, only the fresh ref does.
  void reclaimAliasedRef(Block *block, OpBuilder &builder, Location loc,
                         MemRef ref, SSAReg wire) {
    auto newRef =
        cudaq::quake::WrapNewOp::create(builder, loc, ref.getType(), wire);
    ref.replaceAllUsesWith(newRef);
    cancelBinding(block, newRef);
    cancelBinding(block, ref);
  }

  /// Cancel the binding for a single \p ref in \p block: wrap its current
  /// wire back into the ref (or, if \p ref is an alloca-promoted ref about to
  /// be erased, give it a fresh live stand-in via reclaimAliasedRef instead)
  /// and cancel the binding. No-op if \p ref has no active binding.
  ///
  /// If \p triggeringOp is non-null and \p ref's current wire is already a
  /// direct operand of \p triggeringOp, this is a no-op: wrapping that wire
  /// here would give it a second use (the wrap and the op), violating wire
  /// linearity — the op itself is about to "consume" that wire.
  void cancelSingleBinding(Block *block, OpBuilder &builder, Location loc,
                           SmallPtrSetImpl<Operation *> &cleanUps, MemRef ref,
                           Operation *triggeringOp) {
    if (!rMap.count(block))
      return;
    auto it = rMap[block].find(ref);
    if (it == rMap[block].end())
      return;
    SSAReg wire = it->second;
    if (!wire)
      return;
    if (triggeringOp && llvm::is_contained(triggeringOp->getOperands(), wire))
      return;

    // If wire's only use is already a WrapOp targeting this same ref, the
    // physical ref state is already up to date -- e.g. this binding was
    // last set by handleDefinition processing a real quake.wrap already
    // present in the IR (such as the one convertToQLS unconditionally
    // inserts after a measurement). Emitting another WrapOp here would
    // consume wire a second time, violating wire linearity, and is
    // unnecessary: just clear the binding so a later use of ref gets a
    // fresh unwrap.
    if (wire.hasOneUse())
      if (auto existingWrap =
              dyn_cast<cudaq::quake::WrapOp>(*wire.getUsers().begin()))
        if (existingWrap.getRefValue() == ref) {
          cancelBinding(block, ref);
          return;
        }

    // Alloca refs in cleanUps are being SSA-promoted; skip the wrap but
    // give ref a live stand-in before cancelling so subsequent uses get a
    // fresh binding instead of a dangling operand.
    if (auto *defOp = ref.getDefiningOp())
      if (cleanUps.count(defOp)) {
        reclaimAliasedRef(block, builder, loc, ref, wire);
        return;
      }

    auto wrapOp = cudaq::quake::WrapOp::create(builder, loc, wire, ref);
    cleanUps.insert(wrapOp);
    cancelBinding(block, ref);
  }

  /// Wrap all active quantum-ref bindings back into their refs and cancel them.
  /// Used when an operation may alias any qubit through a veq whose membership
  /// cannot be precisely determined.
  void wrapAndCancelAllQuantumBindings(Block *block, OpBuilder &builder,
                                       Location loc,
                                       SmallPtrSetImpl<Operation *> &cleanUps,
                                       Operation *triggeringOp = nullptr) {
    if (!rMap.count(block))
      return;
    SmallVector<MemRef> toCancel;
    for (auto &[ref, wire] : rMap[block]) {
      if (!wire)
        continue;
      if (!isa<cudaq::quake::RefType>(ref.getType()))
        continue;
      toCancel.push_back(ref);
    }
    for (MemRef ref : toCancel)
      cancelSingleBinding(block, builder, loc, cleanUps, ref, triggeringOp);
  }

  /// Cancel every active ref binding in \p block whose provenance cannot be
  /// proven independent of \p v (see definitelyNotDerivedFrom). This is
  /// naturally scoped to whatever is currently live/tracked in \p block: a
  /// ref extracted somewhere else in the function that never became a
  /// binding here is untouched regardless of what veq it came from.
  void cancelBindingsAliasing(Value v, Block *block, OpBuilder &builder,
                              Location loc,
                              SmallPtrSetImpl<Operation *> &cleanUps,
                              Operation *triggeringOp) {
    if (!rMap.count(block))
      return;
    SmallVector<MemRef> toCancel;
    for (auto &[ref, wire] : rMap[block]) {
      if (!wire)
        continue;
      if (!isa<cudaq::quake::RefType>(ref.getType()))
        continue;
      if (definitelyNotDerivedFrom(ref, v))
        continue;
      toCancel.push_back(ref);
    }
    for (MemRef ref : toCancel)
      cancelSingleBinding(block, builder, loc, cleanUps, ref, triggeringOp);
  }

  /// For each veq value in \p veqsToCancel, wrap any active ref bindings back
  /// to their refs and cancel them. When the veq is the result of a
  /// quake.concat whose members are known statically, only the individual ref
  /// operands of that concat are cancelled. Otherwise, only the currently
  /// active bindings in \p block whose provenance cannot be proven
  /// independent of the veq are cancelled (see cancelBindingsAliasing).
  ///
  /// \p veqsToCancel is a SetVector: iteration order must be deterministic,
  /// since it drives the order wrap ops are inserted in.
  void cancelBindings(const SetVector<Value> &veqsToCancel, Block *block,
                      SmallPtrSetImpl<Operation *> &cleanUps, Operation *op) {
    for (Value v : veqsToCancel) {
      OpBuilder builder(op);
      Location loc = op->getLoc();
      auto concat = v.getDefiningOp<cudaq::quake::ConcatOp>();
      if (!concat) {
        cancelBindingsAliasing(v, block, builder, loc, cleanUps, op);
        continue;
      }
      bool fallback = false;
      for (Value arg : concat.getTargets()) {
        if (fallback)
          break;
        // A quake.wrap_new result is a stand-in ref minted to let some
        // other op (e.g. this very concat) consume a memref that couldn't
        // be used directly — see reclaimAliasedRef and the toReclaim
        // pattern above. Its binding in rMap is a one-time snapshot that is
        // never updated as the real, underlying memref's binding evolves,
        // so cancelling *it* would silently leave the real memref's (now
        // stale) binding live. There is no way to trace a stand-in back to
        // the memref it stood in for, so fall back to the conservative
        // cancel-everything path.
        if (arg.getDefiningOp<cudaq::quake::WrapNewOp>()) {
          fallback = true;
          continue;
        }
        if (isa<cudaq::quake::RefType>(arg.getType())) {
          SSAReg cur = lookupBinding(block, arg);
          if (cur && !llvm::is_contained(op->getOperands(), cur)) {
            bool argInCleanUps =
                arg.getDefiningOp() && cleanUps.count(arg.getDefiningOp());
            if (argInCleanUps) {
              reclaimAliasedRef(block, builder, loc, arg, cur);
            } else {
              auto wrapOp =
                  cudaq::quake::WrapOp::create(builder, loc, cur, arg);
              cleanUps.insert(wrapOp);
              cancelBinding(block, arg);
            }
          }
        } else if (auto veqTy =
                       dyn_cast<cudaq::quake::VeqType>(arg.getType())) {
          if (!veqTy.hasSpecifiedSize()) {
            fallback = true;
          } else {
            // A specified-size veq member of the concat may itself alias
            // refs extracted from a different view of the same underlying
            // storage (e.g. arg is a quake.subveq of some %q, and some
            // other live ref was extracted directly from %q) -- those
            // bindings must be invalidated too, exactly as for a veq
            // operand outside of a concat (see cancelBindingsAliasing).
            cancelBindingsAliasing(arg, block, builder, loc, cleanUps, op);
          }
        } else {
          fallback = true;
        }
      }
      if (fallback)
        wrapAndCancelAllQuantumBindings(block, builder, loc, cleanUps, op);
    }
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
      for (auto memref : liveOutSet) {
        if (onlyLinearTypes && !isLinearType(promotedDefs[memref]))
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
  /// while iterating a block's bindings (e.g. wrapAndCancelAllQuantumBindings)
  /// come out in a deterministic order instead of DenseMap's pointer-hash
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

/// The log_output operation is also an oddball like the reset operation.
class LogOutputOpPattern : public OpRewritePattern<cudaq::quake::LogOutputOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::LogOutputOp op,
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

    auto newOp = cudaq::quake::LogOutputOp::create(rewriter, loc, newArgs);
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
  using VeqAccessMap = DenseMap<Operation *, SmallVector<Value, 4>>;

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
    MemoryAnalysis memAnalysis(func);
    SmallPtrSet<Operation *, 4> cleanUps;
    std::optional<DominanceInfo> domOpt;
    VeqAccessMap unusedMap;
    processOpWithRegions(func, memAnalysis, cleanUps, domOpt, unusedMap);

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
                        SmallPtrSetImpl<Operation *> &cleanUps,
                        std::optional<DominanceInfo> &domOpt,
                        VeqAccessMap &childMap) {
    for (auto &region : parent->getRegions())
      for (auto &block : region)
        for (auto &op : block)
          if (op.getNumRegions()) {
            Operation *finalOp = processOpWithRegions(
                &op, memAnalysis, cleanUps, domOpt, childMap);
            // processOpWithRegions may replace &op with a new operation (when
            // it has live-outs that need to be appended as new results. &op
            // itself survives physically in the block (only erased later, via
            // cleanUps) but is a dead husk from here on: any childMap summary
            // deposited under the *old* key must be rekeyed to the surviving
            // op, or the caller's own block walk — which will encounter both
            // the new and the (still physically present, soon-to-be-erased) old
            // op as separate entries — would read the summary off the wrong
            // (dead) operation, whose operand list no longer reflects reality.
            if (finalOp != &op) {
              auto it = childMap.find(&op);
              if (it != childMap.end()) {
                // Copy the summary out and erase via `it` *before* inserting
                // under the new key: DenseMap's operator[] may rehash on
                // insertion, which would invalidate `it`.
                auto summary = std::move(it->second);
                childMap.erase(it);
                childMap[finalOp] = std::move(summary);
              }
            }
          }
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
                                  SmallPtrSetImpl<Operation *> &cleanUps,
                                  std::optional<DominanceInfo> &domOpt,
                                  VeqAccessMap &parentMap) {
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

    // 1. If any operations held by the blocks of \p parent contain regions,
    // recursively process those operations. This establishes the value
    // semantics interface for these macro ops.
    // childMap accumulates veq-access summaries from the recursive calls so
    // the block loop below can apply binding cancellations at the right scope.
    VeqAccessMap childMap;
    handleSubRegions(parent, memAnalysis, cleanUps, domOpt, childMap);

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
              Value v = cudaq::quake::UnwrapOp::create(builder, arg.getLoc(),
                                                       wireTy, arg);
              dataFlow.addBinding(block, arg, v);
            }
          }
        }

        // True once a dynamic extract_ref from a veq defined *outside* this
        // block's parent scope has been seen.  After that point, any
        // outer-scope ref first imported as a live-in gets a fresh unwrap
        // rather than being replaced by the live-in block argument (see
        // handleUse below).
        bool aliasForBlock = false;

        // Loop over all operations in the block.
        for (Operation &operRef : *block) {
          Operation *op = &operRef;

          // Veq aliasing: an op that reads from or passes a veq (dynamic
          // extract_ref, mz, subveq, func.call, etc.) may access any qubit
          // inside that veq, potentially modifying qubits we are tracking via
          // individual ref bindings.  Wrap and cancel all affected bindings so
          // that subsequent uses re-load the up-to-date qubit state.
          //
          // This check must run FIRST — before any handler that issues a
          // `continue` — so it fires even for ops like extract_ref that both
          // use a veq operand and produce a !quake.ref result.
          //
          // Ops that act like quantum-gate operators (see
          // actsLikeQuantumOperator) actually dereference a ref and modify
          // the wire it refers to. A dynamic-index quake.extract_ref is the
          // other trigger: it doesn't itself mutate anything, but it returns
          // a ref that may alias any individually-tracked ref in the same
          // veq, so any such stale binding must be invalidated right here —
          // waiting for a later op to dereference the extracted ref would be
          // too late, since that op's own operand is the ref, not the veq,
          // and so it can't be traced back to the veq it may alias. Every
          // other op — quake.concat, quake.dealloc, quake.relax_size,
          // cc.instantiate_callable, cc.callable_closure, etc. — only
          // manages references/veqs and cannot alias or mutate a qubit we
          // are tracking, so this conservative cancellation must not fire
          // for them. The third trigger is any op with regions (cc.loop,
          // cc.if, cc.scope, ...): the aliasing access may be nested inside
          // one of its regions rather than be a direct operand of the op
          // itself, reported back via childMap/parentMap (Source 2 below)
          // by the recursive processOpWithRegions call that already ran
          // over its regions. That summary is only ever consumed here, so
          // an op with regions must always enter this branch to have a
          // chance to consume it, regardless of what its own operands are.
          if (quantumValues && (actsLikeQuantumOperator(op) ||
                                isa<cudaq::quake::ExtractRefOp>(op) ||
                                op->getNumRegions() > 0)) {
            // Collect veqs whose ref-bindings need to be cancelled.
            // Two sources feed into this set:
            //
            //  1. Direct veq operands of this op that represent a
            //     non-conservative access (dynamic-index extract_ref, mz, …).
            //     If the veq's defining op is *outside* the current `parent`
            //     scope we cannot cancel here — record it for the outer scope.
            //
            //  2. Summary entries deposited by the inner processOpWithRegions
            //     call for this op (via externalVeqAccesses).  If any of those
            //     veqs are *also* from outside the current `parent`, propagate
            //     them one level further up; otherwise cancel now.
            SetVector<Value> veqsToCancel;

            // Helper: is `v` defined outside of `parent`'s regions?
            auto isFromOutsideParent = [&](Value v) -> bool {
              if (auto *defOp = v.getDefiningOp())
                return !parent->isAncestor(defOp);
              if (auto ba = dyn_cast<BlockArgument>(v))
                return !parent->isAncestor(ba.getOwner()->getParentOp());
              return false;
            };

            // Source 1: direct veq operands.
            // Two paths:
            //   (a) veq defined inside this scope  → precise per-concat cancel
            //   (b) veq defined outside this scope → conservative: cancel ALL
            //       active ref bindings and set aliasForBlock so that any
            //       outer-scope ref first imported after this point gets a
            //       fresh unwrap instead of the (potentially stale) live-in.
            bool hadOuterVeq = false;
            SmallVector<Value, 2> outerVeqs;
            for (Value v : op->getOperands()) {
              if (!isa<cudaq::quake::VeqType>(v.getType()))
                continue;
              if (auto ext = dyn_cast<cudaq::quake::ExtractRefOp>(op))
                if (ext.hasConstantIndex())
                  continue;
              if (isFromOutsideParent(v)) {
                // Record in parentMap so the outer processOpWithRegions
                // can cancel the binding at the right scope level.
                parentMap[parent].push_back(v);
                outerVeqs.push_back(v);
                hadOuterVeq = true;
              } else {
                veqsToCancel.insert(v);
              }
            }
            if (hadOuterVeq) {
              // v is known precisely here (a direct operand), so only refs
              // whose provenance can't be proven independent of it need to
              // be invalidated — see cancelBindingsAliasing.
              OpBuilder builder(op);
              for (Value v : outerVeqs)
                dataFlow.cancelBindingsAliasing(v, block, builder, op->getLoc(),
                                                cleanUps, op);
              aliasForBlock = true;
            }

            // Source 2: summary deposited into childMap by the inner
            // processOpWithRegions call for this op-with-regions.
            auto it = childMap.find(op);
            if (it != childMap.end()) {
              for (Value v : it->second) {
                if (isFromOutsideParent(v)) {
                  // Still from above — propagate one more level up so the
                  // scope that actually owns v also checks its own
                  // bindings.
                  parentMap[parent].push_back(v);
                  aliasForBlock = true;
                }
                // Always also attempt cancellation against *this* level's
                // own bindings, regardless of who owns v: this scope may
                // have threaded its own local wire state (e.g. a
                // loop-carried region argument aliased into op's subtree)
                // that must not survive past this aliasing event either.
                // Deferring only to the owning scope is not enough — by the
                // time that scope's own cancellation runs, this scope's
                // threading has already been built.
                veqsToCancel.insert(v);
              }
              childMap.erase(it);
            }

            // Apply binding cancellations for inner-scope veqs.
            dataFlow.cancelBindings(veqsToCancel, block, cleanUps, op);
          }

          // For any operation that creates a value of quantum reference type,
          // replace it with a null wire (if it is an AllocaOp) or unwrap the
          // reference to get the wire.
          if (opResultOfType(op, qrefTy)) {
            if (!quantumValues)
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
              operRef.emitError("use before def in function");
              signalPassFailure();
              return;
            }

            // Parent is not a function.
            if (!isDescendantOf(parent, memuse)) {
              // `block` is using a value from another scope.
              //
              // A memAnalysis-member alloca is destined for total
              // elimination: its every use gets rewritten to a null-wire
              // thread, and its defining op itself is erased at the end of
              // the pass (see cleanUps). It must never end up as the literal
              // operand of a surviving op, so the aliasForBlock path (which
              // deliberately keeps `useop` — a real, permanent dereference
              // of memuse — alive in the IR) cannot apply to it: doing so
              // leaves a dangling reference once the alloca is erased. Fall
              // through to the normal path instead, which folds useop away
              // via the same self-correcting replacement (into a null wire)
              // used for every other member reference.
              bool memuseIsEliminatedAlloca =
                  memuse.getDefiningOp() &&
                  memAnalysis.isMember(memuse.getDefiningOp());
              if (aliasForBlock && !memuseIsEliminatedAlloca &&
                  cudaq::quake::isQuantumReferenceType(memuse.getType())) {
                // A dynamic veq-aliasing event already occurred in this block:
                // a non-constant extract_ref from a veq defined outside this
                // scope acts as a barrier — all wire chains must be wrapped
                // back to their refs, and subsequent uses get a fresh unwrap.
                // We still create a promoted value (so it lands in liveInArgs
                // and the outer scope threads the pre-aliasing state into this
                // block) and a live-in block arg (so getLiveInToBlock can find
                // it). The binding is set to the fresh useop (unwrap), NOT to
                // the live-in arg — the gate sees the current ref state, not
                // the pre-aliasing state.
                dataFlow.createPromotedValue(parent, memuse);
                dataFlow.addLiveInToBlock(block, memuse); // creates block arg
                dataFlow.addBinding(block, memuse, useop);
                // useop stays in the IR (not replaced, not in cleanUps).
                return;
              }
              // Normal path: create a promoted value that dominates parent and
              // thread it in as a live-in block argument.
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
          if (auto unwrap = dyn_cast<cudaq::quake::UnwrapOp>(op)) {
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
          if (auto wrap = dyn_cast<cudaq::quake::WrapOp>(op)) {
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
              cudaq::quake::WrapOp::create(builder, term->getLoc(), sigma,
                                           liveOut);
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
                    LogOutputOpPattern>(ctx);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<RAW_QUANTUM_OPS, cudaq::quake::ResetOp,
                                 cudaq::quake::DeallocOp,
                                 cudaq::quake::LogOutputOp>(
        [](Operation *op) { return !cudaq::quake::hasNonVectorReference(op); });
    target.addLegalOp<cudaq::quake::UnwrapOp, cudaq::quake::WrapOp,
                      cudaq::quake::NullWireOp, cudaq::quake::SinkOp>();
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
