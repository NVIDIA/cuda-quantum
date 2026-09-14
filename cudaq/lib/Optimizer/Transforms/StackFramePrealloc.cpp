/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_STACKFRAMEPREALLOC
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "stack-frame-prealloc"

using namespace mlir;

/**
   \file

   Classic low-level stack frame preallocation pass.

   NB: This optimization must be correct above all else. That means that in
   cases where an allocation cannot be correctly moved or an llvm.stacksave.p0,
   llvm.stackrestore.p0 call pair cannot be erased, this pass should ABSOLUTELY,
   POSITIVELY NEVER EVER do so. If a particular transport layer does not support
   certain valid operations, it is up to the conformity and/or verifier tools of
   that particular transport layer (not the core compiler) to decide what
   action(s) to take.

   This pass is applied independently to each "allocation scope": a
   func.func, or a cc.create_lambda (a λ expression's body is an entirely
   separate callable activation -- an alloca inside it can never be lifted
   into the enclosing func.func's frame, since the closure may be invoked any
   number of times, on its own schedule, independent of the enclosing
   function). Every cc.create_lambda found is processed recursively, exactly
   as if it were its own func.func, before the enclosing scope's own analysis
   runs; a cc.create_lambda's contents never contribute to (or are moved by)
   an enclosing scope's analysis. See StackFramePreallocPass::processScope.

   Within a single such scope, cc.scope, cc.if, and cc.loop may all appear.
   cc.scope and cc.if are single-entry/single-exit (SESE): entering either is
   equivalent to an implicit stacksave and leaving it (however control
   leaves, including an unwind) is equivalent to the paired stackrestore, so
   they are usable as region fences a structural form of the same
   stacksave/stackrestore-pair fencing for lowered CFG form. cc.loop is clearly
   different: its while/do/step regions form a SCC and may execute many times
   per invocation of the loop, so it is instead treated as its own atomic loop
   boundary exactly like any other SCC of the function's CFG.

   The IR guarantees that a cc.alloca can never appear directly inside a
   cc.if or cc.loop's own regions without an intervening cc.scope that is
   itself nested inside that same cc.if/cc.loop (see validateAllocaNesting).
   If that invariant is ever violated, this pass declines to touch the
   violating scope (emitting a warning) rather than risk reasoning about IR
   it cannot correctly analyze.
 */

// Return the region that holds \p root's own body: a func.func's body, or a
// cc.create_lambda's init region.
static Region &getRootBody(Operation *root) {
  if (auto func = dyn_cast<func::FuncOp>(root))
    return func.getBody();
  return cast<cudaq::cc::CreateLambdaOp>(root).getInitRegion();
}

// Walk every operation transitively nested in \p root's own regions, EXCEPT do
// not descend into the body of any cc.create_lambda encountered: a
// create_lambda is an entirely separate allocation scope and is analyzed
// independently, as if it were a distinct func.func.
static void walkOwnScope(Operation *root,
                         llvm::function_ref<void(Operation *)> callback) {
  for (Region &region : root->getRegions())
    for (Block &block : region)
      for (Operation &op : block) {
        callback(&op);
        if (!isa<cudaq::cc::CreateLambdaOp>(op))
          walkOwnScope(&op, callback);
      }
}

// Collect the cc.create_lambda ops directly reachable from \p root without
// crossing another cc.create_lambda's boundary first (i.e. the immediate
// nested lambdas, wherever they are lexically -- possibly inside a
// cc.scope/cc.if/cc.loop of \p root).
static void
collectImmediateLambdas(Operation *root,
                        SmallVectorImpl<cudaq::cc::CreateLambdaOp> &lambdas) {
  walkOwnScope(root, [&](Operation *op) {
    if (auto lambda = dyn_cast<cudaq::cc::CreateLambdaOp>(op))
      lambdas.push_back(lambda);
  });
}

// Returns true if and only if control can flow from \p region, through zero
// or more of \p regionOp's own other regions, back into \p region itself. That
// is regionOp has a backedge internally.
static bool regionMayRepeat(RegionBranchOpInterface regionOp, Region *region) {
  DenseMap<Region *, SmallVector<Region *>> succs;
  for (Region &r : regionOp->getRegions()) {
    if (r.empty())
      continue;
    for (Block &b : r) {
      auto *term = b.getTerminator();
      auto termIface = dyn_cast<RegionBranchTerminatorOpInterface>(term);
      if (!termIface)
        continue;
      SmallVector<RegionSuccessor> successors;
      regionOp.getSuccessorRegions(termIface, successors);
      for (auto &s : successors)
        if (Region *to = s.getSuccessor())
          succs[&r].push_back(to);
    }
  }
  SmallPtrSet<Region *, 4> visited;
  SmallVector<Region *> worklist(succs[region].begin(), succs[region].end());
  while (!worklist.empty()) {
    Region *cur = worklist.pop_back_val();
    if (cur == region)
      return true;
    if (!visited.insert(cur).second)
      continue;
    for (Region *next : succs[cur])
      worklist.push_back(next);
  }
  return false;
}

// Verify that every cc.alloca directly within \p root's own scope (i.e. not
// inside a further-nested cc.create_lambda, which is validated independently by
// its own recursive processScope call) satisfies the IR invariant: if it is
// nested inside a cc.if or cc.loop, it must also be nested inside a cc.scope
// that is itself nested inside that same cc.if/cc.loop. This is a structural
// property this pass relies on but does not itself enforce or repair; if it is
// violated, something upstream produced invalid IR, so this pass declines to
// run over \p root at all (emitting a warning) rather than act on IR it cannot
// reason about correctly.
static LogicalResult validateAllocaNesting(Operation *root) {
  LogicalResult result = success();
  walkOwnScope(root, [&](Operation *op) {
    auto alloc = dyn_cast<cudaq::cc::AllocaOp>(op);
    if (!alloc)
      return;

    // Once a cc.scope ancestor is seen, it satisfies the requirement for every
    // cc.if/cc.loop further up the chain too: scope-nesting is transitive, so a
    // scope reached through an intervening cc.if still sits inside any cc.loop
    // that encloses that cc.if. Do not reset sawScope on each cc.if/cc.loop.
    bool sawScope = false;
    for (Operation *p = alloc->getParentOp(); p != root; p = p->getParentOp()) {
      if (isa<cudaq::cc::ScopeOp>(p)) {
        sawScope = true;
        continue;
      }
      if (isa<cudaq::cc::IfOp, cudaq::cc::LoopOp>(p) && !sawScope) {
        alloc.emitWarning("cc.alloca is nested inside a cc.if or cc.loop "
                          "without an intervening cc.scope; skipping "
                          "stack-frame-prealloc for this scope");
        result = failure();
        return;
      }
    }
  });
  return result;
}

static bool hasStackCalls(Operation *root) {
  bool found = false;
  walkOwnScope(root, [&](Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op))
      if (call.getCallee() == cudaq::llvmStackSave ||
          call.getCallee() == cudaq::llvmStackRestore)
        found = true;
  });
  return found;
}

/// Return true if and only if \p root has a `cc.alloca` `op` that does not
/// appear in \p entry, including one nested inside a `cc.scope` / `cc.if` /
/// `cc.loop`'s own (separate) region.
static bool hasNonEntryAlloca(Operation *root, Block *entry) {
  bool found = false;
  walkOwnScope(root, [&](Operation *op) {
    if (auto alloc = dyn_cast<cudaq::cc::AllocaOp>(op))
      if (alloc->getBlock() != entry)
        found = true;
  });
  return found;
}

namespace {
struct SFPAnalysis {
  explicit SFPAnalysis(Operation *root) : root(root), body(getRootBody(root)) {
    classifyAllocas();
    collectStackSaveCalls();
    collectRegionFences();
    modifiedKosaraju();
  }

  // Return a list of all static sized cc.alloca ops that do not appear in/ the
  // entry block of \p root's own body.
  void classifyAllocas() {
    Block *entry = &body.front();
    walkOwnScope(root, [&](Operation *op) {
      auto alloc = dyn_cast<cudaq::cc::AllocaOp>(op);
      if (!alloc || alloc->getBlock() == entry)
        return;
      if (!alloc.getSeqSize())
        candidates.push_back(alloc);
      else
        pinned.push_back(alloc);
    });
  }

  void collectStackSaveCalls() {
    walkOwnScope(root, [&](Operation *op) {
      if (auto call = dyn_cast<func::CallOp>(op))
        if (call.getCallee() == cudaq::llvmStackSave)
          stackSaveCalls.push_back(call);
    });
  }

  // Collect every op that is SESE per including cc.scope, cc.if as region
  // fences. These are a structured form of the same stacksave/stackrestore pair
  // fencing. A cc.loop is never itself a fence, any alloca directly inside one
  // of its regions must in turn be wrapped in one of these region fences.
  void collectRegionFences() {
    walkOwnScope(root, [&](Operation *op) {
      auto regionOp = dyn_cast<RegionBranchOpInterface>(op);
      if (!regionOp || isa<cudaq::cc::LoopOp>(op))
        return;
      regionFences.push_back(op);
    });
  }

  // Find all strongly connected components in `O(v+e)` over \p root's own
  // (flat) block graph, AND independently over the (flat) block graph of
  // every other region nested within \p root's own scope (a cc.scope/cc.if's
  // single region, or each of a cc.loop's while/do/step regions).
  //
  // A region fence only guards against repetition of the construct that owns
  // it, not against a lowered CFG loop (a block with a backedge to itself or an
  // ancestor) already present inside its own body, so that body's block graph
  // must be searched for SCCs too. In this modified version, a block is not a
  // singleton SCC unless it has a backedge to itself.
  void modifiedKosaraju() {
    computeSCCs(body);
    walkOwnScope(root, [&](Operation *op) {
      if (isa<cudaq::cc::CreateLambdaOp>(op))
        return;
      for (Region &r : op->getRegions())
        if (!r.empty())
          computeSCCs(r);
    });
  }

  void computeSCCs(Region &region) {
    DenseMap<Block *, unsigned> finishTimes;
    SmallVector<Block *> stack;
    dfs_build(&region.front(), finishTimes, stack);
    for (Block *visit : llvm::reverse(stack)) {
      llvm::SmallPtrSet<Block *, 4> scc;
      reverse_dfs(scc, visit, finishTimes[visit], finishTimes);
      if (!scc.empty()) {
        LLVM_DEBUG({
          llvm::dbgs() << "found SCC: [{\n";
          for (auto *b : scc) {
            llvm::dbgs() << '\t' << b << '\n';
            b->print(llvm::dbgs());
          }
          llvm::dbgs() << "}]\n";
        });
        sccList.emplace_back(std::move(scc));
      }
    }
  }

  // Step 1 of Kosaraju.
  void dfs_build(Block *block, DenseMap<Block *, unsigned> &finishTimes,
                 SmallVectorImpl<Block *> &stack) {
    if (finishTimes.count(block))
      return;
    unsigned time = finishTimes.size();
    finishTimes[block] = time;
    stack.push_back(block);
    for (Block *succ : block->getSuccessors())
      dfs_build(succ, finishTimes, stack);
  }

  // Step 2 of Kosaraju.
  void reverse_dfs(llvm::SmallPtrSet<Block *, 4> &scc, Block *block,
                   unsigned currentTime,
                   DenseMap<Block *, unsigned> &finishTimes) {
    for (Block *succ : block->getSuccessors()) {
      // Test for a backedge.
      if (finishTimes[succ] <= currentTime) {
        auto pair = scc.insert(succ);
        if (pair.second)
          reverse_dfs(scc, succ, currentTime, finishTimes);
      }
    }
  }

  // The smallest repeating construct enclosing some op, if any. Exactly one of
  // `sccIdx` (an index into sccList, i.e. a strongly connected component of
  // some region's own flat block graph) or `loopOp` (a cc.loop whose
  // repeating region encloses the op) is set when a repeating construct was
  // found.
  struct LoopBoundary {
    std::optional<unsigned> sccIdx;
    Operation *loopOp = nullptr;
    explicit operator bool() const { return sccIdx.has_value() || loopOp; }
  };

  // Find the smallest SCC (of any region's own block graph, see
  // modifiedKosaraju) that \p block belongs to, if any.
  std::optional<unsigned> findSCC(Block *block) {
    unsigned size = sccList.size();
    for (unsigned i = 0; i < size; ++i) {
      // Scan the vector back to front to find the smallest SCC, if any.
      unsigned j = size - 1 - i;
      if (sccList[j].contains(block))
        return j;
    }
    return std::nullopt;
  }

  // Find the smallest repeating construct enclosing \p op, if any. \p op may
  // be nested inside any number of cc.scope/cc.if regions and cc.loop
  // regions. A repeating construct may be a lowered CFG loop (a block with a
  // backedge) found directly in the block graph of the innermost region
  // enclosing \p op, or, one level further out, a cc.loop whose own
  // while/do/step regions form a cycle. Every level walked must be checked
  // for both: a cc.scope/cc.if's own region is a SESE fence for repetition
  // of *that* construct, but it does not itself guard against a lowered CFG
  // loop already present inside its body.
  LoopBoundary findEnclosingLoop(Operation *op) {
    Block *block = op->getBlock();
    while (true) {
      if (auto sccIdx = findSCC(block))
        return {sccIdx, nullptr};
      if (block->getParentOp() == root)
        return {};
      Operation *parentOp = block->getParentOp();
      if (auto regionOp = dyn_cast<RegionBranchOpInterface>(parentOp))
        if (regionMayRepeat(regionOp, block->getParent()))
          return {std::nullopt, parentOp};
      block = parentOp->getBlock();
    }
  }

  // Get the paired llvm.stackrestore.p0 given the llvm.stacksave.p0.
  func::CallOp getStackRestore(func::CallOp stackSave) {
    if (stackSave->getUsers().begin() != stackSave->getUsers().end())
      if (auto c = dyn_cast<func::CallOp>(*stackSave->getUsers().begin()))
        return c;
    return {};
  }

  /// Does \p call properly fence \p op?
  bool properlyFenced(func::CallOp stackSaveCall, Operation *op,
                      DominanceInfo &dom) {
    auto stackRestoreCall = getStackRestore(stackSaveCall);
    return dom.properlyDominates(stackSaveCall, op) &&
           dom.properlyDominates(op, stackRestoreCall);
  }

  // Does the region-fence op \p fence properly fence \p op? Unlike a
  // stacksave/stackrestore call pair, no dominance check against separate
  // start/end ops is needed.
  bool properlyFenced(Operation *fence, Operation *op) {
    return fence->isProperAncestor(op);
  }

  // Find if any stacksave.p0 pair or region fence properly fences \p op within
  // \p boundary, the smallest repeating construct enclosing it (see
  // findEnclosingLoop).
  bool properlyFenced(const LoopBoundary &boundary, Operation *op,
                      DominanceInfo &dom) {
    if (boundary.loopOp) {
      // The IR invariant checked by validateAllocaNesting guarantees a
      // cc.scope sits between op and boundary.loopOp; find it.
      for (auto *fence : regionFences)
        if (boundary.loopOp->isProperAncestor(fence) &&
            fence->isProperAncestor(op))
          return true;
      return false;
    }
    unsigned sccIdx = *boundary.sccIdx;
    for (auto call : stackSaveCalls)
      if (sccList[sccIdx].contains(call->getBlock()))
        if (properlyFenced(call, op, dom))
          return true;
    for (auto *fence : regionFences)
      if (sccList[sccIdx].contains(fence->getBlock()))
        if (properlyFenced(fence, op))
          return true;
    return false;
  }

  Operation *root;
  Region &body;
  SmallVector<cudaq::cc::AllocaOp> candidates;
  SmallVector<cudaq::cc::AllocaOp> pinned;
  SmallVector<func::CallOp> stackSaveCalls;
  SmallVector<Operation *> regionFences;
  SmallVector<llvm::SmallPtrSet<Block *, 4>> sccList;
};

class StackFramePreallocPass
    : public cudaq::opt::impl::StackFramePreallocBase<StackFramePreallocPass> {
public:
  using StackFramePreallocBase::StackFramePreallocBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.empty())
      return;
    processScope(func);
  }

  /// Process \p root (a `func.func` or a `cc.create_lambda`) as an independent
  /// allocation scope. Any `cc.create_lambda` directly nested within \p root
  /// is processed recursively first, as its own independent scope before \p
  /// root's own analysis runs.
  void processScope(Operation *root) {
    Region &body = getRootBody(root);
    if (body.empty())
      return;

    SmallVector<cudaq::cc::CreateLambdaOp> lambdas;
    collectImmediateLambdas(root, lambdas);
    for (auto lambda : lambdas)
      processScope(lambda);

    if (failed(validateAllocaNesting(root)))
      return;

    Block *entry = &body.front();
    const bool stackCalls = hasStackCalls(root);
    if (!stackCalls && !hasNonEntryAlloca(root, entry))
      return;

    DominanceInfo dom(root);
    SFPAnalysis analysis(root);

    LLVM_DEBUG(llvm::dbgs() << "Before stack frame preallocation:\n"
                            << *root << "\n\n");

    // Hoisted candidates are inserted immediately before the first
    // region-bearing op in entry (a cc.scope/cc.if/cc.loop/cc.create_lambda)
    // if there is one, or before entry's terminator otherwise. Entry's
    // terminator alone used to be a safe insertion point back when this pass
    // only ran over an already lowered, high-level-control-flow-free CFG,
    // where entry always just fell through to the function's real code via
    // a branch, so nothing between the last existing entry op and the
    // terminator could possibly use a not-yet-hoisted candidate. Now that
    // entry may directly hold the function's own structured control flow
    // that *uses* the candidate being hoisted, inserting before the
    // terminator (which can now be the function's own return) would place
    // the hoisted alloca *after* that code; inserting before the first such
    // op keeps ordinary leading entry-block ops (e.g. classical setup) in
    // their original relative position while still preceding anything that
    // could actually use the candidate.
    Operation *entryInsertPt = entry->getTerminator();
    for (Operation &op : *entry)
      if (op.getNumRegions() != 0) {
        entryInsertPt = &op;
        break;
      }
    for (auto cand : analysis.candidates) {
      // 1) If the candidate is not enclosed by any repeating construct, move
      //    it.
      auto boundary = analysis.findEnclosingLoop(cand);
      if (!boundary) {
        LLVM_DEBUG(llvm::dbgs() << "moving: " << cand << '\n');
        cand->moveBefore(entryInsertPt);
        continue;
      }
      // 2) Otherwise (the candidate is enclosed by a repeating construct, R.
      //    a) If the candidate is properly fenced within R, move it.
      if (analysis.properlyFenced(boundary, cand, dom)) {
        LLVM_DEBUG(llvm::dbgs() << "moving: " << cand << '\n');
        cand->moveBefore(entryInsertPt);
        continue;
      }
      //    b) Otherwise this is "unbounded" stack growth, so pin it.
      analysis.pinned.push_back(cand);
    }

    DenseSet<func::CallOp> pinnedCalls;
    for (auto pin : analysis.pinned) {
      auto pinBoundary = analysis.findEnclosingLoop(pin);
      if (!pinBoundary)
        continue;
      // Find every stacksave.p0 call that either shares pin's own smallest
      // enclosing boundary or properly fences pin from further out. Pin them.
      for (auto call : analysis.stackSaveCalls) {
        auto callBoundary = analysis.findEnclosingLoop(call);
        if (!callBoundary)
          continue;
        bool sameBoundary =
            (pinBoundary.loopOp && callBoundary.loopOp == pinBoundary.loopOp) ||
            (pinBoundary.sccIdx && callBoundary.sccIdx == pinBoundary.sccIdx);
        if (sameBoundary) {
          pinnedCalls.insert(call);
          break;
        }
        if (analysis.properlyFenced(call, pin, dom))
          pinnedCalls.insert(call);
      }
    }

    // Clean up the stack calls. For any stack save and stack restore call
    // pairs that were not marked pinned, remove them.
    for (auto dead : analysis.stackSaveCalls) {
      if (pinnedCalls.contains(dead))
        continue;
      // Get the stackrestore.p0 call and delete it.
      auto users = dead->getUsers();
      if (++users.begin() != users.end()) {
        LLVM_DEBUG(llvm::dbgs() << "IR is malformed, must be exactly 1 user.");
        break;
      }
      users.begin()->dropAllReferences();
      users.begin()->erase();

      // Delete this stacksave.p0 call.
      LLVM_DEBUG(llvm::dbgs() << "deleting: " << dead << '\n');
      dead->dropAllReferences();
      dead->erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "After stack frame preallocation:\n"
                            << *root << "\n\n");
  }
};
} // namespace
