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
   cases where an allocation cannot be correctly moved or an llvm.stacksave,
   llvm.stackrestore call pair cannot be erased, this pass should ABSOLUTELY,
   POSITIVELY NEVER EVER do so. If a particular transport layer does not support
   certain valid operations, it is up to the conformity and/or verifier tools of
   that particular transport layer (not the core compiler) to decide what
   action(s) to take.
 */

static bool hasHighLevelControlFlow(func::FuncOp func) {
  if (func.walk([](Operation *op) {
            if (isa<cudaq::cc::ScopeOp, cudaq::cc::IfOp, cudaq::cc::LoopOp,
                    cudaq::cc::CreateLambdaOp>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return true;
  return false;
}

static bool hasStackCalls(func::FuncOp func) {
  if (func.walk([](Operation *op) {
            if (auto call = dyn_cast<func::CallOp>(op))
              if (call.getCallee() == cudaq::llvmStackSave ||
                  call.getCallee() == cudaq::llvmStackRestore)
                return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return true;
  return false;
}

namespace {
struct SFPAnalysis {
  explicit SFPAnalysis(func::FuncOp func) : func(func) {
    classifyAllocas();
    collectStackSaveCalls();
    modifiedKosaraju();
  }

  /// Return a list of all static sized cc.alloca ops that do not appear in the
  /// entry block of \p func.
  void classifyAllocas() {
    Block *entry = &func.getBody().front();
    func.walk([&](cudaq::cc::AllocaOp alloc) {
      if (alloc->getBlock() != entry) {
        if (!alloc.getSeqSize())
          candidates.push_back(alloc);
        else
          pinned.push_back(alloc);
      }
    });
  }

  void collectStackSaveCalls() {
    func.walk([&](func::CallOp call) {
      if (call.getCallee() == cudaq::llvmStackSave)
        stackSaveCalls.push_back(call);
    });
  }

  /// Find all strongly connected components in `O(v+e)`. In this modified
  /// version, a block is not a singleton SCC unless it has a backedge to
  /// itself. Note that a consequence of this algorithm is the SCC list will be
  /// built from outermost (largest) to innermost (smallest).
  void modifiedKosaraju() {
    DenseMap<Block *, unsigned> finishTimes;
    SmallVector<Block *> stack;
    dfs_build(&func.getBody().front(), finishTimes, stack);
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

  /// Find the smallest enclosing SCC for \p op if any.
  std::optional<unsigned> findSCC(Operation *op) {
    unsigned size = sccList.size();
    for (unsigned i = 0; i < size; ++i) {
      // Scan the vector back to front to find the smallest SCC, if any.
      unsigned j = size - 1 - i;
      if (sccList[j].contains(op->getBlock()))
        return {j};
    }
    return {};
  }

  // Get the paired llvm.stackrestore given the llvm.stacksave.
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

  /// Find if any stacksave in the SCC, \p sccIdx, properly fences \p op.
  bool properlyFenced(unsigned sccIdx, Operation *op, DominanceInfo &dom) {
    for (auto call : stackSaveCalls)
      if (sccList[sccIdx].contains(call->getBlock()))
        if (properlyFenced(call, op, dom))
          return true;
    return false;
  }

  func::FuncOp func;
  SmallVector<cudaq::cc::AllocaOp> candidates;
  SmallVector<cudaq::cc::AllocaOp> pinned;
  SmallVector<func::CallOp> stackSaveCalls;
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
    if (hasHighLevelControlFlow(func))
      return;
    const bool stackCalls = hasStackCalls(func);
    if (!stackCalls && func.getBody().hasOneBlock())
      return;
    Block *entry = &func.getBody().front();
    DominanceInfo dom(func);
    SFPAnalysis analysis(func);

    LLVM_DEBUG(llvm::dbgs() << "Before stack frame preallocation:\n"
                            << func << "\n\n");

    auto *entryTerm = entry->getTerminator();
    for (auto cand : analysis.candidates) {
      // 1) If the candidate's block is not in a SCC, move it.
      auto optSCC = analysis.findSCC(cand);
      if (!optSCC) {
        LLVM_DEBUG(llvm::dbgs() << "moving: " << cand << '\n');
        cand->moveBefore(entryTerm);
        continue;
      }
      // 2) Otherwise (the candidate is in an SCC)
      //    Let S be the innermost such SCC.
      //    a) If the candidate is dominated by a stacksave also in S and the
      //       candidate dominates the paired stackrestore also in S, move it.
      if (analysis.properlyFenced(*optSCC, cand, dom)) {
        LLVM_DEBUG(llvm::dbgs() << "moving: " << cand << '\n');
        cand->moveBefore(entryTerm);
        continue;
      }
      //    b) Otherwise this is "unbounded" stack growth, so leave the
      //       candidate where it is and add it to the list of pinned allocs.
      analysis.pinned.push_back(cand);
    }

    DenseSet<func::CallOp> pinnedCalls;
    for (auto pin : analysis.pinned) {
      auto pinSCC = analysis.findSCC(pin);
      if (!pinSCC)
        continue;
      // Find the nearest call that dominates pin and pin dominates the
      // associated stack restore call. In this case pin prevents the nearest
      // stack save and stack restore from being removed and we want to mark
      // that call as pinned.
      bool match = false;
      func::CallOp nearest;
      for (auto call : analysis.stackSaveCalls) {
        auto callSCC = analysis.findSCC(call);
        if (!callSCC)
          continue;
        if (*callSCC == *pinSCC) {
          pinnedCalls.insert(call);
          match = true;
          break;
        }
        if (analysis.properlyFenced(call, pin, dom) && *callSCC < *pinSCC)
          nearest = call;
      }
      if (!match && nearest)
        pinnedCalls.insert(nearest);
    }

    // Clean up the stack calls. For any stack save and stack restore call
    // pairs that were not marked pinned, remove them.
    for (auto dead : analysis.stackSaveCalls) {
      if (pinnedCalls.contains(dead))
        continue;
      // Get the stackrestore call and delete it.
      auto users = dead->getUsers();
      if (++users.begin() != users.end()) {
        LLVM_DEBUG(llvm::dbgs() << "IR is malformed, must be exactly 1 user.");
        break;
      }
      users.begin()->dropAllReferences();
      users.begin()->erase();

      // Delete this stacksave call.
      LLVM_DEBUG(llvm::dbgs() << "deleting: " << dead << '\n');
      dead->dropAllReferences();
      dead->erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "After stack frame preallocation:\n"
                            << func << "\n\n");
  }
};
} // namespace
