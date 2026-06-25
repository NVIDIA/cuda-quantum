/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/WireBorrowAnalysis.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <optional>

#define DEBUG_TYPE "wire-borrow-analysis"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Part 1: wire threads.
//===----------------------------------------------------------------------===//

/// Equivalence classes of `!quake.wire` values. All values in one class are the
/// same physical wire at different points in its life.
namespace {
class WireThreads {
public:
  /// Merge the threads of \p a and \p b if both are wires.
  void unite(Value a, Value b) {
    if (!isa<cudaq::quake::WireType>(a.getType()) ||
        !isa<cudaq::quake::WireType>(b.getType()))
      return;
    threads.unionSets(a, b);
  }

  /// Merge element-wise. Mismatched shapes are not forwarding relations the
  /// analysis understands, so they are ignored.
  void forward(ValueRange from, ValueRange to) {
    if (from.size() != to.size())
      return;
    for (auto [f, t] : llvm::zip(from, to))
      unite(f, t);
  }

  /// A value that identifies the thread containing \p v. Only stable once all
  /// merging is done.
  Value root(Value v) { return threads.getOrInsertLeaderValue(v); }

private:
  llvm::EquivalenceClasses<Value> threads;
};
} // namespace

// In every variant the initial arguments reach the region that is entered
// first, `cc.condition` forwards its operands to the do region and to whichever
// of the else region or the loop results is taken on exit, `cc.continue`
// chooses step/while (or exits, from else), and `cc.break` exits.

/// Where a `continue` from \p region of \p loop sends its operands.
static ValueRange loopContinueTarget(cudaq::cc::LoopOp loop, Region *region) {
  if (region == &loop.getBodyRegion())
    return loop.hasStep() ? ValueRange(loop.getStepArguments())
                          : ValueRange(loop.getWhileArguments());
  if (region == &loop.getStepRegion())
    return loop.getWhileArguments();
  if (region == &loop.getElseRegion())
    return loop.getResults();
  return {};
}

/// Place the wire values of \p func into threads.
static void buildThreads(func::FuncOp func, WireThreads &threads) {
  func.walk([&](Operation *op) {
    if (auto loop = dyn_cast<cudaq::cc::LoopOp>(op)) {
      if (loop.isPostConditional()) {
        if (!loop.getBodyRegion().empty())
          threads.forward(loop.getOperands(), loop.getDoEntryArguments());
      } else if (!loop.getWhileRegion().empty()) {
        threads.forward(loop.getOperands(), loop.getWhileArguments());
      }
      return;
    }
    if (auto cond = dyn_cast<cudaq::cc::ConditionOp>(op)) {
      auto loop = cast<cudaq::cc::LoopOp>(cond->getParentOp());
      if (!loop.getBodyRegion().empty())
        threads.forward(cond.getResults(), loop.getDoEntryArguments());
      if (loop.hasPythonElse())
        threads.forward(cond.getResults(), loop.getElseEntryArguments());
      else
        threads.forward(cond.getResults(), loop.getResults());
      return;
    }
    if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(op)) {
      Operation *parent = cont->getParentOp();
      if (auto loop = dyn_cast<cudaq::cc::LoopOp>(parent))
        threads.forward(cont.getOperands(),
                        loopContinueTarget(loop, cont->getParentRegion()));
      else
        threads.forward(cont.getOperands(), parent->getResults());
      return;
    }
    if (auto brk = dyn_cast<cudaq::cc::BreakOp>(op)) {
      threads.forward(brk.getOperands(), brk->getParentOp()->getResults());
      return;
    }
    if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op)) {
      if (ifOp.hasThen())
        threads.forward(ifOp.getLinearArgs(), ifOp.getThenEntryArguments());
      if (ifOp.hasElse())
        threads.forward(ifOp.getLinearArgs(), ifOp.getElseEntryArguments());
      return;
    }
    if (auto br = dyn_cast<BranchOpInterface>(op)) {
      for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
        SuccessorOperands so = br.getSuccessorOperands(i);
        Block *dest = op->getSuccessor(i);
        threads.forward(
            so.getForwardedOperands(),
            dest->getArguments().drop_front(so.getProducedOperandCount()));
      }
      return;
    }
    // Quake operations thread their wire operands to their wire results, in
    // order.
    if (op->getDialect() && op->getDialect()->getNamespace() == "quake") {
      SmallVector<Value> wireIn, wireOut;
      for (Value v : op->getOperands())
        if (isa<cudaq::quake::WireType>(v.getType()))
          wireIn.push_back(v);
      for (Value v : op->getResults())
        if (isa<cudaq::quake::WireType>(v.getType()))
          wireOut.push_back(v);
      threads.forward(wireIn, wireOut);
    }
  });
}

//===----------------------------------------------------------------------===//
// Part 2: dataflow over program points.
//===----------------------------------------------------------------------===//

/// A program point. A point is either an operation, or the point just after the
/// operation has finished.
using Point = llvm::PointerIntPair<Operation *, 1, bool>;

/// The wires borrowed at a program point.
namespace {
struct State {
  bool reached = false;
  /// Borrowed on at least one path reaching this point.
  llvm::SmallBitVector may;
  /// Borrowed on every path reaching this point.
  llvm::SmallBitVector must;
};
} // namespace

/// Join \p src into \p dst. Return true if \p dst changed.
static bool join(State &dst, const State &src) {
  if (!src.reached)
    return false;
  if (!dst.reached) {
    dst = src;
    return true;
  }
  llvm::SmallBitVector may = dst.may;
  may |= src.may;
  llvm::SmallBitVector must = dst.must;
  must &= src.must;
  bool changed = may != dst.may || must != dst.must;
  dst.may = std::move(may);
  dst.must = std::move(must);
  return changed;
}

/// Append to \p out the points that can execute immediately after \p p.
static void successors(Point p, SmallVectorImpl<Point> &out) {
  auto enter = [&](Block *b) {
    if (b && !b->empty())
      out.push_back(Point(&b->front(), false));
  };
  auto enterRegion = [&](Region &r) {
    if (!r.empty())
      enter(&r.front());
  };
  auto exitOf = [&](Operation *x) { out.push_back(Point(x, true)); };
  auto nextAfter = [&](Operation *x) {
    if (Operation *n = x->getNextNode())
      out.push_back(Point(n, false));
  };

  Operation *op = p.getPointer();
  if (p.getInt()) {
    nextAfter(op);
    return;
  }

  if (auto loop = dyn_cast<cudaq::cc::LoopOp>(op)) {
    enterRegion(loop.isPostConditional() ? loop.getBodyRegion()
                                         : loop.getWhileRegion());
    return;
  }
  if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op)) {
    enterRegion(ifOp.getThenRegion());
    if (ifOp.hasElse())
      enterRegion(ifOp.getElseRegion());
    else
      exitOf(op);
    return;
  }
  if (auto scope = dyn_cast<cudaq::cc::ScopeOp>(op)) {
    enterRegion(scope.getInitRegion());
    return;
  }
  if (isa<cudaq::cc::ConditionOp>(op)) {
    auto loop = cast<cudaq::cc::LoopOp>(op->getParentOp());
    enterRegion(loop.getBodyRegion());
    if (loop.hasPythonElse())
      enterRegion(loop.getElseRegion());
    else
      exitOf(loop);
    return;
  }
  if (isa<cudaq::cc::ContinueOp>(op)) {
    Operation *parent = op->getParentOp();
    auto loop = dyn_cast<cudaq::cc::LoopOp>(parent);
    if (!loop) {
      exitOf(parent);
      return;
    }
    Region *region = op->getParentRegion();
    if (region == &loop.getBodyRegion() && loop.hasStep())
      enterRegion(loop.getStepRegion());
    else if (region == &loop.getElseRegion())
      exitOf(loop);
    else
      enterRegion(loop.getWhileRegion());
    return;
  }
  if (isa<cudaq::cc::BreakOp>(op)) {
    exitOf(op->getParentOp());
    return;
  }
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    for (Block *succ : op->getSuccessors())
      enter(succ);
    return;
  }

  // Any other operation, including a region-owning operation this analysis does
  // not model, which is treated as opaque.
  nextAfter(op);
}

cudaq::opt::WireBorrowAnalysis::WireBorrowAnalysis(func::FuncOp func) {
  if (func.empty())
    return;

  // Enumerate the physical wires borrowed in the function.
  using WireKey = std::pair<Attribute, std::uint32_t>;
  DenseMap<WireKey, unsigned> keyIndex;
  SmallVector<cudaq::quake::BorrowWireOp> firstBorrowOfKey;
  SmallVector<SmallVector<Operation *, 2>> borrowsOfKey;
  auto keyOf = [&](cudaq::quake::BorrowWireOp b) {
    return WireKey{b.getSetNameAttr(), b.getIdentity()};
  };
  func.walk([&](cudaq::quake::BorrowWireOp borrow) {
    auto [iter, inserted] =
        keyIndex.try_emplace(keyOf(borrow), keyIndex.size());
    if (inserted) {
      firstBorrowOfKey.push_back(borrow);
      borrowsOfKey.emplace_back();
    }
    borrowsOfKey[iter->second].push_back(borrow);
  });
  if (keyIndex.empty())
    return;

  // Place wires in threads. Only threads that contain a borrow are tracked. The
  // dataflow state is the set of tracked threads that are currently borrowed,
  // since a `return_wire` releases exactly the thread it is given.
  WireThreads threads;
  buildThreads(func, threads);
  DenseMap<Value, unsigned> trackedIndex;
  SmallVector<SmallVector<unsigned, 1>> threadKeys;
  func.walk([&](cudaq::quake::BorrowWireOp borrow) {
    auto [iter, inserted] = trackedIndex.try_emplace(
        threads.root(borrow.getResult()), trackedIndex.size());
    if (inserted)
      threadKeys.emplace_back();
    unsigned k = keyIndex.lookup(keyOf(borrow));
    if (!llvm::is_contained(threadKeys[iter->second], k))
      threadKeys[iter->second].push_back(k);
  });
  const unsigned numThreads = trackedIndex.size();
  auto trackedThread = [&](Value v) -> std::optional<unsigned> {
    auto iter = trackedIndex.find(threads.root(v));
    if (iter == trackedIndex.end())
      return std::nullopt;
    return iter->second;
  };

  // A wire handed to an operation this analysis does not understand may have
  // its ownership transferred, so it is never definitely lost.
  llvm::SmallBitVector opaque(numThreads);
  func.walk([&](Operation *op) {
    if (isa<cudaq::quake::ReturnWireOp, func::ReturnOp>(op) ||
        isa<BranchOpInterface>(op))
      return;
    if (op->getDialect() && (op->getDialect()->getNamespace() == "quake" ||
                             op->getDialect()->getNamespace() == "cc"))
      return;
    for (Value v : op->getOperands())
      if (isa<cudaq::quake::WireType>(v.getType()))
        if (auto t = trackedThread(v))
          opaque.set(*t);
  });

  // Forward dataflow to a fixed point.
  DenseMap<Point, State> in;
  auto transfer = [&](Operation *op, State s) {
    if (auto borrow = dyn_cast<cudaq::quake::BorrowWireOp>(op)) {
      unsigned t = *trackedThread(borrow.getResult());
      s.may.set(t);
      s.must.set(t);
      return s;
    }
    if (auto ret = dyn_cast<cudaq::quake::ReturnWireOp>(op)) {
      if (auto t = trackedThread(ret.getTarget())) {
        s.may.reset(*t);
        s.must.reset(*t);
      } else {
        // Unknown wire; it may release any wire.
        s.must.reset();
      }
    }
    return s;
  };

  State entryState;
  entryState.reached = true;
  entryState.may.resize(numThreads);
  entryState.must.resize(numThreads);
  Point entry(&func.front().front(), false);
  in[entry] = entryState;
  SmallVector<Point> worklist{entry};
  while (!worklist.empty()) {
    Point p = worklist.pop_back_val();
    State out = p.getInt() ? in[p] : transfer(p.getPointer(), in[p]);
    SmallVector<Point, 2> succs;
    successors(p, succs);
    for (Point s : succs) {
      auto [iter, inserted] = in.try_emplace(s, State{});
      if (join(iter->second, out))
        worklist.push_back(s);
    }
  }

  auto addConflict = [&](Kind kind, Certainty certainty, Operation *op,
                         unsigned k) {
    Conflict c{kind,
               certainty,
               op,
               firstBorrowOfKey[k].getSetName(),
               firstBorrowOfKey[k].getIdentity(),
               {}};
    for (Operation *other : borrowsOfKey[k])
      if (other != op)
        c.otherBorrows.push_back(other);
    conflicts.push_back(std::move(c));
  };

  func.walk([&](Operation *op) {
    auto iter = in.find(Point(op, false));
    if (iter == in.end() || !iter->second.reached)
      return;
    const State &s = iter->second;
    if (auto borrow = dyn_cast<cudaq::quake::BorrowWireOp>(op)) {
      unsigned k = keyIndex.lookup(keyOf(borrow));
      // Definite if some thread is borrowed on every path and can only be this
      // wire. Possible if some thread that may carry this wire may be borrowed.
      std::optional<Certainty> found;
      for (unsigned t : s.may.set_bits()) {
        if (!llvm::is_contained(threadKeys[t], k))
          continue;
        if (s.must.test(t) && threadKeys[t].size() == 1) {
          found = Certainty::Definite;
          break;
        }
        found = Certainty::Possible;
      }
      if (found)
        addConflict(Kind::DoubleBorrow, *found, op, k);
      return;
    }
    if (auto ret = dyn_cast<cudaq::quake::ReturnWireOp>(op)) {
      if (!trackedThread(ret.getTarget()))
        unresolvedReturns.push_back(op);
      return;
    }
    if (isa<func::ReturnOp, cudaq::cc::ReturnOp>(op)) {
      // A borrowed wire that leaves in a result has not been lost.
      llvm::SmallBitVector escapes(numThreads);
      bool unresolvedEscape = false;
      for (Value v : op->getOperands()) {
        if (!isa<cudaq::quake::WireType>(v.getType()))
          continue;
        if (auto t = trackedThread(v))
          escapes.set(*t);
        else
          unresolvedEscape = true;
      }
      for (unsigned t : s.may.set_bits()) {
        if (escapes.test(t))
          continue;
        const bool definite =
            s.must.test(t) && !opaque.test(t) && !unresolvedEscape;
        for (unsigned k : threadKeys[t])
          addConflict(Kind::Unreturned,
                      definite && threadKeys[t].size() == 1
                          ? Certainty::Definite
                          : Certainty::Possible,
                      op, k);
      }
    }
  });
}
