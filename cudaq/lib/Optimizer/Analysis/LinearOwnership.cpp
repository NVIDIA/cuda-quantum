/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/LinearOwnership.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"

#include <algorithm>
#include <optional>

using namespace mlir;

namespace {

/// One use of a linear value, positioned relative to the region that defines
/// the value.
struct LinearUse {
  Operation *owner;
  /// (region-bearing ancestor, region number) steps from the defining region
  /// down to the region that holds `owner`. Empty when `owner` sits directly
  /// in the defining region, or when it is detached from the IR.
  SmallVector<std::pair<Operation *, unsigned>, 4> chain;

  /// The op at `depth` on the path to `owner`: a chain ancestor, or the owner
  /// itself once the chain is exhausted.
  Operation *frontier(size_t depth) const {
    return depth < chain.size() ? chain[depth].first : owner;
  }
  bool endsAt(size_t depth) const { return depth >= chain.size(); }
};

Region *definingRegion(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return argument.getOwner()->getParent();
  return value.getDefiningOp()->getParentRegion();
}

Block *definingBlock(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return argument.getOwner();
  return value.getDefiningOp()->getBlock();
}

LinearUse makeUse(Region *defining, Operation *owner) {
  LinearUse use{owner, {}};
  Operation *op = owner;
  while (op->getParentRegion() != defining) {
    Region *region = op->getParentRegion();
    Operation *parent = region ? region->getParentOp() : nullptr;
    if (!parent) {
      // A detached consumer cannot be placed; treat it as a top-level use.
      use.chain.clear();
      return use;
    }
    use.chain.emplace_back(parent, region->getRegionNumber());
    op = parent;
  }
  std::reverse(use.chain.begin(), use.chain.end());
  return use;
}

bool borrows(Operation *owner) {
  return owner->hasTrait<cudaq::LinearBorrowsOperands>();
}

/// True when both consuming uses can execute for one owner.
bool consumesConflict(const LinearUse &left, const LinearUse &right) {
  for (size_t depth = 0;; ++depth) {
    Operation *leftOp = left.frontier(depth);
    Operation *rightOp = right.frontier(depth);
    if (leftOp != rightOp)
      // Two distinct ops in one block run in sequence. Ops in different
      // blocks of a CFG region may sit on exclusive paths.
      return leftOp->getBlock() == rightOp->getBlock();
    // The same op consumes twice (two operands), or one use sits inside the
    // region of the op making the other use: both run.
    if (left.endsAt(depth) || right.endsAt(depth))
      return true;
    if (left.chain[depth].second != right.chain[depth].second)
      return !leftOp->hasTrait<cudaq::LinearExclusiveRegions>();
  }
}

/// True when the observation provably runs after the consume.
bool borrowFollowsConsume(const LinearUse &borrow, const LinearUse &consume) {
  for (size_t depth = 0;; ++depth) {
    Operation *borrowOp = borrow.frontier(depth);
    Operation *consumeOp = consume.frontier(depth);
    if (borrowOp == consumeOp) {
      if (borrow.endsAt(depth) || consume.endsAt(depth))
        return false;
      if (borrow.chain[depth].second != consume.chain[depth].second)
        return false;
      continue;
    }
    if (!borrowOp->getBlock() || borrowOp->getBlock() != consumeOp->getBlock())
      return false;
    return consumeOp->isBeforeInBlock(borrowOp);
  }
}

/// Program order of two uses whose frontiers share a block, for messages.
std::pair<const LinearUse *, const LinearUse *>
programOrder(const LinearUse &left, const LinearUse &right) {
  for (size_t depth = 0;; ++depth) {
    Operation *leftOp = left.frontier(depth);
    Operation *rightOp = right.frontier(depth);
    if (leftOp == rightOp) {
      if (left.endsAt(depth) || right.endsAt(depth) ||
          left.chain[depth].second != right.chain[depth].second)
        return {&left, &right};
      continue;
    }
    if (leftOp->getBlock() && leftOp->getBlock() == rightOp->getBlock() &&
        rightOp->isBeforeInBlock(leftOp))
      return {&right, &left};
    return {&left, &right};
  }
}

unsigned nonEmptyRegionCount(Operation *op) {
  return llvm::count_if(op->getRegions(),
                        [](Region &region) { return !region.empty(); });
}

class ScopeAnalysis {
public:
  ScopeAnalysis(Operation *scope, cudaq::opt::LinearOwnershipReport &report,
                bool firstOnly)
      : scope(scope), report(report), firstOnly(firstOnly) {}

  void run() {
    scope->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op != scope && op->hasTrait<cudaq::LinearOwnershipScope>())
        return WalkResult::skip();
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            checkValue(argument);
      if (op != scope)
        for (Value result : op->getResults())
          checkValue(result);
      return done() ? WalkResult::interrupt() : WalkResult::advance();
    });
  }

private:
  bool done() const { return firstOnly && !report.violations.empty(); }

  void checkValue(Value value) {
    if (done() || !cudaq::isLinearOwnershipType(value.getType()))
      return;
    ++report.linearValues;

    // Fast path: one consume in the defining block cannot cross region
    // control or conflict with another use.
    if (value.hasOneUse()) {
      Operation *owner = value.use_begin()->getOwner();
      if (!borrows(owner) && owner->getBlock() == definingBlock(value))
        return;
    }

    Region *defining = definingRegion(value);
    SmallVector<LinearUse, 4> consumes;
    SmallVector<LinearUse, 4> observations;
    for (OpOperand &operand : value.getUses()) {
      LinearUse use = makeUse(defining, operand.getOwner());
      (borrows(use.owner) ? observations : consumes).push_back(std::move(use));
    }
    // Use lists are in reverse creation order; restore source order so
    // diagnostics name the first consume first.
    std::reverse(consumes.begin(), consumes.end());
    std::reverse(observations.begin(), observations.end());

    bool mayDrop = cast<cudaq::LinearTypeInterface>(value.getType()).mayDrop();
    Operation *definer = value.getDefiningOp();
    Location definitionLoc =
        definer
            ? definer->getLoc()
            : cast<BlockArgument>(value).getOwner()->getParentOp()->getLoc();

    // Rule 1: no consume inside a repeated region that does not own it.
    for (const LinearUse &use : consumes)
      for (auto [ancestor, region] : use.chain)
        if (ancestor->hasTrait<cudaq::LinearRepeatedRegions>()) {
          report_("loop-capture-consume", value, use.owner->getLoc(),
                  std::nullopt, [&](raw_ostream &os) {
                    os << " defined outside " << ancestor->getName()
                       << " is consumed inside its body by "
                       << use.owner->getName()
                       << "; loop-carried linear state must be threaded as an "
                          "explicit init/carry";
                  });
          if (done())
            return;
          break;
        }

    // Rule 2: at most one dynamic consume on every path.
    for (auto [index, first] : llvm::enumerate(consumes))
      for (const LinearUse &second : ArrayRef(consumes).drop_front(index + 1))
        if (consumesConflict(first, second)) {
          auto [earlier, later] = programOrder(first, second);
          report_("double-consume", value, later->owner->getLoc(),
                  earlier->owner->getLoc(), [&](raw_ostream &os) {
                    os << " is consumed more than once: by "
                       << earlier->owner->getName() << " and again by "
                       << later->owner->getName();
                  });
          if (done())
            return;
        }

    // Rule 3: observation must precede consumption on the same path.
    for (const LinearUse &borrow : observations)
      for (const LinearUse &consume : consumes)
        if (borrowFollowsConsume(borrow, consume)) {
          report_("borrow-after-consume", value, borrow.owner->getLoc(),
                  consume.owner->getLoc(), [&](raw_ostream &os) {
                    os << " is observed by " << borrow.owner->getName()
                       << " after being consumed by "
                       << consume.owner->getName();
                  });
          if (done())
            return;
        }

    if (mayDrop)
      return;

    // Rule 4: every path consumes the value.
    if (consumes.empty()) {
      report_("unconsumed", value, definitionLoc, std::nullopt,
              [&](raw_ostream &os) {
                os << " is never consumed; every linear owner needs exactly "
                      "one consuming use (return/yield it, or dispose of it "
                      "explicitly)";
              });
      return;
    }
    llvm::MapVector<Operation *, llvm::SmallSet<unsigned, 4>> covered;
    for (const LinearUse &use : consumes)
      for (auto [ancestor, region] : use.chain)
        if (ancestor->hasTrait<cudaq::LinearExclusiveRegions>())
          covered[ancestor].insert(region);
    for (auto &[op, regions] : covered)
      if (regions.size() < nonEmptyRegionCount(op)) {
        report_("unconsumed", value, op->getLoc(), definitionLoc,
                [&](raw_ostream &os) {
                  os << " is consumed in some but not all regions of "
                     << op->getName() << "; it must be consumed on every path";
                });
        if (done())
          return;
      }
  }

  std::string describe(Value value) {
    if (!asmState)
      // assumeVerified: this runs inside the verifier, and AsmState would
      // otherwise verify (and recurse into) the scope again.
      asmState.emplace(scope,
                       OpPrintingFlags().useLocalScope().assumeVerified());
    std::string text;
    llvm::raw_string_ostream os(text);
    value.printAsOperand(os, *asmState);
    os << " : " << value.getType() << " (";
    if (auto argument = dyn_cast<BlockArgument>(value))
      os << "block argument " << argument.getArgNumber() << " of "
         << argument.getOwner()->getParentOp()->getName();
    else
      os << "result of " << value.getDefiningOp()->getName();
    os << ")";
    return text;
  }

  template <typename Detail>
  void report_(StringRef kind, Value value, Location location,
               std::optional<Location> related, Detail &&detail) {
    cudaq::opt::LinearOwnershipViolation violation{
        scope, kind.str(), describe(value), {}, location, related};
    llvm::raw_string_ostream os(violation.message);
    os << "linear value " << violation.value;
    detail(os);
    report.violations.push_back(std::move(violation));
  }

  Operation *scope;
  cudaq::opt::LinearOwnershipReport &report;
  bool firstOnly;
  std::optional<AsmState> asmState;
};

} // namespace

void cudaq::opt::analyzeLinearOwnershipScope(Operation *scope,
                                             LinearOwnershipReport &report,
                                             bool firstOnly) {
  report.checkedScopes.push_back(scope);
  ScopeAnalysis(scope, report, firstOnly).run();
}

bool cudaq::opt::isDeclarationOnlyScope(Operation *scope) {
  if (scope->getNumRegions() != 1)
    return false;
  Region &region = scope->getRegion(0);
  if (region.empty())
    return true;
  if (!region.hasOneBlock())
    return false;
  Block &block = region.front();
  if (!block.empty()) {
    if (!llvm::hasSingleElement(block))
      return false;
    Operation &terminator = block.front();
    if (terminator.getNumOperands() || terminator.getNumResults() ||
        terminator.getNumRegions())
      return false;
  }
  return llvm::all_of(block.getArguments(), [](BlockArgument argument) {
    return argument.use_empty();
  });
}

cudaq::opt::LinearOwnershipReport
cudaq::opt::analyzeLinearOwnership(Operation *root) {
  LinearOwnershipReport report;
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->hasTrait<cudaq::LinearOwnershipScope>() &&
        !isDeclarationOnlyScope(op))
      analyzeLinearOwnershipScope(op, report);
  });
  return report;
}

void cudaq::opt::emitLinearOwnershipViolation(
    const LinearOwnershipViolation &violation) {
  InFlightDiagnostic diagnostic = emitError(violation.location)
                                  << violation.message;
  if (violation.related)
    diagnostic.attachNote(*violation.related)
        << (violation.kind == "unconsumed" ? "value defined here"
                                           : "earlier consume here");
}

LogicalResult cudaq::opt::verifyLinearOwnership(Operation *scope) {
  LinearOwnershipReport report;
  analyzeLinearOwnershipScope(scope, report, /*firstOnly=*/true);
  if (report.ok())
    return success();
  emitLinearOwnershipViolation(report.violations.front());
  return failure();
}
