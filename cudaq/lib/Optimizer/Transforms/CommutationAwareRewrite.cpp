/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Transforms/CommutationAwareRewrite.h"
#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cassert>
#include <utility>

using namespace mlir;

namespace {

/// One cursor per virtual qubit in the anchor's support, each walking that
/// qubit's chain of operations. `value` is the wire standing for the qubit
/// where the cursor sits, and `next` is the operation it would reach next.
///
/// Each cursor follows one wire's def-use chain, which is already in program
/// order, so the search is a k-way merge over the frontier rather than a graph
/// traversal: take the earliest head, then advance past it. The frontier never
/// grows, because only the anchor's own qubits matter and a multi-qubit
/// operation joins chains instead of branching into new ones.
struct WireCursor {
  Value value;
  Operation *next = nullptr;
};

} // namespace

// The search assumes it can tell which qubits an operation touches by looking
// at its operands. That only holds for an operation whose quantum values are
// all scalar wires and that owns no nested code. A reusable control, reference,
// or aggregate operand falls outside the supported form, while a region or
// successor hides code that could reach anything. In each case the operands
// understate what the operation really acts on and the search stops there.
static bool hasBoundedQuantumSupport(Operation *operation) {
  if (operation->getNumRegions() != 0 || operation->getNumSuccessors() != 0)
    return false;
  auto isScalarOrClassical = [](Type type) {
    return !cudaq::quake::isQuantumType(type) ||
           isa<cudaq::quake::WireType>(type);
  };
  return llvm::all_of(operation->getOperandTypes(), isScalarOrClassical) &&
         llvm::all_of(operation->getResultTypes(), isScalarOrClassical);
}

// Order two operations of one block along the search direction: true when `lhs`
// is reached before `rhs`.
static bool comesFirst(Operation *lhs, Operation *rhs, bool isForward) {
  return isForward ? lhs->isBeforeInBlock(rhs) : rhs->isBeforeInBlock(lhs);
}

// Map a wire operand to the result carrying the same virtual qubit, or the
// reverse when `toResult` is false. These are the same one-to-one forms that
// `QubitIdentityAnalysis` propagates. Measurement's classical result is not
// part of its `getWires()` range.
static Value mapWireAcross(Operation *operation, Value value, bool toResult) {
  llvm::SmallVector<Value> wireInputs;
  ValueRange wireResults;
  if (auto op = dyn_cast<cudaq::quake::OperatorInterface>(operation)) {
    wireInputs = cudaq::quake::getWireOperands(op);
    wireResults = op.getWires();
  } else if (auto measurement =
                 dyn_cast<cudaq::quake::MeasurementInterface>(operation)) {
    for (Value target : measurement.getTargets())
      if (isa<cudaq::quake::WireType>(target.getType()))
        wireInputs.push_back(target);
    wireResults = measurement.getWires();
  } else if (auto reset = dyn_cast<cudaq::quake::ResetOp>(operation)) {
    if (isa<cudaq::quake::WireType>(reset.getTargets().getType()))
      wireInputs.push_back(reset.getTargets());
    wireResults = reset.getWires();
  } else {
    return {};
  }

  if (wireInputs.size() != wireResults.size())
    return {};
  for (auto [input, result] : llvm::zip(wireInputs, wireResults)) {
    if (toResult && input == value)
      return result;
    if (!toResult && result == value)
      return input;
  }
  return {};
}

// One use identifies the next operation on the virtual qubit. No use ends the
// chain. Multiple uses can occur at valid control-flow boundaries, where the
// block-local search cannot choose one path.
static Operation *getSoleWireUser(Value wire) {
  if (!wire.hasOneUse())
    return nullptr;
  return *wire.getUsers().begin();
}

// Open one cursor per virtual qubit in the anchor's support.
static llvm::SmallVector<WireCursor>
openFrontier(cudaq::quake::OperatorInterface anchor,
             cudaq::opt::CommutationSearchDirection direction) {
  llvm::SmallVector<WireCursor> frontier;

  if (direction == cudaq::opt::CommutationSearchDirection::Forward) {
    for (Value wire : anchor.getWires())
      frontier.push_back({wire, getSoleWireUser(wire)});
    return frontier;
  }

  for (Value wire : cudaq::quake::getWireOperands(anchor))
    frontier.push_back({wire, wire.getDefiningOp()});
  return frontier;
}

// Take the next operation off the frontier: the head closest to the anchor in
// the search direction. Returns null when every chain has ended, and also when
// a chain leaves the block, since a use nested in a region or reached through a
// block edge is beyond a block-local search. Both mean the same thing to the
// caller, so they share an answer.
static Operation *takeNext(llvm::ArrayRef<WireCursor> frontier, Block *block,
                           bool isForward) {
  Operation *nearest = nullptr;
  for (const WireCursor &cursor : frontier) {
    if (!cursor.next)
      continue;
    if (cursor.next->getBlock() != block)
      return nullptr;
    if (!nearest || comesFirst(cursor.next, nearest, isForward))
      nearest = cursor.next;
  }
  return nearest;
}

// Step past `candidate`. An operation on several of the anchor's qubits is the
// head of several chains at once, so every cursor pointing at it moves on
// together and the operation is visited once rather than once per qubit. A
// cursor whose qubit does not continue past `candidate`, such as one reaching a
// sink, a returned wire, or a block argument, simply ends.
static void
advanceFrontierPast(llvm::SmallVectorImpl<WireCursor> &frontier,
                    Operation *candidate,
                    cudaq::opt::CommutationSearchDirection direction) {
  bool isForward = direction == cudaq::opt::CommutationSearchDirection::Forward;
  for (WireCursor &cursor : frontier) {
    if (cursor.next != candidate)
      continue;
    Value stepped = mapWireAcross(candidate, cursor.value, isForward);
    cursor.value = stepped;
    cursor.next = stepped ? (isForward ? getSoleWireUser(stepped)
                                       : stepped.getDefiningOp())
                          : nullptr;
  }
}

class cudaq::opt::CommutationAwareRewriteMatcher::Impl {
public:
  // Build block analysis lazily. Reconstructing previously discarded state is
  // also accounted as a fallback rebuild.
  cudaq::quake::detail::CommutationAnalysis &getAnalysis(Block *block) {
    auto [entry, inserted] = analyses.try_emplace(block);
    if (inserted) {
      entry->second =
          std::make_unique<cudaq::quake::detail::CommutationAnalysis>(*block);
      ++statistics.analysisBuilds;
      if (invalidatedBlocks.erase(block))
        ++statistics.fallbackRebuilds;
    }
    return *entry->second;
  }

  // Drop cached state for a live block and mark its next lazy construction as a
  // fallback rebuild.
  void discardBlock(Block *block) {
    if (block && analyses.erase(block))
      invalidatedBlocks.insert(block);
  }

  // Drop state for an erased block without retaining future rebuild accounting.
  void forgetBlock(Block *block) {
    analyses.erase(block);
    invalidatedBlocks.erase(block);
  }

  llvm::DenseMap<Block *,
                 std::unique_ptr<cudaq::quake::detail::CommutationAnalysis>>
      analyses;
  llvm::DenseSet<Block *> invalidatedBlocks;
  cudaq::opt::CommutationAwareRewriteStatistics statistics;
};

namespace cudaq::opt::detail {
class CommutationAwareRewriteListener
    : public RewriterBase::ForwardingListener {
public:
  CommutationAwareRewriteListener(CommutationAwareRewriteMatcher &matcher,
                                  OpBuilder::Listener *listener)
      : RewriterBase::ForwardingListener(listener), matcher(matcher) {}

  void notifyOperationInserted(Operation *operation,
                               OpBuilder::InsertPoint previous) override {
    RewriterBase::ForwardingListener::notifyOperationInserted(operation,
                                                              previous);
    // A previous insertion point identifies a move. Invalidate cached state
    // for both the source and destination blocks.
    if (Block *previousBlock = previous.getBlock()) {
      discardBlock(previousBlock);
      if (operation->getBlock() != previousBlock)
        discardBlock(operation->getBlock());
      return;
    }

    // A new operation can be maintained incrementally only when its result
    // identities propagate unambiguously in the current block.
    auto analysis = matcher.impl->analyses.find(operation->getBlock());
    if (analysis != matcher.impl->analyses.end() &&
        !analysis->second->registerIdentityPreservingOperation(operation))
      discardBlock(operation->getBlock());
  }

  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override {
    RewriterBase::ForwardingListener::notifyBlockInserted(block, previous,
                                                          previousIt);
    discardBlock(block);
  }

  void notifyBlockErased(Block *block) override {
    RewriterBase::ForwardingListener::notifyBlockErased(block);
    matcher.impl->forgetBlock(block);
  }

  void notifyOperationModified(Operation *operation) override {
    RewriterBase::ForwardingListener::notifyOperationModified(operation);
    // A use rewired by a validated identity-preserving replacement changes the
    // operand value but not the virtual qubit it denotes, and every commutation
    // rule reads only identities, roles, polarity, and semantics. Such a user
    // therefore needs no invalidation; the notification is consumed only so it
    // is not mistaken for an unexplained in-place change.
    auto pending = pendingIdentityPreservingUsers.find(operation);
    if (pending != pendingIdentityPreservingUsers.end()) {
      assert(pending->second != 0 && "pending replacement count underflow");
      if (--pending->second == 0)
        pendingIdentityPreservingUsers.erase(pending);
      return;
    }
    // Any other in-place change may alter commutation semantics or identity
    // placement, so rebuild the affected block conservatively.
    discardBlock(operation->getBlock());
  }

  void notifyOperationReplaced(Operation *operation,
                               Operation *replacement) override {
    RewriterBase::ForwardingListener::notifyOperationReplaced(operation,
                                                              replacement);
    updateReplacement(operation, replacement->getResults());
  }

  void notifyOperationReplaced(Operation *operation,
                               ValueRange replacement) override {
    RewriterBase::ForwardingListener::notifyOperationReplaced(operation,
                                                              replacement);
    updateReplacement(operation, replacement);
  }

  void notifyOperationErased(Operation *operation) override {
    RewriterBase::ForwardingListener::notifyOperationErased(operation);
    pendingIdentityPreservingUsers.erase(operation);
    auto analysis = matcher.impl->analyses.find(operation->getBlock());
    if (analysis == matcher.impl->analyses.end())
      return;
    analysis->second->eraseOperation(operation);
  }

private:
  void updateReplacement(Operation *operation, ValueRange replacement) {
    // Replacement callbacks and their per-use modification callbacks are
    // synchronous. Starting another replacement or falling back must never
    // leave counts that could suppress a later genuine modification.
    pendingIdentityPreservingUsers.clear();
    auto analysis = matcher.impl->analyses.find(operation->getBlock());
    if (analysis == matcher.impl->analyses.end())
      return;

    // Quantum rewiring is incrementally maintainable only after identity
    // validation. A used classical result can feed an operator parameter or
    // Pauli value, so its user semantics may change even when every quantum
    // identity is unchanged.
    for (Value result : operation->getResults()) {
      if (!result.use_empty() &&
          !cudaq::quake::isQuantumType(result.getType())) {
        discardBlock(operation->getBlock());
        return;
      }
    }

    // A validated replacement preserves qubit identity state and invalidates
    // only pairs incident to the replaced endpoint. Failure requires block
    // fallback.
    if (!analysis->second->prepareIdentityPreservingReplacement(operation,
                                                                replacement)) {
      discardBlock(operation->getBlock());
      return;
    }

    // RewriterBase emits exactly one modification notification per replaced
    // use. Count quantum uses per owner so a multi-result replacement into one
    // user consumes every callback without suppressing anything afterward.
    for (Value result : operation->getResults())
      for (OpOperand &use : result.getUses())
        ++pendingIdentityPreservingUsers[use.getOwner()];
  }

  void discardBlock(Block *block) {
    pendingIdentityPreservingUsers.clear();
    matcher.impl->discardBlock(block);
  }

  CommutationAwareRewriteMatcher &matcher;
  llvm::DenseMap<Operation *, std::size_t> pendingIdentityPreservingUsers;
};
} // namespace cudaq::opt::detail

class cudaq::opt::CommutationAwareRewriteDriver::Impl {
public:
  Impl(MLIRContext &context, GreedyRewriteConfig config,
       std::unique_ptr<CommutationAwareRewriteMatcher> matcher)
      : matcher(std::move(matcher)), patterns(&context),
        config(std::move(config)),
        listener(*this->matcher, this->config.getListener()) {
    this->config.setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Disabled);
    this->config.setListener(&listener);
  }

  std::unique_ptr<CommutationAwareRewriteMatcher> matcher;
  RewritePatternSet patterns;
  GreedyRewriteConfig config;
  cudaq::opt::detail::CommutationAwareRewriteListener listener;
  bool hasRun = false;
};

cudaq::opt::CommutationAwareRewriteMatcher::CommutationAwareRewriteMatcher(
    std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

cudaq::opt::CommutationAwareRewriteMatcher::~CommutationAwareRewriteMatcher() =
    default;

std::optional<cudaq::opt::CommutationAwareRewriteMatch>
cudaq::opt::CommutationAwareRewriteMatcher::findNearest(
    Operation *anchor, cudaq::opt::CommutationSearchDirection direction,
    llvm::function_ref<bool(Operation *)> isEndpoint) {
  if (!anchor || !anchor->getBlock())
    return std::nullopt;
  auto anchorInterface = dyn_cast<cudaq::quake::OperatorInterface>(anchor);
  if (!anchorInterface)
    return std::nullopt;

  Block *block = anchor->getBlock();
  auto &analysis = impl->getAnalysis(block);
  // The anchor must be resolvable before any pair result involving it means
  // anything, and its own support must be bounded by its operands. A self-query
  // may cache one pair, but it preserves the existing public analysis boundary;
  // a separate cache-neutral support API needs measured justification.
  if (!hasBoundedQuantumSupport(anchor) || !analysis.canCommute(anchor, anchor))
    return std::nullopt;

  // Follow Quake's own wire dataflow rather than block order. Only
  // operations sharing a virtual qubit with the anchor are reachable this way.
  // Every operation skipped is disjoint from the anchor's support and therefore
  // commutes with it, so it needs neither a probe nor a cache entry.
  auto frontier = openFrontier(anchorInterface, direction);

  bool isForward = direction == cudaq::opt::CommutationSearchDirection::Forward;
  cudaq::opt::CommutationAwareRewriteMatch match;
  while (Operation *candidate = takeNext(frontier, block, isForward)) {
    ++impl->statistics.frontierCandidates;
    // Reference and aggregate quantum values, and nested code that could reach
    // further qubits, are outside the adopted semantics. Measurement and reset
    // are the only effects with supported scalar-wire flow; all other reached
    // operations retain the conservative self-query barrier.
    if (!hasBoundedQuantumSupport(candidate))
      return std::nullopt;
    bool isTraversalEffect =
        isa<cudaq::quake::MeasurementInterface, cudaq::quake::ResetOp>(
            candidate);
    auto candidateInterface =
        dyn_cast<cudaq::quake::OperatorInterface>(candidate);
    if (!isTraversalEffect && !analysis.canCommute(candidate, candidate))
      return std::nullopt;

    // Consumer policy decides compatibility first. An accepted endpoint is
    // where the anchor stops, so it is never crossed and needs no commutation
    // proof; the consumer owns the endpoint pair's algebraic identity.
    if (candidateInterface && isEndpoint(candidate)) {
      // A backward search collects in reverse, so restore block order.
      if (!isForward)
        std::reverse(match.crossed.begin(), match.crossed.end());
      match.endpoint = candidate;
      return match;
    }

    // A candidate that is crossed rather than accepted must be proven to
    // commute with the anchor.
    ++impl->statistics.commutationProbes;
    if (!analysis.canCommute(anchor, candidate))
      return std::nullopt;

    match.crossed.push_back(candidate);
    advanceFrontierPast(frontier, candidate, direction);
  }
  return std::nullopt;
}

bool cudaq::opt::CommutationAwareRewriteMatcher::haveSameOrderedQuantumOperands(
    Operation *lhs, Operation *rhs) {
  if (!lhs || !rhs || !lhs->getBlock() || lhs->getBlock() != rhs->getBlock())
    return false;
  return impl->getAnalysis(lhs->getBlock())
      .haveSameOrderedQuantumOperands(lhs, rhs);
}

cudaq::opt::CommutationAwareRewriteDriver::CommutationAwareRewriteDriver(
    MLIRContext &context, GreedyRewriteConfig config)
    : impl(std::make_unique<Impl>(
          context, std::move(config),
          std::unique_ptr<CommutationAwareRewriteMatcher>(
              new CommutationAwareRewriteMatcher(
                  std::make_unique<CommutationAwareRewriteMatcher::Impl>())))) {
}

cudaq::opt::CommutationAwareRewriteDriver::~CommutationAwareRewriteDriver() =
    default;

RewritePatternSet &cudaq::opt::CommutationAwareRewriteDriver::getPatterns() {
  return impl->patterns;
}

cudaq::opt::CommutationAwareRewriteMatcher &
cudaq::opt::CommutationAwareRewriteDriver::getMatcher() {
  return *impl->matcher;
}

cudaq::opt::CommutationAwareRewriteStatistics
cudaq::opt::CommutationAwareRewriteDriver::getStatistics() const {
  return impl->matcher->impl->statistics;
}

LogicalResult cudaq::opt::CommutationAwareRewriteDriver::run(Region &region) {
  if (impl->hasRun)
    return failure();
  impl->hasRun = true;
  return applyPatternsGreedily(region, std::move(impl->patterns), impl->config);
}
