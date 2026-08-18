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

// Map one lane through a validated scalar-wire flow.
static Value mapWireAcross(const cudaq::quake::detail::ScalarWireFlow &flow,
                           Value value) {
  for (auto [input, result] : llvm::zip(flow.inputs, flow.results))
    if (input == value)
      return result;
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

// Open one cursor per virtual qubit in the anchor's ordered support.
static llvm::SmallVector<WireCursor>
openFrontier(const cudaq::quake::detail::ScalarWireFlow &flow) {
  llvm::SmallVector<WireCursor> frontier;
  for (Value wire : flow.results)
    frontier.push_back({wire, getSoleWireUser(wire)});
  return frontier;
}

// Take the next operation off the frontier: the head closest to the anchor in
// block order. Every anchor wire must retain an unambiguous path in the block.
// Otherwise the frontier no longer represents the anchor's complete support,
// so the block-local search ends.
static Operation *takeNext(llvm::ArrayRef<WireCursor> frontier, Block *block) {
  Operation *nearest = nullptr;
  for (const WireCursor &cursor : frontier) {
    if (!cursor.next || cursor.next->getBlock() != block)
      return nullptr;
    if (!nearest || cursor.next->isBeforeInBlock(nearest))
      nearest = cursor.next;
  }
  return nearest;
}

// `QubitIdentityAnalysis` identifies logical qubits, not SSA paths. For
// example, an endpoint can consume a second `borrow_wire` for the same wire-set
// slot while the cursor from the anchor's result reaches another operation.
// The identities match, but the complete frontier reaches the endpoint only
// when every cursor points to it.
static bool doesCompleteFrontierReach(llvm::ArrayRef<WireCursor> frontier,
                                      Operation *candidate) {
  return llvm::all_of(frontier, [candidate](const WireCursor &cursor) {
    return cursor.next == candidate;
  });
}

static bool hasSameOrderedWireTypes(ValueRange lhs, ValueRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
    auto [lhsValue, rhsValue] = pair;
    return lhsValue.getType() == rhsValue.getType() &&
           isa<cudaq::quake::WireType>(lhsValue.getType());
  });
}

enum class DirectWireThreading { NotDirect, Exact, Mismatch };

// A unary operation cannot repeat a qubit in another operand role. Multi-wire
// operations need identity normalization to establish that precondition.
static bool
requiresDistinctQubitProof(const cudaq::quake::detail::ScalarWireFlow &flow) {
  return flow.inputs.size() > 1;
}

// With no crossed operation, exact result-to-operand threading proves the
// endpoint pair's ordered SSA correspondence. The caller still owns the
// endpoint algebra, and identity analysis still validates operand uniqueness
// for multi-wire operations.
//
// A direct pair that permutes values or changes control and target roles is a
// definitive mismatch. Falling back to logical identity matching could hide
// that mismatch by resolving different SSA values to the same qubit.
static DirectWireThreading classifyDirectWireThreading(Operation *lhs,
                                                       Operation *rhs) {
  auto lhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(lhs);
  auto rhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(rhs);
  auto lhsFlow = cudaq::quake::detail::getScalarWireFlow(lhs);
  auto rhsFlow = cudaq::quake::detail::getScalarWireFlow(rhs);
  if (!lhsInterface || !rhsInterface || !lhsFlow || !rhsFlow)
    return DirectWireThreading::NotDirect;

  bool lhsIsProducer = lhs->isBeforeInBlock(rhs);
  const auto &producerResults =
      lhsIsProducer ? lhsFlow->results : rhsFlow->results;
  const auto &consumerOperands =
      lhsIsProducer ? rhsFlow->inputs : lhsFlow->inputs;
  Operation *consumer = lhsIsProducer ? rhs : lhs;
  if (!hasSameOrderedWireTypes(producerResults, consumerOperands))
    return DirectWireThreading::NotDirect;

  if (!llvm::all_of(producerResults, [consumer](Value result) {
        return result.hasOneUse() && result.use_begin()->getOwner() == consumer;
      }))
    return DirectWireThreading::NotDirect;

  bool hasSameRoles = hasSameOrderedWireTypes(lhsInterface.getControls(),
                                              rhsInterface.getControls()) &&
                      hasSameOrderedWireTypes(lhsInterface.getTargets(),
                                              rhsInterface.getTargets());
  bool hasSameValues =
      llvm::all_of(llvm::zip(producerResults, consumerOperands), [](auto pair) {
        auto [producer, consumer] = pair;
        return producer == consumer;
      });
  if (hasSameValues)
    return hasSameRoles ? DirectWireThreading::Exact
                        : DirectWireThreading::Mismatch;

  llvm::DenseSet<Value> producerValues(producerResults.begin(),
                                       producerResults.end());
  if (llvm::all_of(consumerOperands, [&](Value operand) {
        return producerValues.contains(operand);
      }))
    return DirectWireThreading::Mismatch;
  return DirectWireThreading::NotDirect;
}

// Step past `candidate`. An operation on several of the anchor's qubits is the
// head of several chains at once, so every cursor pointing at it moves on
// together and the operation is visited once rather than once per qubit. A
// cursor whose qubit does not continue past `candidate`, such as one reaching a
// sink, a returned wire, or a block argument, simply ends.
static void
advanceFrontierPast(llvm::SmallVectorImpl<WireCursor> &frontier,
                    Operation *candidate,
                    const cudaq::quake::detail::ScalarWireFlow &flow) {
  for (WireCursor &cursor : frontier) {
    if (cursor.next != candidate)
      continue;
    Value stepped = mapWireAcross(flow, cursor.value);
    cursor.value = stepped;
    cursor.next = stepped ? getSoleWireUser(stepped) : nullptr;
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

    // A validated replacement preserves qubit identity state and clears cached
    // relations. Failure requires block fallback.
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

Operation *cudaq::opt::CommutationAwareRewriteMatcher::findNearest(
    Operation *anchor, llvm::function_ref<bool(Operation *)> isEndpoint) {
  if (!anchor || !anchor->getBlock())
    return nullptr;
  auto anchorInterface = dyn_cast<cudaq::quake::OperatorInterface>(anchor);
  auto anchorFlow = cudaq::quake::detail::getScalarWireFlow(anchor);
  if (!anchorInterface || !anchorFlow)
    return nullptr;

  Block *block = anchor->getBlock();
  // Follow Quake's own wire dataflow rather than block order. Only
  // operations sharing a virtual qubit with the anchor are reachable this way.
  // Every operation skipped is disjoint from the anchor's support and therefore
  // commutes with it, so it needs neither a probe nor a cache entry.
  auto frontier = openFrontier(*anchorFlow);

  cudaq::quake::detail::CommutationAnalysis *analysis = nullptr;
  auto requireAnalysis = [&]() -> cudaq::quake::detail::CommutationAnalysis & {
    if (!analysis)
      analysis = &impl->getAnalysis(block);
    return *analysis;
  };
  // A self-query builds the normalized operation view and rejects a logical
  // qubit used in more than one role. Unary operations cannot violate that
  // constraint and retain the analysis-free adjacent path.
  auto hasDistinctQubits =
      [&](Operation *operation,
          const cudaq::quake::detail::ScalarWireFlow &flow) {
        return !requiresDistinctQubitProof(flow) ||
               requireAnalysis().canCommute(operation, operation);
      };
  while (Operation *candidate = takeNext(frontier, block)) {
    auto candidateFlow = cudaq::quake::detail::getScalarWireFlow(candidate);
    if (!candidateFlow)
      return nullptr;
    bool isTraversableMeasurementOrReset =
        isa<cudaq::quake::MeasurementInterface, cudaq::quake::ResetOp>(
            candidate);
    auto candidateInterface =
        dyn_cast<cudaq::quake::OperatorInterface>(candidate);

    // Consumer policy decides endpoint compatibility. Every operation crossed
    // before an accepted endpoint requires the commutation proof below.
    if (candidateInterface && (!hasDistinctQubits(anchor, *anchorFlow) ||
                               !hasDistinctQubits(candidate, *candidateFlow)))
      return nullptr;
    if (candidateInterface && isEndpoint(candidate)) {
      if (!doesCompleteFrontierReach(frontier, candidate))
        return nullptr;
      return candidate;
    }

    // A candidate that is crossed rather than accepted must be proven to
    // commute with the anchor.
    auto &blockAnalysis = requireAnalysis();
    // The anchor and candidate must be resolvable before any pair result
    // involving them means anything. Measurement instruments and reset
    // channels are traversable through their scalar-wire flow, but other
    // candidates keep the conservative self-query barrier.
    if (!blockAnalysis.canCommute(anchor, anchor) ||
        (!isTraversableMeasurementOrReset &&
         !blockAnalysis.canCommute(candidate, candidate)))
      return nullptr;
    if (!blockAnalysis.canCommute(anchor, candidate))
      return nullptr;

    advanceFrontierPast(frontier, candidate, *candidateFlow);
  }
  return nullptr;
}

bool cudaq::opt::CommutationAwareRewriteMatcher::haveSameOrderedQuantumOperands(
    Operation *lhs, Operation *rhs) {
  if (!lhs || !rhs || !lhs->getBlock() || lhs->getBlock() != rhs->getBlock())
    return false;
  auto directThreading = classifyDirectWireThreading(lhs, rhs);
  if (directThreading == DirectWireThreading::Exact) {
    // Exact threading proves ordered operands for unary endpoints. Multi-wire
    // endpoints still need normalized views that reject duplicate qubit roles.
    auto lhsFlow = cudaq::quake::detail::getScalarWireFlow(lhs);
    auto rhsFlow = cudaq::quake::detail::getScalarWireFlow(rhs);
    assert(lhsFlow && rhsFlow && "exact threading requires scalar-wire flow");
    if (!requiresDistinctQubitProof(*lhsFlow) &&
        !requiresDistinctQubitProof(*rhsFlow))
      return true;
    auto &analysis = impl->getAnalysis(lhs->getBlock());
    return analysis.canCommute(lhs, lhs) && analysis.canCommute(rhs, rhs);
  }
  if (directThreading == DirectWireThreading::Mismatch)
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
