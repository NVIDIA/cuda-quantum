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

/// One cursor per frontier lane in the anchor's ordered scalar-wire support.
/// `value` is the result consumed by `downstream`, and `previous` is its
/// defining operation.
///
/// Each cursor follows one scalar-wire use-def chain. The frontier never grows
/// because only the anchor's input lanes participate.
struct WireCursor {
  Value value;
  Operation *previous = nullptr;
  Operation *downstream = nullptr;
};

} // namespace

static Value mapWireBackward(const cudaq::quake::detail::ScalarWireFlow &flow,
                             Value value) {
  for (auto [input, result] : llvm::zip(flow.inputs, flow.results))
    if (result == value)
      return input;
  return {};
}

// Open one cursor per frontier lane in the anchor's ordered support.
static llvm::SmallVector<WireCursor>
openFrontier(Operation *anchor,
             const cudaq::quake::detail::ScalarWireFlow &flow) {
  llvm::SmallVector<WireCursor> frontier;
  for (Value wire : flow.inputs)
    frontier.push_back({wire, wire.getDefiningOp(), anchor});
  return frontier;
}

// Validate the backward frontier and collect its defining operations. The
// unique-use check ties every producer result to the downstream operation
// expected by its cursor; a unique defining operation alone would not reject
// branched flow.
static bool collectFrontierHeads(llvm::ArrayRef<WireCursor> frontier,
                                 Block *block,
                                 llvm::DenseSet<Operation *> &heads) {
  heads.clear();
  if (frontier.empty())
    return false;
  for (const WireCursor &cursor : frontier) {
    if (!cursor.previous || cursor.previous->getBlock() != block ||
        !cursor.value.hasOneUse() ||
        cursor.value.use_begin()->getOwner() != cursor.downstream)
      return false;
    heads.insert(cursor.previous);
  }
  return true;
}

// `QubitIdentityAnalysis` identifies logical qubits, not SSA paths. For
// example, an endpoint can consume a second `borrow_wire` for the same wire-set
// slot while the frontier lane from the anchor reaches another defining
// operation.
// The identities match, but the complete frontier reaches the endpoint only
// when every lane points to it.
static bool doesCompleteFrontierReach(llvm::ArrayRef<WireCursor> frontier,
                                      Operation *candidate) {
  return llvm::all_of(frontier, [candidate](const WireCursor &cursor) {
    return cursor.previous == candidate;
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
// for multi-wire operations. Direct def-use determines producer orientation
// without consulting mutable block-order metadata.
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

  auto feeds = [](const cudaq::quake::detail::ScalarWireFlow &producerFlow,
                  const cudaq::quake::detail::ScalarWireFlow &consumerFlow,
                  Operation *consumer) {
    return hasSameOrderedWireTypes(producerFlow.results, consumerFlow.inputs) &&
           llvm::all_of(producerFlow.results, [consumer](Value result) {
             return result.hasOneUse() &&
                    result.use_begin()->getOwner() == consumer;
           });
  };
  bool lhsIsProducer = feeds(*lhsFlow, *rhsFlow, rhs);
  if (!lhsIsProducer && !feeds(*rhsFlow, *lhsFlow, lhs))
    return DirectWireThreading::NotDirect;
  const auto &producerResults =
      lhsIsProducer ? lhsFlow->results : rhsFlow->results;
  const auto &consumerOperands =
      lhsIsProducer ? rhsFlow->inputs : lhsFlow->inputs;

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

// An operation shared by several frontier lanes advances those lanes together
// and is visited once.
static void
stepFrontierBackward(llvm::SmallVectorImpl<WireCursor> &frontier,
                     Operation *candidate,
                     const cudaq::quake::detail::ScalarWireFlow &flow) {
  for (WireCursor &cursor : frontier) {
    if (cursor.previous != candidate)
      continue;
    Value stepped = mapWireBackward(flow, cursor.value);
    cursor.value = stepped;
    cursor.downstream = candidate;
    cursor.previous = stepped ? stepped.getDefiningOp() : nullptr;
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
    // updateReplacement already validated quantum identities and cleared
    // relation caches in affected analyzed blocks. Consume each expected
    // callback without further invalidation so it is not mistaken for an
    // unexplained in-place change.
    auto pending = pendingIdentityPreservingUsers.find(operation);
    if (pending != pendingIdentityPreservingUsers.end()) {
      assert(pending->second != 0 && "pending replacement count underflow");
      if (--pending->second == 0)
        pendingIdentityPreservingUsers.erase(pending);
      return;
    }
    // An unexplained in-place change invalidates the operation's block and
    // analyzed blocks that depend on its results. Unrelated block analyses do
    // not observe the changed operation and remain valid.
    discardDependentBlocks(operation);
  }

  void notifyOperationReplaced(Operation *operation,
                               Operation *replacement) override {
    RewriterBase::ForwardingListener::notifyOperationReplaced(operation,
                                                              replacement);
    updateReplacement(operation, replacement->getResults(), replacement);
  }

  void notifyOperationReplaced(Operation *operation,
                               ValueRange replacement) override {
    RewriterBase::ForwardingListener::notifyOperationReplaced(operation,
                                                              replacement);
    Operation *replacementOp = nullptr;
    if (!replacement.empty()) {
      Operation *definition = replacement.front().getDefiningOp();
      if (definition && llvm::equal(definition->getResults(), replacement))
        replacementOp = definition;
    }
    updateReplacement(operation, replacement, replacementOp);
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
  void discardDependentBlocks(Operation *operation) {
    pendingIdentityPreservingUsers.clear();
    llvm::SmallVector<Operation *> worklist{operation};
    llvm::DenseSet<Operation *> visited;
    llvm::DenseSet<Value> visitedValues;
    llvm::DenseSet<Block *> affectedBlocks;
    auto markBlock = [&](Block *block) {
      if (block && matcher.impl->analyses.contains(block))
        affectedBlocks.insert(block);
    };

    while (!worklist.empty()) {
      Operation *dependent = worklist.pop_back_val();
      if (!visited.insert(dependent).second)
        continue;
      markBlock(dependent->getBlock());

      // A region owner or branch can carry a changed value into blocks without
      // preserving an SSA result-to-use edge through their block arguments.
      dependent->walk(
          [&](Operation *nested) { markBlock(nested->getBlock()); });
      for (Block *successor : dependent->getSuccessors()) {
        markBlock(successor);
        for (BlockArgument argument : successor->getArguments())
          if (visitedValues.insert(argument).second)
            for (Operation *user : argument.getUsers())
              worklist.push_back(user);
      }
      for (Value result : dependent->getResults())
        if (visitedValues.insert(result).second)
          for (Operation *user : result.getUsers())
            worklist.push_back(user);
    }

    for (Block *block : affectedBlocks)
      discardBlock(block);
  }

  void updateReplacement(Operation *operation, ValueRange replacement,
                         Operation *replacementOp) {
    // Replacement callbacks and their per-use modification callbacks are
    // synchronous. Starting another replacement or falling back must never
    // leave counts that could suppress a later genuine modification.
    pendingIdentityPreservingUsers.clear();
    auto analysis = matcher.impl->analyses.find(operation->getBlock());
    if (analysis == matcher.impl->analyses.end())
      return;

    // A valid replacement preserves qubit identities, clears cached relations,
    // and either updates the old index position or discards the index.
    // Modification notifications cover every rewired quantum or classical use.
    // Failure requires block fallback.
    if (!analysis->second->prepareIdentityPreservingReplacement(
            operation, replacement, replacementOp)) {
      discardBlock(operation->getBlock());
      return;
    }

    // RewriterBase emits exactly one modification notification per replaced
    // use. Count result uses per owner so a multi-result replacement into one
    // user consumes every callback without suppressing anything afterward.
    // A user in another analyzed block keeps its proved qubit identities, but
    // relations involving its changed classical or quantum operand must be
    // recomputed before its expected notification is consumed.
    for (Value result : operation->getResults()) {
      for (OpOperand &use : result.getUses()) {
        Operation *owner = use.getOwner();
        Block *ownerBlock = owner->getBlock();
        if (!ownerBlock)
          continue;
        auto ownerAnalysis = matcher.impl->analyses.find(ownerBlock);
        if (ownerAnalysis != matcher.impl->analyses.end())
          ownerAnalysis->second->clearCachedRelations();
        ++pendingIdentityPreservingUsers[owner];
      }
    }
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

Operation *cudaq::opt::CommutationAwareRewriteMatcher::find_nearest(
    Operation *anchor, llvm::function_ref<bool(Operation *)> isEndpoint) {
  if (!anchor || !anchor->getBlock())
    return nullptr;
  auto anchorInterface = dyn_cast<cudaq::quake::OperatorInterface>(anchor);
  auto anchorFlow = cudaq::quake::detail::getScalarWireFlow(anchor);
  if (!anchorInterface || !anchorFlow)
    return nullptr;

  Block *block = anchor->getBlock();
  auto frontier = openFrontier(anchor, *anchorFlow);

  cudaq::quake::detail::CommutationAnalysis *analysis = nullptr;
  auto requireAnalysis = [&]() -> cudaq::quake::detail::CommutationAnalysis & {
    if (!analysis)
      analysis = &impl->getAnalysis(block);
    return *analysis;
  };
  auto canCrossScope = [&](Operation *operation) {
    llvm::SmallVector<Value> captures;
    return cudaq::quake::detail::CommutationAnalysis::collectScopeWireCaptures(
               operation, captures) &&
           requireAnalysis().hasDisjointQuantumSupport(anchor, captures);
  };
  auto canCrossIdentityBoundary = [&](Operation *operation) {
    if (isa<cudaq::quake::AllocaOp, cudaq::quake::NullWireOp>(operation))
      return true;

    Value wire =
        cudaq::quake::detail::CommutationAnalysis::getIdentityBoundaryWire(
            operation);
    if (!wire)
      return false;
    llvm::SmallVector<Value, 1> wires{wire};
    return requireAnalysis().hasDisjointQuantumSupport(anchor, wires);
  };
  llvm::DenseSet<Operation *> frontierHeads;
  if (!collectFrontierHeads(frontier, block, frontierHeads))
    return nullptr;

  Operation *match = nullptr;
  auto processCandidate = [&](Operation *candidate) {
    bool isFrontierHead = frontierHeads.contains(candidate);
    if (!isFrontierHead && cudaq::quake::detail::CommutationAnalysis::
                               isIgnorableNonQuantumOperation(candidate))
      return true;
    if (!isFrontierHead && candidate->getNumRegions() != 0) {
      return canCrossScope(candidate);
    }

    auto candidateFlow = cudaq::quake::detail::getScalarWireFlow(candidate);
    if (!isFrontierHead && !candidateFlow) {
      if (canCrossIdentityBoundary(candidate) ||
          requireAnalysis().canCommute(anchor, candidate))
        return true;
      return false;
    }
    if (!candidateFlow)
      return false;
    auto candidateInterface =
        dyn_cast<cudaq::quake::OperatorInterface>(candidate);

    // Consumer policy owns the endpoint algebra.
    if (isFrontierHead && candidateInterface &&
        (!has_distinct_quantum_operands(anchor) ||
         !has_distinct_quantum_operands(candidate)))
      return false;
    if (isFrontierHead && candidateInterface && isEndpoint(candidate)) {
      if (!doesCompleteFrontierReach(frontier, candidate))
        return false;
      match = candidate;
      return false;
    }

    // Every frontier head the consumer declines and every other crossed
    // scalar-wire operation requires a commutation proof.
    auto &blockAnalysis = requireAnalysis();
    if (!isFrontierHead &&
        blockAnalysis.hasDisjointQuantumSupport(anchor, candidateFlow->inputs))
      return true;
    if (!blockAnalysis.canCommute(anchor, candidate))
      return false;

    if (isFrontierHead) {
      stepFrontierBackward(frontier, candidate, *candidateFlow);
      if (!collectFrontierHeads(frontier, block, frontierHeads))
        return false;
    }
    return true;
  };

  // Check nearby frontier heads and ignorable operations before building the
  // index. Once analysis is needed, indexed traversal skips operations on
  // unrelated known qubits when it is available.
  bool useBlockOrderScan = false;
  for (Operation *candidate = anchor->getPrevNode(); candidate;) {
    if (!processCandidate(candidate))
      return match;

    Operation *inclusiveUpperBound = candidate->getPrevNode();
    if (!inclusiveUpperBound)
      return nullptr;
    if (analysis && !useBlockOrderScan) {
      bool searchFinished = analysis->tryWalkPriorOperations(
          anchor, inclusiveUpperBound, processCandidate);
      if (searchFinished)
        return match;
      useBlockOrderScan = true;
    }
    candidate = inclusiveUpperBound;
  }
  return nullptr;
}

bool cudaq::opt::CommutationAwareRewriteMatcher::has_distinct_quantum_operands(
    Operation *operation) {
  if (!operation || !operation->getBlock() ||
      !isa<cudaq::quake::OperatorInterface>(operation))
    return false;
  auto flow = cudaq::quake::detail::getScalarWireFlow(operation);
  if (!flow)
    return false;
  return !requiresDistinctQubitProof(*flow) ||
         impl->getAnalysis(operation->getBlock())
             .hasDistinctQuantumOperands(operation);
}

bool cudaq::opt::CommutationAwareRewriteMatcher::
    have_same_ordered_quantum_operands(Operation *lhs, Operation *rhs) {
  if (!lhs || !rhs || !lhs->getBlock() || lhs->getBlock() != rhs->getBlock())
    return false;
  auto directThreading = classifyDirectWireThreading(lhs, rhs);
  if (directThreading == DirectWireThreading::Exact) {
    // Exact threading proves ordered roles; distinct logical operands remain a
    // separate precondition of the pair algebra.
    return has_distinct_quantum_operands(lhs) &&
           has_distinct_quantum_operands(rhs);
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

RewritePatternSet &cudaq::opt::CommutationAwareRewriteDriver::get_patterns() {
  return impl->patterns;
}

cudaq::opt::CommutationAwareRewriteMatcher &
cudaq::opt::CommutationAwareRewriteDriver::get_matcher() {
  return *impl->matcher;
}

cudaq::opt::CommutationAwareRewriteStatistics
cudaq::opt::CommutationAwareRewriteDriver::get_statistics() const {
  return impl->matcher->impl->statistics;
}

LogicalResult cudaq::opt::CommutationAwareRewriteDriver::run(Region &region) {
  if (impl->hasRun)
    return failure();
  impl->hasRun = true;
  return applyPatternsGreedily(region, std::move(impl->patterns), impl->config);
}
