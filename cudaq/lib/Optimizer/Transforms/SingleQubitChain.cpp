/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "SingleQubitChain.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include <optional>
#include <utility>

using namespace mlir;

namespace {

struct WireStep {
  Operation *operation;
  std::optional<cudaq::opt::SingleQubitChainScopeStep> scopeStep;
};

struct WirePathEnd {
  Operation *operation;
  Value wire;
};

} // namespace

// Returns whether `nested` is inside `outer` through only single-block
// `cc.scope` operations. Any other enclosing region prevents traversal.
static bool entersSingleBlockLexicalScopesOnly(Block *nested, Block *outer) {
  while (nested != outer) {
    if (!nested)
      return false;
    auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(nested->getParentOp());
    if (!scope || scope.getAtomicQuantumRegionAttr() ||
        !scope.getInitRegion().hasOneBlock())
      return false;
    nested = scope->getBlock();
  }
  return true;
}

/// Return whether an operation can be followed as a direct wire step.
/// Calls, region operations, and terminators require control-flow semantics
/// that this traversal deliberately does not model.
static bool isDirectWireStep(Operation *operation) {
  return !isa<CallOpInterface>(operation) && operation->getNumRegions() == 0 &&
         !operation->hasTrait<OpTrait::IsTerminator>();
}

// Follow the unique use of a wire forward. A direct use reaches its user;
// a `cc.continue` use reaches the matching result of its enclosing scope.
static std::optional<WireStep> followWire(Value wire) {
  if (!isa<cudaq::quake::WireType>(wire.getType()) || !wire.hasOneUse())
    return std::nullopt;

  OpOperand *use = &*wire.getUses().begin();
  Operation *user = use->getOwner();
  if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(user)) {
    auto scope = dyn_cast<cudaq::cc::ScopeOp>(cont->getParentOp());
    if (!scope || scope.getAtomicQuantumRegionAttr() ||
        !scope.getInitRegion().hasOneBlock() ||
        scope.getInitRegion().front().getTerminator() != user ||
        cont.getNumOperands() != scope->getNumResults())
      return std::nullopt;
    unsigned index = use->getOperandNumber();
    if (index >= scope->getNumResults() ||
        !isa<cudaq::quake::WireType>(scope->getResult(index).getType()))
      return std::nullopt;
    Value result = scope->getResult(index);
    if (!result.hasOneUse())
      return std::nullopt;
    return WireStep{scope, cudaq::opt::SingleQubitChainScopeStep{result, use}};
  }
  if (!isDirectWireStep(user) || !entersSingleBlockLexicalScopesOnly(
                                     user->getBlock(), wire.getParentBlock()))
    return std::nullopt;
  return WireStep{user, std::nullopt};
}

// Follow one tuple lane through transparent scopes to its next direct user.
static std::optional<WirePathEnd> traceWire(
    Value wire,
    llvm::SmallVectorImpl<cudaq::opt::SingleQubitChainScopeStep> &scopeSteps) {
  auto step = followWire(wire);
  while (step && step->scopeStep) {
    scopeSteps.push_back(*step->scopeStep);
    wire = step->scopeStep->wire;
    step = followWire(wire);
  }
  return step ? std::optional<WirePathEnd>{WirePathEnd{step->operation, wire}}
              : std::nullopt;
}

// Return the wire flow of a gate the caller accepts as a chain member.
static std::optional<cudaq::quake::detail::WireFlow>
getChainFlow(Operation *operation,
             cudaq::opt::SingleQubitChainPredicate isSupported) {
  if (!isSupported(operation))
    return std::nullopt;
  return cudaq::quake::detail::getWireFlow(operation);
}

// A controlled chain continues only when every output lane reaches the same
// gate at its corresponding input position with the same ordered predicate.
// `canContinue` decides whether the next gate's predicate matches the head's.
// Returns the next gate and replaces `flow` with its wire flow, or returns null
// and leaves `flow` unchanged.
static Operation *
matchNextOperation(cudaq::quake::detail::WireFlow &flow,
                   Operation *firstOperation,
                   llvm::MutableArrayRef<
                       llvm::SmallVector<cudaq::opt::SingleQubitChainScopeStep>>
                       scopeSteps,
                   cudaq::opt::SingleQubitChainPredicate isSupported,
                   cudaq::opt::SingleQubitChainContinuation canContinue) {
  llvm::SmallVector<std::optional<WirePathEnd>> pathEnds;
  pathEnds.reserve(flow.results.size());
  for (auto [output, steps] : llvm::zip(flow.results, scopeSteps))
    pathEnds.push_back(traceWire(output, steps));

  if (llvm::any_of(pathEnds, [](const auto &path) { return !path; }))
    return nullptr;

  Operation *nextOperation = pathEnds.front()->operation;
  if (llvm::any_of(pathEnds, [&](const auto &path) {
        return path->operation != nextOperation;
      }))
    return nullptr;

  auto next = getChainFlow(nextOperation, isSupported);
  if (!next || next->inputs.size() != flow.results.size() ||
      !canContinue(firstOperation, nextOperation))
    return nullptr;

  for (auto [index, path] : llvm::enumerate(pathEnds))
    if (next->inputs[index] != path->wire)
      return nullptr;
  flow = std::move(*next);
  return nextOperation;
}

llvm::SmallVector<cudaq::opt::SingleQubitChain, 0>
cudaq::opt::collectSingleQubitChains(Block &block,
                                     SingleQubitChainPredicate isSupported,
                                     SingleQubitChainContinuation canContinue) {
  llvm::SmallVector<SingleQubitChain, 0> chains;
  llvm::SmallDenseSet<Operation *> collected;

  for (Operation &operation : block) {
    if (collected.contains(&operation))
      continue;

    auto flow = getChainFlow(&operation, isSupported);
    if (!flow || llvm::any_of(flow->inputs,
                              [](Value input) { return !input.hasOneUse(); }))
      continue;

    SingleQubitChain chain;
    chain.inputs = flow->inputs;
    chain.scopeSteps.resize(flow->inputs.size());
    for (Operation *current = &operation; current;
         current =
             matchNextOperation(*flow, chain.operations.front(),
                                chain.scopeSteps, isSupported, canContinue)) {
      chain.operations.push_back(current);
      chain.outputs = flow->results;
      collected.insert(current);
      if (llvm::any_of(flow->results,
                       [](Value output) { return !output.hasOneUse(); }))
        break;
    }
    chains.push_back(std::move(chain));
  }
  return chains;
}

void cudaq::opt::replaceSingleQubitChain(SingleQubitChain &&chain,
                                         ValueRange replacements) {
  assert(replacements.size() == chain.outputs.size());
  for (auto [replacementValue, original, steps] :
       llvm::zip(replacements, chain.outputs, chain.scopeSteps)) {
    Value replacement = replacementValue;
    for (SingleQubitChainScopeStep &scopeStep : steps) {
      scopeStep.continueOperand->set(replacement);
      replacement = scopeStep.wire;
    }
    original.replaceAllUsesWith(replacement);
  }
  for (Operation *operation : llvm::reverse(chain.operations))
    operation->erase();
}
