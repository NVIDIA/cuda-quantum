/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "ScalarWireChain.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include <optional>

using namespace mlir;

namespace {

struct ScalarWireStep {
  Operation *operation;
  std::optional<cudaq::opt::ScalarWireChainScopeStep> scopeStep;
};

struct WirePathEnd {
  Operation *operation;
  Value wire;
};

struct ChainWireOp {
  Operation *operation;
  llvm::SmallVector<Value> inputs;
  llvm::SmallVector<Value> outputs;
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

/// Return whether an operation can be followed as a direct scalar-wire step.
/// Calls, region operations, and terminators require control-flow semantics
/// that this traversal deliberately does not model.
static bool isDirectScalarWireStep(Operation *operation) {
  return !isa<CallOpInterface>(operation) && operation->getNumRegions() == 0 &&
         !operation->hasTrait<OpTrait::IsTerminator>();
}

// Follow the unique scalar-wire use forward. A direct use reaches its user;
// a `cc.continue` use reaches the matching result of its enclosing scope.
static std::optional<ScalarWireStep> traverseScalarWire(Value wire) {
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
    return ScalarWireStep{scope,
                          cudaq::opt::ScalarWireChainScopeStep{result, use}};
  }
  if (!isDirectScalarWireStep(user) ||
      !entersSingleBlockLexicalScopesOnly(user->getBlock(),
                                          wire.getParentBlock()))
    return std::nullopt;
  return ScalarWireStep{user, std::nullopt};
}

// Follow one tuple lane through transparent scopes to its next direct user.
static std::optional<WirePathEnd> traceWire(
    Value wire,
    llvm::SmallVectorImpl<cudaq::opt::ScalarWireChainScopeStep> &scopeSteps) {
  auto step = traverseScalarWire(wire);
  while (step && step->scopeStep) {
    scopeSteps.push_back(*step->scopeStep);
    wire = step->scopeStep->wire;
    step = traverseScalarWire(wire);
  }
  return step ? std::optional<WirePathEnd>{WirePathEnd{step->operation, wire}}
              : std::nullopt;
}

static std::optional<ChainWireOp>
getChainWireOp(Operation *operation,
               cudaq::opt::ScalarWireChainPredicate isSupported) {
  if (!isSupported(operation))
    return std::nullopt;
  auto flow = cudaq::quake::detail::getScalarWireFlow(operation);
  if (!flow)
    return std::nullopt;
  return ChainWireOp{operation, std::move(flow->inputs),
                     std::move(flow->results)};
}

// A controlled chain continues only when every output lane reaches the same
// gate at its corresponding input position with the same ordered predicate.
// `canContinue` decides whether the next gate's predicate matches the head's.
static std::optional<ChainWireOp>
matchNextOperation(const ChainWireOp &current, Operation *firstOperation,
                   llvm::MutableArrayRef<
                       llvm::SmallVector<cudaq::opt::ScalarWireChainScopeStep>>
                       scopeSteps,
                   cudaq::opt::ScalarWireChainPredicate isSupported,
                   cudaq::opt::ScalarWireChainContinuation canContinue) {
  llvm::SmallVector<std::optional<WirePathEnd>> pathEnds;
  pathEnds.reserve(current.outputs.size());
  for (auto [output, steps] : llvm::zip(current.outputs, scopeSteps))
    pathEnds.push_back(traceWire(output, steps));

  if (llvm::any_of(pathEnds, [](const auto &path) { return !path; }))
    return std::nullopt;

  Operation *nextOperation = pathEnds.front()->operation;
  if (llvm::any_of(pathEnds, [&](const auto &path) {
        return path->operation != nextOperation;
      }))
    return std::nullopt;

  std::optional<ChainWireOp> next = getChainWireOp(nextOperation, isSupported);
  if (!next || next->inputs.size() != current.outputs.size() ||
      !canContinue(firstOperation, nextOperation))
    return std::nullopt;

  for (auto [index, path] : llvm::enumerate(pathEnds))
    if (next->inputs[index] != path->wire)
      return std::nullopt;
  return next;
}

llvm::SmallVector<cudaq::opt::ScalarWireChain, 0>
cudaq::opt::collectScalarWireChains(Block &block,
                                    ScalarWireChainPredicate isSupported,
                                    ScalarWireChainContinuation canContinue) {
  llvm::SmallVector<ScalarWireChain, 0> chains;
  llvm::SmallDenseSet<Operation *> collected;

  for (Operation &operation : block) {
    if (collected.contains(&operation))
      continue;

    std::optional<ChainWireOp> first = getChainWireOp(&operation, isSupported);
    if (!first || llvm::any_of(first->inputs,
                               [](Value input) { return !input.hasOneUse(); }))
      continue;

    ScalarWireChain chain;
    chain.inputs = first->inputs;
    chain.scopeSteps.resize(first->inputs.size());
    std::optional<ChainWireOp> current = std::move(first);
    while (current) {
      chain.operations.push_back(current->operation);
      chain.outputs = current->outputs;
      collected.insert(current->operation);
      if (llvm::any_of(current->outputs,
                       [](Value output) { return !output.hasOneUse(); }))
        break;
      current = matchNextOperation(*current, chain.operations.front(),
                                   chain.scopeSteps, isSupported, canContinue);
    }
    chains.push_back(std::move(chain));
  }
  return chains;
}

void cudaq::opt::replaceScalarWireChain(ScalarWireChain &&chain,
                                        ValueRange replacements) {
  assert(replacements.size() == chain.outputs.size());
  for (auto [replacementValue, original, steps] :
       llvm::zip(replacements, chain.outputs, chain.scopeSteps)) {
    Value replacement = replacementValue;
    for (ScalarWireChainScopeStep &scopeStep : steps) {
      scopeStep.continueOperand->set(replacement);
      replacement = scopeStep.wire;
    }
    original.replaceAllUsesWith(replacement);
  }
  for (Operation *operation : llvm::reverse(chain.operations))
    operation->erase();
}
