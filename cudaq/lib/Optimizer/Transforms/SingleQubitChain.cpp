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
#include "llvm/ADT/STLExtras.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include <optional>
#include <utility>

using namespace mlir;

namespace {

struct WirePathEnd {
  Operation *operation;
  Value wire;
};

} // namespace

// A chain may cross a `cc.scope` only when the scope is an ordinary
// single-block lexical scope, not an atomic quantum region.
static bool isTransparentScope(cudaq::cc::ScopeOp scope) {
  return !scope.getAtomicQuantumRegionAttr() &&
         scope.getInitRegion().hasOneBlock();
}

// Returns whether `nested` is inside `outer` through only transparent
// `cc.scope` operations. Any other enclosing region prevents traversal.
static bool entersSingleBlockLexicalScopesOnly(Block *nested, Block *outer) {
  while (nested != outer) {
    if (!nested)
      return false;
    auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(nested->getParentOp());
    if (!scope || !isTransparentScope(scope))
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

// Follow the unique use of one lane forward to its next direct user. A
// `cc.continue` use passes the wire to the matching result of its transparent
// enclosing scope, and that result is followed in turn.
static std::optional<WirePathEnd> traceWire(Value wire) {
  while (true) {
    if (!isa<cudaq::quake::WireType>(wire.getType()) || !wire.hasOneUse())
      return std::nullopt;

    OpOperand &use = *wire.getUses().begin();
    Operation *user = use.getOwner();
    auto cont = dyn_cast<cudaq::cc::ContinueOp>(user);
    if (!cont) {
      if (!isDirectWireStep(user) ||
          !entersSingleBlockLexicalScopesOnly(user->getBlock(),
                                              wire.getParentBlock()))
        return std::nullopt;
      return WirePathEnd{user, wire};
    }

    auto scope = dyn_cast<cudaq::cc::ScopeOp>(cont->getParentOp());
    if (!scope || !isTransparentScope(scope) ||
        scope.getInitRegion().front().getTerminator() != user ||
        cont.getNumOperands() != scope->getNumResults())
      return std::nullopt;
    unsigned index = use.getOperandNumber();
    if (index >= scope->getNumResults() ||
        !isa<cudaq::quake::WireType>(scope->getResult(index).getType()))
      return std::nullopt;
    wire = scope->getResult(index);
  }
}

static llvm::SmallVector<bool> getPolarities(Operation *operation) {
  if (auto gate = dyn_cast<cudaq::quake::OperatorInterface>(operation))
    return cudaq::quake::getControlPolarities(gate);
  return {};
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
// gate at its corresponding input position and that gate has the same control
// polarities as the chain's first gate. Returns the next gate and replaces
// `flow` with its wire flow, or returns null and leaves `flow` unchanged.
static Operation *
matchNextOperation(cudaq::quake::detail::WireFlow &flow, Operation *head,
                   cudaq::opt::SingleQubitChainPredicate isSupported) {
  llvm::SmallVector<std::optional<WirePathEnd>> pathEnds;
  pathEnds.reserve(flow.results.size());
  for (Value output : flow.results)
    pathEnds.push_back(traceWire(output));

  if (llvm::any_of(pathEnds, [](const auto &path) { return !path; }))
    return nullptr;

  Operation *nextOperation = pathEnds.front()->operation;
  if (llvm::any_of(pathEnds, [&](const auto &path) {
        return path->operation != nextOperation;
      }))
    return nullptr;

  auto next = getChainFlow(nextOperation, isSupported);
  if (!next || next->inputs.size() != flow.results.size() ||
      getPolarities(head) != getPolarities(nextOperation))
    return nullptr;

  for (auto [index, path] : llvm::enumerate(pathEnds))
    if (next->inputs[index] != path->wire)
      return nullptr;
  flow = std::move(*next);
  return nextOperation;
}

llvm::SmallVector<cudaq::opt::SingleQubitChain, 0>
cudaq::opt::collectSingleQubitChains(Block &block,
                                     SingleQubitChainPredicate isSupported) {
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
    for (Operation *current = &operation; current;
         current =
             matchNextOperation(*flow, chain.operations.front(), isSupported)) {
      chain.operations.push_back(current);
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
  // Erase the later gates last to first, forwarding each one's wire results to
  // its inputs. Every use of the chain's results, including a `cc.continue`
  // operand that carries the chain out of a scope, then refers to the first
  // gate's results.
  for (Operation *operation :
       llvm::reverse(llvm::drop_begin(chain.operations))) {
    auto flow = cudaq::quake::detail::getWireFlow(operation);
    assert(flow && "chain operations have a wire flow");
    for (auto [result, input] : llvm::zip_equal(flow->results, flow->inputs))
      result.replaceAllUsesWith(input);
    operation->erase();
  }

  Operation *head = chain.operations.front();
  auto flow = cudaq::quake::detail::getWireFlow(head);
  assert(flow && flow->results.size() == replacements.size());
  for (auto [result, replacement] :
       llvm::zip_equal(flow->results, replacements))
    result.replaceAllUsesWith(replacement);
  head->erase();
}

void cudaq::opt::forEachNestedBlock(
    Region &region, llvm::function_ref<void(Block &)> rewriteBlock) {
  for (Block &block : region) {
    rewriteBlock(block);
    for (Operation &operation : block)
      for (Region &nested : operation.getRegions())
        forEachNestedBlock(nested, rewriteBlock);
  }
}
