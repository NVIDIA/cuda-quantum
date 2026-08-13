/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Transforms/ScalarWireTraversal.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;

static Block *getValueBlock(Value wire) {
  if (auto argument = dyn_cast<BlockArgument>(wire))
    return argument.getOwner();
  if (auto result = dyn_cast<OpResult>(wire))
    return result.getOwner()->getBlock();
  return nullptr;
}

// Returns whether a direct use is nested only in single-block lexical scopes.
// Other region operations are traversal boundaries.
static bool entersSingleBlockLexicalScopesOnly(Block *nested, Block *outer) {
  while (nested != outer) {
    if (!nested)
      return false;
    auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(nested->getParentOp());
    if (!scope || !scope.getInitRegion().hasOneBlock())
      return false;
    nested = scope->getBlock();
  }
  return true;
}

/// Returns the `cc.continue` that forwards values from a single-block lexical
/// scope. Direction-specific helpers validate its positional mapping to the
/// scope results.
static std::optional<cudaq::cc::ContinueOp>
getSingleBlockScopeContinue(cudaq::cc::ScopeOp scope) {
  if (!scope || !scope.getInitRegion().hasOneBlock())
    return std::nullopt;
  auto cont = dyn_cast<cudaq::cc::ContinueOp>(
      scope.getInitRegion().front().getTerminator());
  if (!cont || cont.getNumOperands() != scope->getNumResults())
    return std::nullopt;
  return cont;
}

/// Return whether an operation can be followed as a direct scalar-wire step.
/// Calls, region operations, and terminators require control-flow semantics
/// that this helper deliberately does not model.
static bool isDirectScalarWireStep(Operation *operation) {
  return !isa<CallOpInterface>(operation) && operation->getNumRegions() == 0 &&
         !operation->hasTrait<OpTrait::IsTerminator>();
}

/// Traverses from a `cc.continue` operand to the scope result at the same
/// position. The returned step retains the continuation operand so callers
/// can update lexical forwarding when rewriting the wire.
static std::optional<cudaq::opt::ScalarWireStep>
traverseScopeForward(OpOperand *use) {
  auto cont = dyn_cast<cudaq::cc::ContinueOp>(use->getOwner());
  if (!cont)
    return std::nullopt;
  auto scope = dyn_cast<cudaq::cc::ScopeOp>(cont->getParentOp());
  if (!scope || !getSingleBlockScopeContinue(scope))
    return std::nullopt;
  unsigned index = use->getOperandNumber();
  if (index >= scope->getNumResults() ||
      !isa<cudaq::quake::WireType>(scope->getResult(index).getType()))
    return std::nullopt;
  Value result = scope->getResult(index);
  if (!result.hasOneUse())
    return std::nullopt;
  return cudaq::opt::ScalarWireStep{result, scope, scope->getBlock(), use};
}

/// Traverses from a scope result to the `cc.continue` operand at the same
/// position. The returned step retains the continuation operand so callers
/// can update lexical forwarding when rewriting the wire.
static std::optional<cudaq::opt::ScalarWireStep>
traverseScopeBackward(OpResult result, cudaq::cc::ScopeOp scope) {
  auto cont = getSingleBlockScopeContinue(scope);
  if (!cont || result.getResultNumber() >= cont->getNumOperands())
    return std::nullopt;
  Operation *continueOperation = cont->getOperation();
  OpOperand *operand =
      &continueOperation->getOpOperand(result.getResultNumber());
  if (!isa<cudaq::quake::WireType>(operand->get().getType()) ||
      !operand->get().hasOneUse())
    return std::nullopt;
  return cudaq::opt::ScalarWireStep{operand->get(), *cont,
                                    continueOperation->getBlock(), operand};
}

std::optional<cudaq::opt::ScalarWireStep>
cudaq::opt::traverseScalarWire(Value wire,
                               ScalarWireTraversalDirection direction) {
  if (!isa<cudaq::quake::WireType>(wire.getType()) || !wire.hasOneUse())
    return std::nullopt;

  if (direction == ScalarWireTraversalDirection::Forward) {
    OpOperand *use = &*wire.getUses().begin();
    Operation *user = use->getOwner();
    // A `cc.continue` forwards a value defined inside a lexical scope to its
    // corresponding scope result.
    if (isa<cudaq::cc::ContinueOp>(user))
      return traverseScopeForward(use);
    if (!isDirectScalarWireStep(user))
      return std::nullopt;
    // Values defined outside a lexical scope are captured implicitly. Accept
    // the use only when every intervening region is a supported scope.
    if (!entersSingleBlockLexicalScopesOnly(user->getBlock(),
                                            getValueBlock(wire)))
      return std::nullopt;
    return ScalarWireStep{wire, user, user->getBlock()};
  }

  auto result = dyn_cast<OpResult>(wire);
  if (!result)
    return std::nullopt;
  // A scope result is reached by following its corresponding `cc.continue`
  // operand back into the lexical scope.
  if (auto scope = dyn_cast<cudaq::cc::ScopeOp>(result.getOwner())) {
    return traverseScopeBackward(result, scope);
  }
  if (!isDirectScalarWireStep(result.getOwner()))
    return std::nullopt;
  return ScalarWireStep{wire, result.getOwner(), result.getOwner()->getBlock()};
}
