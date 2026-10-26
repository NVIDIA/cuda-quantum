/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Dialect/Event/IR/EventDialect.h"
#include "qlx/Dialect/Event/IR/EventInterfaces.h"
#include "qlx/Dialect/Event/IR/EventOps.h"
#include "qlx/Dialect/Event/IR/EventTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace qlx::event;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/Event/IR/EventDialect.cpp.inc"

#include "qlx/Dialect/Event/IR/EventInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "qlx/Dialect/Event/IR/EventTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// HandleType
//===----------------------------------------------------------------------===//

// `ownership` is checked for non-emptiness and nothing more: the set of
// disciplines belongs to the tiers that instantiate the handle, and the
// operations needing a particular one check it where they need it. `stream` is
// unchecked provenance, hence the unnamed parameter.
LogicalResult HandleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Type payload, StringRef ownership,
                                 SymbolRefAttr) {
  if (!payload)
    return emitError() << "handle payload type must not be null";
  if (ownership.empty())
    return emitError() << "handle ownership must not be empty";
  return success();
}

//===----------------------------------------------------------------------===//
// Shared verifier helpers
//===----------------------------------------------------------------------===//

namespace {

// A consuming op moves a delivery to exactly one owner, and "linear" is the
// only discipline that states that obligation, so await/cancel/try_take
// require it with no per-caller opt-out. `action` is how the op is named in
// the diagnostic ("cancellation", not the `cancel` mnemonic, where prose reads
// better), so the message is written once for all three.
LogicalResult verifyLinearOwnership(Operation *op, HandleType eventType,
                                    StringRef action) {
  StringRef ownership = eventType.getOwnership();
  if (ownership != "linear")
    return op->emitOpError(action)
           << " requires a linear event owner, but got '" << ownership << "'";
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// IsOp
//===----------------------------------------------------------------------===//

LogicalResult IsOp::verify() {
  StringRef state = getState();
  if (state != "pending" && state != "ready" && state != "failed" &&
      state != "cancelled" && state != "exhausted")
    return emitOpError(
        "event state must be pending, ready, failed, cancelled, or exhausted");
  return success();
}

//===----------------------------------------------------------------------===//
// SelectReadyOp
//===----------------------------------------------------------------------===//

LogicalResult SelectReadyOp::verify() {
  // Same-type-across-operands is enforced by the SameTypeOperands trait.
  if (getEvents().empty())
    return emitOpError("requires at least one event");
  StringRef policy = getPolicy();
  if (policy != "priority" && policy != "deterministic" && policy != "fair")
    return emitOpError("policy must be priority, deterministic, or fair");
  return success();
}

//===----------------------------------------------------------------------===//
// TryTakeOp
//===----------------------------------------------------------------------===//

LogicalResult TryTakeOp::verify() {
  if (failed(verifyLinearOwnership(getOperation(), getEvent().getType(),
                                   "try_take")))
    return failure();
  if (getCarries().getTypes() != getResultTypes())
    return emitOpError("carry and result types must match exactly");

  // Types that don't implement UniqueCarryOwnerInterface are silently
  // unchecked here -- see that interface's doc comment.
  llvm::SmallDenseSet<Attribute, 8> carriedOwnerKeys;
  for (Type type : getCarries().getTypes()) {
    auto unique = dyn_cast<UniqueCarryOwnerInterface>(type);
    if (!unique)
      continue;
    Attribute key = unique.getUniqueOwnerKey();
    assert(key && "UniqueCarryOwnerInterface must not return a null key");
    if (!carriedOwnerKeys.insert(key).second)
      return emitOpError("cannot carry more than one owner for resource ")
             << key;
  }
  return success();
}

// Region contents are not yet verified when verify() runs, so the checks
// below -- which inspect block arguments and the terminator -- live here
// instead: verifyRegions() runs after blocks are confirmed to have a
// terminator and after SingleBlockImplicitTerminator has confirmed it is a
// YieldOp.
LogicalResult TryTakeOp::verifyRegions() {
  auto verifyBranch = [&](Region &region, Type alternative,
                          StringRef label) -> LogicalResult {
    Block &block = region.front();
    if (block.getNumArguments() != getCarries().size() + 1)
      return emitOpError()
             << label
             << " region requires one alternative argument plus carries";
    if (block.getArgument(0).getType() != alternative)
      return emitOpError() << label
                           << " alternative argument has the wrong type";
    for (auto [argument, carry] :
         llvm::zip(block.getArguments().drop_front(), getCarries()))
      if (argument.getType() != carry.getType())
        return emitOpError()
               << label << " carry arguments have the wrong types";
    auto yield = cast<YieldOp>(block.getTerminator());
    if (yield.getOperandTypes() != getResultTypes())
      return emitOpError() << label << " yield types must match results";
    return success();
  };
  HandleType eventType = getEvent().getType();
  if (failed(verifyBranch(getReady(), eventType.getPayload(), "ready")) ||
      failed(verifyBranch(getPending(), eventType, "pending")) ||
      failed(verifyBranch(getFailed(), IntegerType::get(getContext(), 8),
                          "failed")))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// CancelOp
//===----------------------------------------------------------------------===//

LogicalResult CancelOp::verify() {
  return verifyLinearOwnership(getOperation(), getEvent().getType(),
                               "cancellation");
}

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

// Payload/result type equality is enforced by the TypesMatchWith trait.
LogicalResult AwaitOp::verify() {
  return verifyLinearOwnership(getOperation(), getEvent().getType(), "await");
}

//===----------------------------------------------------------------------===//
// FenceOp
//===----------------------------------------------------------------------===//

LogicalResult FenceOp::verify() {
  ArrayAttr effects = getEffects();
  if (effects.empty())
    return emitOpError("requires at least one semantic effect");
  llvm::SmallDenseSet<StringRef, 8> seen;
  for (Attribute effect : effects) {
    // Each element is already known to be a StringAttr: `effects` is
    // StrArrayAttr, so ODS rejects any other element type before this
    // verifier runs.
    StringRef name = cast<StringAttr>(effect).getValue();
    if (name != "all" && name != "quantum" && name != "classical" &&
        name != "resource" && name != "event" && name != "frame" &&
        name != "outcome" && name != "selection")
      return emitOpError("unknown semantic effect '") << name << "'";
    if (!seen.insert(name).second)
      return emitOpError("semantic effects must be unique");
  }
  if (seen.contains("all") && effects.size() != 1)
    return emitOpError("effect 'all' cannot be combined with other effects");
  return success();
}

//===----------------------------------------------------------------------===//
// SelectionOp
//===----------------------------------------------------------------------===//

LogicalResult SelectionOp::verify() {
  StringRef mode = getMode();
  if (mode != "require" && mode != "condition_results" && mode != "abort_on")
    return emitOpError("mode must be require, condition_results, or abort_on");
  bool expected = mode != "abort_on";
  if (getAcceptWhen() != expected)
    return emitOpError("accept_when disagrees with the selection mode");
  if (getProfileAttr() && !getAttemptAttr())
    return emitOpError("profile requires an explicit selection attempt");
  return success();
}

// Symbol resolution is costly (it walks to the enclosing symbol table) and
// is therefore hooked in here rather than in verify(), which the verifier
// framework re-runs on every IR change; SymbolUserOpInterface gives the
// caller a cached SymbolTableCollection instead. Only confirms attempt/
// profile resolve to a symbol: event depends on no other dialect, so it
// cannot check what that symbol names.
LogicalResult
SelectionOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (auto attempt = getAttemptAttr())
    if (!symbolTable.lookupNearestSymbolFrom(getOperation(), attempt))
      return emitOpError("attempt must resolve to a symbol");
  if (auto profile = getProfileAttr())
    if (!symbolTable.lookupNearestSymbolFrom(getOperation(), profile))
      return emitOpError("profile must resolve to a symbol");
  return success();
}

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qlx/Dialect/Event/IR/EventOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void EventDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "qlx/Dialect/Event/IR/EventTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "qlx/Dialect/Event/IR/EventOps.cpp.inc"
      >();
}
