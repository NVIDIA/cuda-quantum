/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Dialect/Cflow/IR/CflowDialect.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace qlx::cflow;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/Cflow/IR/CflowDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verifyRegions() {
  for (Region *region : {&getThenRegion(), &getElseRegion()}) {
    auto yield = cast<YieldOp>(region->front().getTerminator());
    if (yield.getNumOperands() != getNumResults() ||
        !llvm::equal(yield.getOperandTypes(), getResultTypes()))
      return emitOpError("branch yield types must match cflow.if results");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::verify() {
  if (getInits().getTypes() != getResultTypes())
    return emitOpError("init and result types must be identical");
  return success();
}

LogicalResult WhileOp::verifyRegions() {
  Block &before = getBeforeRegion().front();
  Block &after = getAfterRegion().front();
  if (before.getArgumentTypes() != getResultTypes() ||
      after.getArgumentTypes() != getResultTypes())
    return emitOpError(
        "before/after block arguments must match the carried result types");

  auto condition = dyn_cast<WhileConditionOp>(before.getTerminator());
  if (!condition)
    return emitOpError(
        "before region must terminate with cflow.while_condition");
  if (condition.getForwarded().getTypes() != getResultTypes())
    return emitOpError(
        "while_condition forwarded types must match loop result types");

  auto yield = dyn_cast<YieldOp>(after.getTerminator());
  if (!yield)
    return emitOpError("after region must terminate with cflow.yield");
  if (yield.getOperandTypes() != getResultTypes())
    return emitOpError("after-region yield types must match loop result types");
  return success();
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//
//
// %out = cflow.repeat 64
//     iter(%arg : T = %init, ...) event_id = "..."? {
//   ...
//   cflow.yield %next : T
// }
//

ParseResult RepeatOp::parse(OpAsmParser &parser, OperationState &result) {
  int64_t count;
  if (parser.parseInteger(count))
    return failure();
  result.addAttribute("count", parser.getBuilder().getI64IntegerAttr(count));

  SmallVector<OpAsmParser::Argument> iterArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> initOperands;
  SmallVector<Type> initTypes;

  if (parser.parseKeyword("iter") || parser.parseLParen())
    return failure();

  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          OpAsmParser::Argument arg;
          OpAsmParser::UnresolvedOperand init;
          if (parser.parseArgument(arg, /*allowType=*/true,
                                   /*allowAttrs=*/false))
            return failure();
          if (parser.parseEqual() || parser.parseOperand(init))
            return failure();
          iterArgs.push_back(arg);
          initOperands.push_back(init);
          initTypes.push_back(arg.type);
          return success();
        }))
      return failure();
    if (parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(initOperands, initTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  result.addTypes(initTypes);

  if (succeeded(parser.parseOptionalKeyword("event_id"))) {
    StringAttr eventId;
    if (parser.parseEqual() || parser.parseAttribute(eventId))
      return failure();
    result.addAttribute("event_id", eventId);
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, iterArgs, /*enableNameShadowing=*/false))
    return failure();
  ensureTerminator(*body, parser.getBuilder(), result.location);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void RepeatOp::print(OpAsmPrinter &printer) {
  printer << " " << getCount() << " iter(";

  // Indexed against the block's own argument count, not `inits`' -- this
  // runs on IR a debugger or `--mlir-print-ir-after-all` can present before
  // verification, where the two are not yet known to agree.
  Block &entryBlock = getBody().front();
  Operation::operand_range inits = getInits();
  for (auto [i, arg] : llvm::enumerate(entryBlock.getArguments())) {
    if (i > 0)
      printer << ", ";
    printer.printRegionArgument(arg);
    if (i < inits.size())
      printer << " = " << inits[i];
  }
  printer << ")";

  // Print through the attribute's own printer so a value containing a
  // quote or backslash escapes correctly; hand-quoting a bare StringRef
  // (as an earlier version of this printer did) round-trips only for
  // values with no characters that need escaping.
  if (StringAttr eventId = getEventIdAttr())
    printer << " event_id = " << eventId;

  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"count", "event_id"});
}

LogicalResult RepeatOp::verify() {
  if (getInits().getTypes() != getResultTypes())
    return emitOpError("iter-init and result types must be identical");
  return success();
}

LogicalResult RepeatOp::verifyRegions() {
  Block &body = getBody().front();
  if (body.getArgumentTypes() != getInits().getTypes())
    return emitOpError("body block arguments must match the iter-init types");

  // SingleBlockImplicitTerminator<"YieldOp"> has already confirmed the
  // terminator is a YieldOp by the time verifyRegions() runs.
  auto yield = cast<YieldOp>(body.getTerminator());
  if (yield.getOperandTypes() != getResultTypes())
    return emitOpError("yielded types must match loop result types");
  return success();
}

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qlx/Dialect/Cflow/IR/CflowOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void CflowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "qlx/Dialect/Cflow/IR/CflowOps.cpp.inc"
      >();
}
