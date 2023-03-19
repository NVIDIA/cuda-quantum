/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Dialect/Common/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Parameters directive
//===----------------------------------------------------------------------===//

// The following functions implement custom assembly directives.  We need these
// because we cannot use `|` in the assembly format string :(

ParseResult cudaq::parseParameters(
    OpAsmParser &parser, UnitAttr &isAdj,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &parameters) {
  if (parser.parseOptionalLess())
    return success();
  if (succeeded(parser.parseOptionalGreater()))
    return success(); // Empty parameters, OK!

  if (succeeded(parser.parseOptionalKeyword("adj"))) {
    isAdj = parser.getBuilder().getUnitAttr();
    if (succeeded(parser.parseOptionalGreater()))
      return success(); // Only <adj>, OK!

    if (parser.parseComma() || parser.parseOperandList(parameters))
      return failure();
  } else if (parser.parseOperandList(parameters)) {
    return failure();
  }

  return parser.parseGreater();
}

void cudaq::printParameters(OpAsmPrinter &printer, Operation *, UnitAttr isAdj,
                            OperandRange parameters) {
  if (isAdj == nullptr && parameters.empty())
    return;
  printer << '<';
  if (isAdj) {
    printer << "adj";
    if (!parameters.empty())
      printer << ", " << parameters;
  } else {
    printer << parameters;
  }
  printer << '>';
}

// TODO: Remove this once we refactor Quake too
ParseResult cudaq::parseParameters(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &parameters) {
  if (parser.parseOptionalVerticalBar())
    return success();
  if (succeeded(parser.parseOptionalVerticalBar()))
    return success();
  if (parser.parseOperandList(parameters))
    return failure();
  return parser.parseVerticalBar();
}

ParseResult cudaq::parseParameters(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &parameters,
    SmallVectorImpl<Type> &types) {
  if (parser.parseOptionalVerticalBar())
    return success();
  if (succeeded(parser.parseOptionalVerticalBar()))
    return success();
  if (parser.parseOperandList(parameters))
    return failure();
  if (parser.parseColonTypeList(types))
    return failure();
  return parser.parseVerticalBar();
}

void cudaq::printParameters(OpAsmPrinter &printer, Operation *,
                            OperandRange parameters, TypeRange types) {
  if (parameters.empty())
    return;
  printer << '|';
  printer << parameters;
  if (!types.empty()) {
    printer << " : " << types;
  }
  printer << '|';
}
