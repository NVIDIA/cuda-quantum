/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Veq's custom parser and pretty printing.
//
// veq `<` (`?` | int) `>`
//===----------------------------------------------------------------------===//

void quake::VeqType::print(AsmPrinter &os) const {
  os << '<';
  if (hasSpecifiedSize())
    os << getSize();
  else
    os << '?';
  os << '>';
}

Type quake::VeqType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};
  std::size_t size;
  if (succeeded(parser.parseOptionalQuestion()))
    size = 0;
  else if (parser.parseInteger(size))
    return {};
  if (parser.parseGreater())
    return {};
  return get(parser.getContext(), size);
}

LogicalResult
quake::VeqType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                       std::size_t size) {
  // FIXME: Do we want to check the size of the veq for some bound?
  return success();
}

//===----------------------------------------------------------------------===//

void quake::QuakeDialect::registerTypes() {
  addTypes<VeqType, RefType, WireType, ControlType>();
}
