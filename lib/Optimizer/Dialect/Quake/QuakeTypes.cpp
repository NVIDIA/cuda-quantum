/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
  std::size_t size = 0;
  if (succeeded(parser.parseOptionalQuestion()))
    size = 0;
  else if (parser.parseInteger(size))
    return {};
  if (parser.parseGreater())
    return {};
  return get(parser.getContext(), size);
}

//===----------------------------------------------------------------------===//

Type quake::StruqType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};
  std::string name;
  auto *ctx = parser.getContext();
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalString(&name))) {
    nameAttr = StringAttr::get(ctx, name);
    if (parser.parseColon())
      return {};
  }
  SmallVector<Type> members;
  do {
    Type member;
    auto optTy = parser.parseOptionalType(member);
    if (!optTy.has_value())
      break;
    if (!succeeded(*optTy))
      return {};
    if (!llvm::isa<quake::RefType, quake::VeqType>(member))
      parser.emitError(parser.getCurrentLocation(),
                       "invalid struq member type");
    members.push_back(member);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseGreater())
    return {};
  return quake::StruqType::get(ctx, nameAttr, members);
}

void quake::StruqType::print(AsmPrinter &printer) const {
  printer << '<';
  if (getName())
    printer << getName() << ": ";
  llvm::interleaveComma(getMembers(), printer);
  printer << '>';
}

//===----------------------------------------------------------------------===//

void quake::QuakeDialect::registerTypes() {
  addTypes<ControlType, MeasureType, RefType, StruqType, VeqType, WireType>();
}
