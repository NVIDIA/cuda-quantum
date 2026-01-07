/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
  std::size_t size = kDynamicSize;
  if (succeeded(parser.parseOptionalQuestion()))
    size = kDynamicSize;
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

// This recursive function returns true if and only if \p ty is a quake
// type in the set \e R, `{ ref, veq, struq }`, (loosely known as "reference"
// types) and the number of qubits is a compile-time known constant. This
// function returns false for any type not in the set \e R or if the composition
// of types contains a `veq` of unspecified size.
static bool isConstQuantumBits(Type ty) {
  if (isa<quake::RefType>(ty))
    return true;
  if (auto t = dyn_cast<quake::StruqType>(ty)) {
    for (auto m : t.getMembers())
      if (!isConstQuantumBits(m))
        return false;
    return true;
  }
  if (auto t = dyn_cast<quake::VeqType>(ty))
    if (t.hasSpecifiedSize())
      return true;
  return false;
}

bool quake::isConstantQuantumRefType(Type ty) { return isConstQuantumBits(ty); }

std::size_t quake::getAllocationSize(Type ty) {
  if (isa<quake::RefType>(ty))
    return 1;
  if (auto stq = dyn_cast<quake::StruqType>(ty)) {
    std::size_t size = 0;
    for (auto m : stq.getMembers())
      size += getAllocationSize(m);
    return size;
  }
  auto veq = cast<quake::VeqType>(ty);
  assert(veq.hasSpecifiedSize() && "veq type must have constant size");
  return veq.getSize();
}

//===----------------------------------------------------------------------===//

void quake::QuakeDialect::registerTypes() {
  addTypes<CableType, ControlType, MeasureType, RefType, StateType, StruqType,
           VeqType, WireType>();
}
