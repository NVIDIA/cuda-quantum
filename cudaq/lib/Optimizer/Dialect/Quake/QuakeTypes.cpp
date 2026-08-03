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
#include <limits>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.cpp.inc"

static std::optional<cudaq::quake::Pauli> symbolizePauli(char value) {
  switch (value) {
  case 'I':
    return cudaq::quake::Pauli::I;
  case 'X':
    return cudaq::quake::Pauli::X;
  case 'Y':
    return cudaq::quake::Pauli::Y;
  case 'Z':
    return cudaq::quake::Pauli::Z;
  default:
    return std::nullopt;
  }
}

std::optional<cudaq::quake::PauliWord>
cudaq::quake::symbolizePauliWord(llvm::StringRef value) {
  cudaq::quake::PauliWord result;
  result.reserve(value.size());
  for (char character : value) {
    auto pauli = symbolizePauli(character);
    if (!pauli)
      return std::nullopt;
    result.push_back(*pauli);
  }
  return result;
}

static std::optional<std::size_t> checkedAdd(std::size_t lhs, std::size_t rhs) {
  if (std::numeric_limits<std::size_t>::max() - lhs < rhs)
    return std::nullopt;
  return lhs + rhs;
}

//===----------------------------------------------------------------------===//
// Veq's custom parser and pretty printing.
//
// veq `<` (`?` | int) `>`
//===----------------------------------------------------------------------===//

void cudaq::quake::VeqType::print(AsmPrinter &os) const {
  os << '<';
  if (hasSpecifiedSize())
    os << getSize();
  else
    os << '?';
  os << '>';
}

Type cudaq::quake::VeqType::parse(AsmParser &parser) {
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

Type cudaq::quake::StruqType::parse(AsmParser &parser) {
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
    if (!llvm::isa<cudaq::quake::RefType, cudaq::quake::VeqType>(member))
      parser.emitError(parser.getCurrentLocation(),
                       "invalid struq member type");
    members.push_back(member);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseGreater())
    return {};
  return cudaq::quake::StruqType::get(ctx, nameAttr, members);
}

bool cudaq::quake::StruqType::hasSpecifiedSize() const {
  for (auto ty : getMembers())
    if (auto veqTy = llvm::dyn_cast<cudaq::quake::VeqType>(ty))
      if (!veqTy.hasSpecifiedSize())
        return false;
  return true;
}

std::optional<std::size_t> cudaq::quake::StruqType::getArity() const {
  return getQubitCount(*this);
}

void cudaq::quake::StruqType::print(AsmPrinter &printer) const {
  printer << '<';
  if (getName())
    printer << getName() << ": ";
  llvm::interleaveComma(getMembers(), printer);
  printer << '>';
}

//===----------------------------------------------------------------------===//

bool cudaq::quake::isConstantQuantumRefType(Type ty) {
  if (!isQuantumReferenceType(ty))
    return false;
  return getQubitCount(ty).has_value();
}

std::optional<std::size_t> cudaq::quake::getQubitCount(Type ty) {
  assert(isQuantumType(ty) && "expected a quantum type");
  if (isa<RefType, WireType, ControlType>(ty))
    return 1;
  if (auto veqTy = dyn_cast<VeqType>(ty)) {
    if (!veqTy.hasSpecifiedSize())
      return std::nullopt;
    return veqTy.getSize();
  }
  if (auto cableTy = dyn_cast<CableType>(ty))
    return cableTy.getSize();
  auto struqTy = cast<StruqType>(ty);
  std::size_t count = 0;
  for (auto member : struqTy.getMembers()) {
    if (!isa<RefType, VeqType>(member))
      return std::nullopt;
    auto memberCount = getQubitCount(member);
    if (!memberCount)
      return std::nullopt;
    auto updatedCount = checkedAdd(count, *memberCount);
    if (!updatedCount)
      return std::nullopt;
    count = *updatedCount;
  }
  return count;
}

std::size_t cudaq::quake::getAllocationSize(Type ty) {
  assert(isQuantumReferenceType(ty) && "expected a quantum reference type");
  auto count = getQubitCount(ty);
  assert(count && "quantum reference type must have constant size");
  return *count;
}

std::size_t cudaq::quake::getWireCount(Type ty) {
  assert(isQuantumValueType(ty) && "expected a quantum value type");
  auto count = getQubitCount(ty);
  assert(count && "quantum value type must have constant size");
  return *count;
}

//===----------------------------------------------------------------------===//

void cudaq::quake::QuakeDialect::registerTypes() {
  addTypes<CableType, ControlType, MeasureType, RefType, StateType, StruqType,
           VeqType, WireType>();
}
