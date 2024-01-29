/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

using namespace mlir;

namespace cudaq {
//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

Type cc::StructType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};
  std::string name;
  auto *ctx = parser.getContext();
  StringAttr nameAttr;
  if (succeeded(parser.parseOptionalString(&name)))
    nameAttr = StringAttr::get(ctx, name);
  SmallVector<Type> members;
  bool isOpaque = true;
  if (succeeded(parser.parseOptionalLBrace())) {
    isOpaque = false;
    do {
      Type member;
      auto optTy = parser.parseOptionalType(member);
      if (!optTy.has_value())
        break;
      if (!succeeded(*optTy))
        return {};
      members.push_back(member);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRBrace())
      return {};
  }
  bool isPacked = false;
  if (succeeded(parser.parseOptionalKeyword("packed")))
    isPacked = true;
  std::uint64_t size = 0;
  unsigned align = 0;
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseInteger(size) || parser.parseComma() ||
        parser.parseInteger(align) || parser.parseRSquare())
      return {};
  }
  if (parser.parseGreater())
    return {};
  return cc::StructType::get(ctx, nameAttr, members, isOpaque, isPacked, size,
                             align);
}

void cc::StructType::print(AsmPrinter &printer) const {
  printer << '<';
  if (getName())
    printer << getName();
  if (!getOpaque()) {
    if (getName())
      printer << ' ';
    printer << '{';
    llvm::interleaveComma(getMembers(), printer);
    printer << '}';
  }
  if (getPacked()) {
    if (getName() || !getOpaque())
      printer << ' ';
    printer << "packed";
  }
  if (getBitSize() || getAlignment()) {
    if (getName() || !getOpaque() || getPacked())
      printer << ' ';
    printer << '[' << getBitSize() << ',' << getAlignment() << ']';
  }
  printer << '>';
}

unsigned
cc::StructType::getTypeSizeInBits(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  return static_cast<unsigned>(getBitSize());
}

unsigned cc::StructType::getABIAlignment(const DataLayout &dataLayout,
                                         DataLayoutEntryListRef params) const {
  return getAlignment();
}

unsigned
cc::StructType::getPreferredAlignment(const DataLayout &dataLayout,
                                      DataLayoutEntryListRef params) const {
  // No distinction between ABI and preferred alignments for now. Clang just
  // gives us an alignment value.
  return getAlignment();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

Type cc::ArrayType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};
  Type element;
  auto optTy = parser.parseOptionalType(element);
  if (optTy.has_value()) {
    if (!succeeded(*optTy))
      return {};
  }
  if (parser.parseKeyword("x"))
    return {};
  SizeType size;
  if (succeeded(parser.parseOptionalQuestion())) {
    size = unknownSize;
  } else {
    if (parser.parseInteger(size))
      return {};
  }
  if (parser.parseGreater())
    return {};
  return cc::ArrayType::get(parser.getContext(), element, size);
}

void cc::ArrayType::print(AsmPrinter &printer) const {
  printer << '<' << getElementType() << " x ";
  if (isUnknownSize())
    printer << '?';
  else
    printer << getSize();
  printer << '>';
}

} // namespace cudaq

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/CC/CCTypes.cpp.inc"

//===----------------------------------------------------------------------===//

namespace cudaq::cc {

Type cc::SpanLikeType::getElementType() const {
  return llvm::TypeSwitch<Type, Type>(*this).Case<StdvecType, CharspanType>(
      [](auto type) { return type.getElementType(); });
}

bool isDynamicType(Type ty) {
  if (isa<StdvecType>(ty))
    return true;
  if (auto strTy = dyn_cast<StructType>(ty)) {
    for (auto memTy : strTy.getMembers())
      if (isDynamicType(memTy))
        return true;
    return false;
  }
  if (auto arrTy = dyn_cast<ArrayType>(ty))
    return arrTy.isUnknownSize() || isDynamicType(arrTy.getElementType());
  // Note: this isn't considering quake, builtin, etc. types.
  return false;
}

CallableType CallableType::getNoSignature(MLIRContext *ctx) {
  return CallableType::get(ctx, FunctionType::get(ctx, {}, {}));
}

void CCDialect::registerTypes() {
  addTypes<ArrayType, CallableType, CharspanType, PointerType, StdvecType,
           StructType>();
}

} // namespace cudaq::cc
