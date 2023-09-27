/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
  if (parser.parseGreater())
    return {};
  return cc::StructType::get(ctx, nameAttr, members, isOpaque, isPacked);
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
  printer << '>';
}

unsigned
cc::StructType::getTypeSizeInBits(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  return 0;
}

unsigned cc::StructType::getABIAlignment(const DataLayout &dataLayout,
                                         DataLayoutEntryListRef params) const {
  return 0;
}

unsigned
cc::StructType::getPreferredAlignment(const DataLayout &dataLayout,
                                      DataLayoutEntryListRef params) const {
  return 0;
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

namespace cudaq {

cc::CallableType cc::CallableType::getNoSignature(MLIRContext *ctx) {
  return CallableType::get(ctx, FunctionType::get(ctx, {}, {}));
}

void cc::CCDialect::registerTypes() {
  addTypes<ArrayType, CallableType, PointerType, StdvecType, StructType>();
}

} // namespace cudaq
