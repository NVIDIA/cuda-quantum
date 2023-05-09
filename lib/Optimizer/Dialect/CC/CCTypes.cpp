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

} // namespace cudaq

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/CC/CCTypes.cpp.inc"

namespace cudaq {

cc::LambdaType cc::LambdaType::getNoSignature(MLIRContext *ctx) {
  return LambdaType::get(ctx, FunctionType::get(ctx, {}, {}));
}

//===----------------------------------------------------------------------===//

void cc::CCDialect::registerTypes() {
  addTypes<ArrayType, LambdaType, PointerType, StdvecType, StructType>();
}

} // namespace cudaq
