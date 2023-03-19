/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/CC/CCTypes.cpp.inc"

cudaq::cc::LambdaType cudaq::cc::LambdaType::getNoSignature(MLIRContext *ctx) {
  return LambdaType::get(ctx, FunctionType::get(ctx, {}, {}));
}

//===----------------------------------------------------------------------===//

void cudaq::cc::CCDialect::registerTypes() {
  addTypes<LambdaType, PointerType, StdvecType>();
}
