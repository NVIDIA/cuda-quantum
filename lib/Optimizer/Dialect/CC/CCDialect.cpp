/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCInterfaces.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Dialect/CC/CCDialect.cpp.inc"

//===----------------------------------------------------------------------===//

void cudaq::cc::CCDialect::registerAttrs() {}

void cudaq::cc::CCDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "cudaq/Optimizer/Dialect/CC/CCOps.cpp.inc"
      >();
  registerAttrs();
  addInterfaces<CCInlinerInterface>();
}
