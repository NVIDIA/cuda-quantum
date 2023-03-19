/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXTypes.h"

#include "mlir/IR/DialectImplementation.h"

using namespace qtx;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.cpp.inc"

//===----------------------------------------------------------------------===//

void QTXDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.cpp.inc"
      >();
}
