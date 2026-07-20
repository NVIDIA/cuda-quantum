/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/QEC/QECDialect.h"
#include "cudaq/Optimizer/Dialect/QEC/QECOps.h"

using namespace mlir;

#include "cudaq/Optimizer/Dialect/QEC/QECDialect.cpp.inc"

void cudaq::qec::QECDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cudaq/Optimizer/Dialect/QEC/QECOps.cpp.inc"
      >();
}
