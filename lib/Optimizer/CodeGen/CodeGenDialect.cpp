/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CodeGenDialect.h"
#include "CodeGenOps.h"
#include "mlir/IR/DialectImplementation.h"

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/CodeGen/CodeGenDialect.cpp.inc"

//===----------------------------------------------------------------------===//

void cudaq::codegen::CodeGenDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "cudaq/Optimizer/CodeGen/CodeGenOps.cpp.inc"
      >();
}
