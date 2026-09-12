/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CFLOW_DIALECT_TD
#define CFLOW_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Cflow Dialect
//===----------------------------------------------------------------------===//

def Cflow_Dialect : Dialect {
  let name = "cflow";
  let cppNamespace = "qlx::cflow";

  let summary = "Shared structured control-flow ops embedded in other dialects";
  let description = [{
    `cflow` defines the structured control-flow ops (`repeat`, `if`, `while`,
    `while_condition`, `yield`) that other dialects in this project embed
    directly into their own operations' bodies, instead of each dialect defining
    its own copy. It owns no pipeline stage of its own: it is a shared utility
    dialect.
  }];
}

#endif // CFLOW_DIALECT_TD
