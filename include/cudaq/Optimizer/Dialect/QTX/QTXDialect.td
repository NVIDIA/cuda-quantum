/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_DIALECT
#define CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition.
//===----------------------------------------------------------------------===//

def QTXDialect : Dialect {
  let name = "qtx";
  let summary = "The QTX dialect for quantum computing.";
  let description = [{
    QTX stands for quantum thread execution.  This dialect defines core types
    and operations for managing quantum memory and operating on quantum data.
  }];
  let cppNamespace = "::qtx";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    /// Register all QTX types.
    void registerTypes();
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_DIALECT
