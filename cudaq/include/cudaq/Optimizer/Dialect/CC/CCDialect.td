/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_DIALECT_CC_DIALECT
#define CUDAQ_DIALECT_CC_DIALECT

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// Dialect definition.
//===----------------------------------------------------------------------===//

def CCDialect : Dialect {
  let name = "cc";
  let summary = "Classical compute dialect for the CUDA-Q programming model.";
  let description = [{
    CC is a high-level, portable MLIR dialect for the classical computing side
    of the CUDA-Q programming model. It represents the subset of C++ supported
    in CUDA-Q kernels.
  }];

  let cppNamespace = "cudaq::cc";
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all CC types.
    void registerTypes();
    /// Register all CC attributes.
    void registerAttrs();
  }];
}

#endif // CUDAQ_DIALECT_CC_DIALECT
