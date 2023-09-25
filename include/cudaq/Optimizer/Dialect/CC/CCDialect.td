/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
  let summary = "Classical compute dialect for the CUDA Quantum programming model.";
  let description = [{
    CC is envisioned to be a high-level, portable MLIR dialect that is very
    close to the classical computing side of the CUDA Quantum programming model.
    The classical computation found in CUDA Quantum is a restricted subdomain of C++.
  }];

  let cppNamespace = "cudaq::cc";
  let useDefaultTypePrinterParser = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;

  let extraClassDeclaration = [{
    /// Register all CC types.
    void registerTypes();
    /// Register all CC attributes.
    void registerAttrs();
  }];
}

#endif // CUDAQ_DIALECT_CC_DIALECT
