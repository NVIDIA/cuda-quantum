/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QUAKE_IR_QUAKE
#define CUDAQ_OPTIMIZER_DIALECT_QUAKE_IR_QUAKE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition.
//===----------------------------------------------------------------------===//

def QuakeDialect : Dialect {
  let name = "quake";
  let summary = "Higher level, memory-semantics dialect for cudaq. ";
  let description = [{
    Quake is envisioned to be a high-level, portable MLIR dialect that is very
    close to the CUDA Quantum programming model. Quake is a "circuit building"
    language in which one writes programs that build quantum circuits. Quake is 
    not tied to a particular quantum hardware/machine.
  }];
  let cppNamespace = "::quake";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    /// Register all Quake types.
    void registerTypes();
  }];
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QUAKE_IR_QUAKE
