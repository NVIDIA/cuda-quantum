/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
  let summary = "Quantum IR dialect for CUDA-Q.";
  let description = [{
    Quake is CUDA-Q's quantum IR dialect. It represents qudits as typed values
    with explicit dataflow constraints. Its instructions model allocation, quantum
    operations, measurement, and calls between quantum kernels. Quake can
    represent quantum data using reference semantics or value semantics and is
    used across compilation stages, including circuits transformed for a
    specific target.
  }];
  let cppNamespace = "cudaq::quake";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    /// Register all Quake types.
    void registerTypes();
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QUAKE_IR_QUAKE
