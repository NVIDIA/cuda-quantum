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
  let summary = "Hardware-independent quantum dialect for CUDA-Q.";
  let description = [{
    Quake is a high-level, portable MLIR dialect for CUDA-Q quantum programs.
    It represents quantum allocation and lifetime, quantum operations,
    measurement, and calls between quantum kernels. Quake supports both
    reference-oriented and value-oriented forms and is not tied to a particular
    quantum hardware or machine.
  }];
  let cppNamespace = "cudaq::quake";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    /// Register all Quake types.
    void registerTypes();
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QUAKE_IR_QUAKE
