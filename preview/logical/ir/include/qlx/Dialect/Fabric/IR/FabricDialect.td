/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// Dialect definition for Fabric — a physical QEC fabric dialect.
//
// Fabric describes composable gadgets on QEC code blocks with:
//   - Value-semantic patches and syndromes
//   - Optional stabilizer-flow assertions for gadget verification
//   - Explicit syndrome data flow across gadget boundaries
//   - Real-time decoding primitives (submit/await)
//   - Success predicates and real-time decoding primitives
//
//===----------------------------------------------------------------------===//

#ifndef FABRIC_DIALECT_TD
#define FABRIC_DIALECT_TD

include "mlir/IR/OpBase.td"

def Fabric_Dialect : Dialect {
  let name = "fabric";
  let cppNamespace = "qlx::fabric";

  let summary = "Physical QEC fabric dialect";
  let description = [{
    Fabric — Fault-Tolerant Assembly of Blocks, Regions, Interconnects, and Codes.

    The Fabric dialect is a physical-level representation for fault-tolerant
    quantum computing on QEC code blocks (patches). It provides:

    - Composable gadgets with stabilizer flow declarations
    - Value-semantic patches and parameterized syndromes
    - Flow-aware syndrome propagation at gadget boundaries
    - Lattice surgery, magic state injection, and transport primitives
    - Pauli frame tracking

    Fabric sits between the logical instruction set (QLX) and the physical
    circuit level (Stim). A gadget in Fabric is a self-contained unit of
    QEC computation that declares how it transforms stabilizer bases,
    enabling algebraic composition of QEC operations.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // FABRIC_DIALECT_TD
