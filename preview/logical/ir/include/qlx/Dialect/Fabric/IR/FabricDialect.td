//===- FabricDialect.td - Fabric dialect definition -------*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// Dialect definition for Fabric — a selected-QEC fabric dialect.
//
// Fabric describes composable gadgets on QEC code blocks with:
//   - Value-semantic patches and syndromes
//   - Explicit syndrome data flow across gadget boundaries
//   - Resource states, protocols, and static device bindings
//
//===----------------------------------------------------------------------===//

#ifndef FABRIC_DIALECT_TD
#define FABRIC_DIALECT_TD

include "mlir/IR/OpBase.td"

def Fabric_Dialect : Dialect {
  let name = "fabric";
  let cppNamespace = "qlx::fabric";

  let summary = "Selected-QEC fabric dialect";
  let description = [{
    Fabric — Fault-Tolerant Assembly of Blocks, Regions, Interconnects, and Codes.

    The Fabric dialect is a selected-QEC representation for fault-tolerant
    quantum computing on encoded code blocks (patches). It provides:

    - Composable gadgets and protocols
    - Value-semantic patches and parameterized syndromes
    - Lattice surgery, magic state injection, and transport primitives
    - Pauli frame tracking

    Fabric sits between the logical instruction set (QLX) and the physical
    circuit level (Stim). A gadget in Fabric is a self-contained unit of
    QEC computation that declares how it transforms stabilizer bases,
    enabling algebraic composition of QEC operations.
  }];

  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

#endif // FABRIC_DIALECT_TD
