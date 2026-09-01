//===- QLXDialect.td - QLX dialect definition --------------*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// This file defines the QLX (Quantum Logical eXecution) dialect.
// This product registration owns machine-free P0 intent plus the common
// P0-P2 manifests and result records used by the compiler.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_DIALECT_TD
#define QLX_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// QLX Dialect
//===----------------------------------------------------------------------===//

def QLX_Dialect : Dialect {
  let name = "qlx";
  let cppNamespace = "qlx";

  let summary = "P0-P2 Quantum Logical eXecution product dialect";
  let description = [{
    The registered product surface contains machine-free P0 programs,
    objective declarations and structured logical control, together with the
    immutable device, lowering, experiment, target, and estimate manifests
    required to carry verified work through P1 and P2. Physical regions,
    timing, runtime events, asynchronous resource promises, routes, and
    execution operations are deliberately not registered.
  }];

  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

#endif // QLX_DIALECT_TD
