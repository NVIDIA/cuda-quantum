//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//

#ifndef QLX_LVM_DIALECT_TD
#define QLX_LVM_DIALECT_TD

include "mlir/IR/OpBase.td"

def LVM_Dialect : Dialect {
  let name = "lvm";
  let cppNamespace = "qlx::lvm";
  let summary = "QLX placed logical virtual machine dialect";
  let description = [{
    The LVM dialect owns P1: logical programs whose values are bound to an
    immutable logical machine structure and explicit placement descriptors,
    but not yet to physical topology, duration, noise, or cost.
  }];
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

#endif
