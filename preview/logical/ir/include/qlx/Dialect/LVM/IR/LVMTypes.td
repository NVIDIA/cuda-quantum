/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_LVM_TYPES_TD
#define QLX_LVM_TYPES_TD

include "qlx/Dialect/LVM/IR/LVMDialect.td"
include "mlir/IR/AttrTypeBase.td"

class LVM_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LVM_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def LVM_LogicalQubitType : LVM_Type<"LogicalQubit", "logical_qubit"> {
  let summary = "logical qubit bound to a logical space or advanced placement";
  let parameters = (ins "::mlir::SymbolRefAttr":$placement);
  let assemblyFormat = "`<` $placement `>`";
}

def LVM_LogicalResourceType : LVM_Type<"LogicalResource", "logical_resource"> {
  let summary = "linear logical resource bound to one machine stream";
  let parameters = (ins
    StringRefParameter<"resource kind">:$kind,
    "mlir::SymbolRefAttr":$stream
  );
  let assemblyFormat = "`<` $kind `,` $stream `>`";
}

def LVM_LogicalFrameType : LVM_Type<"LogicalFrame", "logical_frame"> {
  let summary = "placed-kernel symbolic logical-frame state";
  let parameters = (ins StringRefParameter<"frame domain">:$domain);
  let assemblyFormat = "`<` $domain `>`";
}

#endif
