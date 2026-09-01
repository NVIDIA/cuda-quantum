//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//

#ifndef QLX_LVM_ATTRS_TD
#define QLX_LVM_ATTRS_TD

include "qlx/Dialect/LVM/IR/LVMDialect.td"
include "mlir/IR/AttrTypeBase.td"

def LVM_CapabilityAttr : AttrDef<LVM_Dialect, "Capability"> {
  let mnemonic = "capability";
  let summary = "open qualified logical-machine capability key";
  let parameters = (ins StringRefParameter<"qualified capability key">:$key);
  let assemblyFormat = "`<` $key `>`";
}

#endif
