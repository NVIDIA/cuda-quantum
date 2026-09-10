/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_PHYS_TYPES_TD
#define QLX_PHYS_TYPES_TD

include "qlx/Dialect/Phys/IR/PhysDialect.td"
include "qlx/Dialect/Event/IR/EventInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class Phys_Type<string name, string typeMnemonic>
    : TypeDef<Phys_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Phys_StateType : TypeDef<Phys_Dialect, "State", [
    DeclareTypeInterfaceMethods<Event_UniqueCarryOwnerInterface>
  ]> {
  let mnemonic = "state";
  let parameters = (ins "mlir::FlatSymbolRefAttr":$resource);
  let assemblyFormat = "`<` $resource `>`";
}

def Phys_RecordType : Phys_Type<"Record", "record"> {
  let parameters = (ins "mlir::FlatSymbolRefAttr":$schema);
  let assemblyFormat = "`<` $schema `>`";
}

def Phys_ResourcePayloadType : Phys_Type<"ResourcePayload", "resource_payload"> {
  let summary = "linear payload delivered by a physical resource provider";
  let parameters = (ins "mlir::FlatSymbolRefAttr":$kind);
  let assemblyFormat = "`<` $kind `>`";
}

def Phys_ResourceType : Phys_Type<"Resource", "resource"> {
  let parameters = (ins "mlir::FlatSymbolRefAttr":$declaration);
  let assemblyFormat = "`<` $declaration `>`";
}

def Phys_TokenType : Phys_Type<"Token", "token"> {
  let parameters = (ins StringRefParameter<"dependency edge kind">:$kind);
  let assemblyFormat = "`<` $kind `>`";
}

#endif
