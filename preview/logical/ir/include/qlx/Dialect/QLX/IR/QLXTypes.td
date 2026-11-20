/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef QLX_TYPES_TD
#define QLX_TYPES_TD

include "qlx/Dialect/QLX/IR/QLXDialect.td"
include "cudaq/Optimizer/Dialect/LinearOwnershipInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// QLX Type base class
//===----------------------------------------------------------------------===//

class QLX_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QLX_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// QLX unplaced logical types
//===----------------------------------------------------------------------===//

def QLX_LogicalQubitType : QLX_Type<"LogicalQubit", "logical_qubit",
    LinearTypeTraits<>.traits> {
  let summary = "one unplaced logical quantum degree of freedom";
  let description = [{
    The canonical QLX P0 quantum value.  It carries no machine, region,
    code, encoding, placement, timing, or cost information.  The type is
    linear, and its values are in SSI form: every value has exactly one use.
    Every transforming operation consumes one value and produces exactly one
    successor value on every reachable path.  The type implements
    `LinearTypeInterface`, and the `qlx.program` and `qlx.objective_body`
    verifiers reject a value that is consumed twice, captured into a loop,
    or never consumed.
  }];
}

def QLX_LogicalResourceType : QLX_Type<"LogicalResource", "logical_resource",
    LinearTypeTraits<>.traits> {
  let summary = "one portable linear logical resource payload";
  let parameters = (ins StringRefParameter<"resource kind">:$kind);
  let assemblyFormat = "`<` $kind `>`";
}

def QLX_LogicalFrameType : QLX_Type<"LogicalFrame", "logical_frame",
    LinearTypeTraits<["mayDrop"]>.traits> {
  let summary = "portable symbolic logical-frame state";
  let parameters = (ins StringRefParameter<"frame domain">:$domain);
  let assemblyFormat = "`<` $domain `>`";
}

#endif // QLX_TYPES_TD
