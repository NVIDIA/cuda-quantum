//===- QLXTypes.td - QLX type definitions ------------------*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the QLX dialect. These are the four core
// value types that flow through QLX programs:
//
//   !qlx.lqbit  - logical qubit handle
//   !qlx.region - region reference
//   !qlx.rs     - resource state promise handle
//   !qlx.lbit   - logical classical bit (decoded measurement result)
//
//===----------------------------------------------------------------------===//

#ifndef QLX_TYPES_TD
#define QLX_TYPES_TD

include "qlx/Dialect/QLX/IR/QLXDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// QLX Type base class
//===----------------------------------------------------------------------===//

class QLX_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QLX_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// QLX Types
//===----------------------------------------------------------------------===//

def QLX_LQBitType : QLX_Type<"LQBit", "lqbit"> {
  let summary = "logical qubit handle";
  let description = [{
    A handle to a logical qubit within a region. Logical qubits are
    region-qualified: each `!qlx.lqbit` value is bound to the region in
    which it was prepared (via `qlx.pz` or `qlx.pp`). Single-qubit
    operations require the qubit and its owning region as operands.

    Logical qubits are consumed by destructive measurements (`qlx.mz`,
    `qlx.mx`, `qlx.mp`) and by teleportation (`qlx.tp`). Code-switching
    (`qlx.csw`) consumes the source qubit and produces a new qubit in
    the destination region.
  }];
}

def QLX_RegionType : QLX_Type<"Region", "region"> {
  let summary = "region reference";
  let description = [{
    A reference to a logical region. Regions are the fundamental
    organizational unit in QLX: they represent a contiguous block of
    hardware resources running a single QEC code. Each region has a role
    (compute, memory, or factory), a code type, and a capacity
    (maximum number of logical qubits).

    Region references are declared at module level via `qlx.region_decl`
    and passed as arguments to `qlx.entry` functions.
  }];
}

def QLX_RSType : QLX_Type<"RS", "rs"> {
  let summary = "resource state promise handle";
  let description = [{
    A handle to a resource state promise. Resource states (e.g., T states,
    CCZ states, CS states) are produced by factory regions and consumed
    by `qlx.ars` (apply resource state) to implement non-Clifford gates.

    Resource state handles are also used to represent Bell pairs created
    by `qlx.lk` (link) for inter-region teleportation.

    The handle is a promise: the actual distillation or cultivation may
    proceed asynchronously, and the handle is resolved when consumed.
  }];
}

def QLX_LBitType : QLX_Type<"LBit", "lbit"> {
  let summary = "logical classical measurement result";
  let description = [{
    A logical classical bit representing the result of a logical
    measurement. Produced by measurement operations (`qlx.mz`, `qlx.mx`,
    `qlx.mp`, `qlx.mpp`) and by event testing (`qlx.te`).

    Logical bits can be used as conditions for branching (`qlx.br`).
    The selected QEC gadget owns how the logical outcome is produced from its
    encoded implementation; P0 carries only the logical result.
  }];
}

//===----------------------------------------------------------------------===//
// QLX unplaced logical types
//===----------------------------------------------------------------------===//

def QLX_LogicalQubitType : QLX_Type<"LogicalQubit", "logical_qubit"> {
  let summary = "one unplaced logical quantum degree of freedom";
  let description = [{
    The canonical QLX P0 quantum value.  It carries no machine, region,
    code, encoding, placement, timing, or cost information.  The type is
    linear: every transforming operation consumes one SSA owner and produces
    exactly one successor owner on every reachable path.

    `!qlx.lqbit` remains registered only for the alpha/migration IR.  New
    canonical P0 programs use `!qlx.logical_qubit`.
  }];
}

def QLX_LogicalResourceType : QLX_Type<"LogicalResource", "logical_resource"> {
  let summary = "one portable linear logical resource payload";
  let parameters = (ins StringRefParameter<"resource kind">:$kind);
  let assemblyFormat = "`<` $kind `>`";
}

def QLX_LogicalEventType : QLX_Type<"LogicalEvent", "logical_event"> {
  let summary = "typed portable asynchronous event";
  let parameters = (ins
    "mlir::Type":$payload,
    StringRefParameter<"payload ownership">:$ownership
  );
  let assemblyFormat = "`<` $payload `,` $ownership `>`";
}

def QLX_LogicalFrameType : QLX_Type<"LogicalFrame", "logical_frame"> {
  let summary = "portable symbolic logical-frame state";
  let parameters = (ins StringRefParameter<"frame domain">:$domain);
  let assemblyFormat = "`<` $domain `>`";
}

#endif // QLX_TYPES_TD
