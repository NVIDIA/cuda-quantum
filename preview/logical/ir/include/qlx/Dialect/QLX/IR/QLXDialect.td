/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// This file defines the QLX (Quantum Logical eXecution) dialect.
// QLX is a virtual ISA for fault-tolerant quantum computing, analogous to
// PTX for GPUs. It owns the machine-free P0 program stratum -- portable
// logical intent with no code, region, or device chosen -- plus the
// target-manifest and analysis-result strata that record what a
// compilation committed to and what it measured.
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

  let summary = "Quantum Logical eXecution virtual ISA dialect";
  let description = [{
    The QLX (Quantum Logical eXecution) dialect is a virtual ISA for
    fault-tolerant quantum computing. Its P0 program stratum defines a
    machine-free logical instruction set -- no code, region, or device is
    named anywhere in a `qlx.program` body -- encompassing:

    - Logical actions and instruments (Clifford gates, measurements, and
      user-declared objectives) applied to unplaced logical qubits
    - Typed asynchronous resource events for non-Clifford operations
    - Symbolic Pauli-frame tracking and portable statistical postselection
    - Structured control flow (`cflow.repeat`, `cflow.if`, `cflow.while`)

    Alongside P0, the dialect also hosts the target-manifest stratum
    (`qlx.device` and its layer bindings, recording which stack a
    compilation committed to) and the analysis-result stratum
    (`qlx.estimate_result`, the one interchange format for the whole
    resource-estimation ladder).

    QLX plays for fault-tolerant quantum computation the role that PTX plays
    for GPUs: it captures architecture-independent semantics while code- and
    device-specific backends -- reached through `lvm` (P1), `fabric` (P2),
    and `phys` (P3) -- implement them with appropriate gadgets, stabilizer
    extraction schedules.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    ::mlir::LogicalResult
    verifyOperationAttribute(::mlir::Operation *operation,
                             ::mlir::NamedAttribute attribute) override;
  }];
}

#endif // QLX_DIALECT_TD
