/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_CONVERSION_PASSES_TD
#define QLX_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// QLXToLVM
//===----------------------------------------------------------------------===//

def QLXToLVM : Pass<"qlx-to-lvm", "mlir::ModuleOp"> {
  let summary = "Deterministically place one portable QLX program on an LVM domain";
  let description = [{
    Implements the hardened native P0-to-P1 path.  The pass selects one
    `qlx.program` and one existing `lvm.domain`, assigns every logical qubit
    to logical spaces with deterministic capacity-checked first-fit, and emits
    a verified `lvm.kernel`.  It never invents machine facts and fails closed
    for ambiguous roots, insufficient capacity, cross-space actions, unresolved
    calls, and unsupported operations.

    The Python `qlx.place(...)` API remains the opt-in research route for
    custom constraints, solvers, nonlocal bindings, and experimental policies;
    both routes produce the same P1 `lvm.kernel` contract.
  }];
  let options = [
    Option<"archModule", "arch-module", "std::string", /*default=*/"\"\"",
           "Path to an MLIR module containing the selected lvm.domain">,
    Option<"rootSymbol", "root", "std::string", /*default=*/"\"\"",
           "Portable qlx.program symbol; inferred only when exactly one exists">,
    Option<"domainSymbol", "domain", "std::string", /*default=*/"\"\"",
           "Logical lvm.domain symbol; inferred only when exactly one exists">,
    Option<"resultSymbol", "result", "std::string", /*default=*/"\"\"",
           "P1 lvm.kernel symbol (defaults to <root>_placed)">,
    Option<"witnessOutput", "witness-output", "std::string", /*default=*/"\"\"",
           "Optional path for the detached canonical placement witness JSON">
  ];
  let dependentDialects = [
    "qlx::QLXDialect",
    "qlx::lvm::LVMDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // QLX_CONVERSION_PASSES_TD
