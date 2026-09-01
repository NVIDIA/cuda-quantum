/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef QLX_CONVERSION_PASSES_TD
#define QLX_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def QLXToLVM : Pass<"qlx-to-lvm", "mlir::ModuleOp"> {
  let summary = "Deterministically place one portable QLX program on an LVM domain";
  let description = [{
    Selects one `qlx.program` and one existing `lvm.domain`, assigns logical
    qubits by deterministic capacity-checked first fit, and emits a verified
    `lvm.kernel`. It fails closed for ambiguous roots, insufficient capacity,
    cross-space actions, unresolved calls, and unsupported operations.
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
