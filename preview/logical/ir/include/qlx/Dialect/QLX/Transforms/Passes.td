//===- Passes.td - QLX dialect-local transform passes ------*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_DIALECT_QLX_TRANSFORMS_PASSES_TD
#define QLX_DIALECT_QLX_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// QLXEstimateLogical
//===----------------------------------------------------------------------===//

def QLXEstimateLogical
    : Pass<"qlx-estimate-logical", "::mlir::ModuleOp"> {
  let summary = "Produce a folded, device-free Tier-0 logical estimate";
  let description = [{
    Resolves one verified P0 `qlx.program`, follows its executable call graph,
    composes folded `qlx.repeat` multiplicities without unrolling, and emits a
    symbol-addressable `qlx.estimate_result`.  The result contains logical
    action/instrument counts, peak live logical owners, and the
    serial action-depth upper bound used by the stable Python LogicalProfile.

    The pass never selects a code or device. Missing or ambiguous roots,
    unresolved/recursive calls, inconsistent structured ownership, and count
    overflow fail the pass.
  }];
  let options = [
    Option<"rootSymbol", "root", "std::string", /*default=*/"\"\"",
           "Selected qlx.program symbol; required when several exist">,
    Option<"resultSymbol", "result", "std::string",
           /*default=*/"\"logical_estimate\"",
           "Symbol name for the emitted qlx.estimate_result">
  ];
  let dependentDialects = ["::qlx::QLXDialect"];
}

//===----------------------------------------------------------------------===//
// QLXSynthesizeRotations
//===----------------------------------------------------------------------===//

def QLXSynthesizeRotations
    : Pass<"qlx-synthesize-rotations", "::mlir::ModuleOp"> {
  let summary =
      "Synthesize logical rotations to positive H/S/T/CX";
  let description = [{
    Legalizes P0 logical rotations to the positive H/S/T/CX generator set.
    Finite built-in actions are decomposed exactly. Static Pauli-product
    rotations are synthesized to a target precision; rotations about X or
    Y are conjugated into the Z basis first. A rotation's authored
    `precision` attribute, when present, overrides this pass's `precision`
    option. Rotations with a runtime-dependent (non-static) angle are
    rejected rather than approximated.
  }];
  let options = [
    Option<"precision", "precision", "double",
           /*default=*/"1.0e-10",
           "Default off-lattice rotation synthesis precision; must be "
           "finite and in (0, 1)">
  ];
  let dependentDialects = ["::qlx::QLXDialect", "::mlir::arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// QLXVerifyCliffordT
//===----------------------------------------------------------------------===//

def QLXVerifyCliffordT
    : Pass<"qlx-verify-clifford-t", "::mlir::ModuleOp"> {
  let summary =
      "Verify every logical apply is legal in the positive H/S/T/CX basis";
  let description = [{
    Walks every `qlx.apply` and reports a diagnostic for any op whose
    built-in action is outside {h, s, t, cx, idle}.
  }];
  let dependentDialects = ["::qlx::QLXDialect"];
}

//===----------------------------------------------------------------------===//
// QLXToPBC
//===----------------------------------------------------------------------===//

def QLXToPBC : Pass<"qlx-to-pbc", "::mlir::ModuleOp"> {
  let summary =
      "Lower a synthesized Clifford+T program to Pauli-based-computation form";
  let description = [{
    Rewrites a Clifford+T program into Pauli-based-computation (PBC) form.
    Every Clifford (h/s/sdg/x/y/z/sx/cx/cz) is commuted to the end via one
    symplectic stabilizer frame, so the program becomes a sequence of pi/4
    Pauli-product rotations (one per T/T-dagger) followed by the terminal
    measurements conjugated into Pauli products.

    Scope: single-block programs with terminal measurements. State
    preparation is limited to |0>, |1>, |+>, |->; measurements to the Z and
    X bases. Non-Clifford gates other than T/T-dagger, mid-circuit
    measurement with feedforward, and control flow are reported as errors.
  }];
  let dependentDialects = ["::qlx::QLXDialect", "::mlir::arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// QLXVerifyPBC
//===----------------------------------------------------------------------===//

def QLXVerifyPBC : Pass<"qlx-verify-pbc", "::mlir::ModuleOp"> {
  let summary =
      "Certify a module is in Pauli-based-computation (PBC) normal form";
  let description = [{
    Certifies the contract `qlx-to-pbc` produces for downstream PBC
    lowering:
      1. the body contains only prepare / pi/4 pauli_rotation / mpp /
         discard / return / constant ops -- every Clifford has been
         absorbed;
      2. every pauli_rotation is exactly +/- pi/4;
      3. no pauli_rotation follows a measurement;
      4. the measured Pauli products pairwise commute (simultaneously
         measurable).
    On failure, emits one diagnostic naming the violated condition.
  }];
  let dependentDialects = ["::qlx::QLXDialect"];
}

#endif // QLX_DIALECT_QLX_TRANSFORMS_PASSES_TD
