/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_CONVERSION_QUAKETOQLX_PASSES_TD
#define QLX_CONVERSION_QUAKETOQLX_PASSES_TD

include "mlir/Pass/PassBase.td"

def ExpandQuakeRegisterTraversals :
    Pass<"expand-quake-register-traversals", "mlir::ModuleOp"> {
  let summary = "Expand Quake register traversals that block wire conversion";
  let description = [{
    Directly expands normalized, statically bounded `cc.loop` register
    traversals whose dynamic `quake.extract_ref` or `quake.subveq` operations
    prevent conversion to linear `!quake.wire` values. This pass runs before
    CUDA-Q's linear-values preparation; it is separate from QLX's post-linear
    `prepare-quake-for-qlx` boundary-cleanup pipeline.

    The fast path accepts only single-block precondition/body/step loops with a
    pure integer comparison, an unchanged body induction, a positive constant
    step proven not to wrap, no break, and no Python else. It clones the body
    and step as straight-line SSA, eagerly folds induction arithmetic, and
    materializes static accesses without a temporary CFG or global rewrite
    worklist.

    Loops that do not block wire conversion remain structured. A remaining
    wire-blocking loop fails closed with a diagnostic identifying an unsupported
    shape or exceeded expansion limit. Rewriting is transactional: failure
    leaves the input module unchanged. The pass creates no QLX operations and
    derives no P0 or machine facts.
  }];
  let options = [
    Option<"maximumIterations", "maximum-iterations", "unsigned",
           /*default=*/"4096", "Maximum iterations in one expanded loop">,
    Option<"maximumGeneratedOperations", "maximum-generated-operations",
           "uint64_t", /*default=*/"1000000",
           "Maximum estimated operations generated across the module">
  ];
}

def PruneDeadCCLoopCarries :
    Pass<"prune-dead-cc-loop-carries", "mlir::ModuleOp"> {
  let summary = "Remove dead classical carries from supported CUDA-Q CC loops";
  let description = [{
    Prepares CUDA-Q CC loops with cc.undef initial carries for Quake-to-P0
    conversion without duplicating MLIR's dead-value analysis. The pass runs
    MLIR's remove-dead-values pass on each affected function, then applies
    MLIR's generic region-branch canonicalization patterns to cc.loop. This
    supplies the generic cleanup that CUDA-Q's current cc.loop canonicalization
    hook does not register, while leaving unrelated functions untouched.

    Affected functions must not retain cc.scope. CUDA-Q canonicalizes that op
    through the CF dialect; this focused preparation deliberately does not
    register or introduce CF.

    Cleanup is restricted to the structured loop subset accepted by the QLX
    importer: pre-condition, single-block while/do/step regions with no break
    or Python else. Other loop forms fail closed before the module is changed.
    This pass removes only carries proven dead; live classical carries remain
    and may subsequently be rejected by convert-quake-to-qlx.
  }];
}

def ConvertQuakeToQLX : Pass<"convert-quake-to-qlx", "mlir::ModuleOp"> {
  let summary = "Import value-semantics CUDA-Q Quake as QLX P0";
  let description = [{
    Converts CUDA-Q entry-point functions in linear `!quake.wire` form into
    machine-free `qlx.program` P0 roots. The supported contract covers:
      - null-wire allocation; H/S/Sdg/T/Tdg/X/Y/Z; CX/CZ;
      - T/Tdg as standard logical actions;
      - two-control Z as the standard CCZ action, and two-control X as the
        exact standard-action H-CCZ-H decomposition;
      - swap (lowered to three CX);
      - single-qubit rotations r1/rx/ry/rz (lowered to a `pauli_rotation`
        apply with the axis in `{x_mask,z_mask,sign}`; uncontrolled,
        constant-angle only; `<adj>` negates the angle; uncontrolled R1 and Rz
        are equivalent up to global phase);
      - X/Y/Z measurement, discrimination, sink, and return;
      - `cc.scope` (transparent) and a normalized constant-trip `cc.loop`,
        folded to `cflow.repeat` (never unrolled) with `!quake.wire` carries;
        repeat bodies may prepare iteration-local qubits when they are
        destructively measured or discarded before the iteration yields;
      - `cc.if` with a constant condition (the selected branch is inlined,
        feedback-free) or a measurement-derived (`quake.discriminate`) condition
        (lowered to an adaptive `cflow.if`).

    All quantum operands/results must already have type `!quake.wire`; reference,
    vector, control, and cable types are rejected. Loops must be constant-trip
    counted loops with induction initialized to 0, incremented by 1, and tested
    against a nonnegative literal bound using either `arith.cmpi ne` or forward
    signed `arith.cmpi slt`. Affine-index, dynamic-bound, or otherwise
    unsupported constructs fail closed with an operation-specific diagnostic.

    The input is expected to have passed through CUDA-Q's
    `convert-to-linear-values` preparation, along with any required entry-point
    specialization, quantum-call inlining, control expansion, and gate
    decomposition. Neither this conversion pass nor QLX's
    `prepare-quake-for-qlx` boundary-cleanup pipeline runs that CUDA-Q pipeline.
  }];
  let options = [
    Option<"entryPoint", "entry-point", "std::string",
           /*default=*/"\"\"",
           "Source entry point to import; abbreviated names are accepted only "
           "when they identify exactly one entry">
  ];
  let dependentDialects = [
    "qlx::QLXDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // QLX_CONVERSION_QUAKETOQLX_PASSES_TD
