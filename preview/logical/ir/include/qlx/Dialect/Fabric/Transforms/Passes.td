//===- Passes.td - Fabric dialect-local transform passes ---*- tablegen -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_DIALECT_FABRIC_TRANSFORMS_PASSES_TD
#define QLX_DIALECT_FABRIC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FabricLowerProtocols (P2N backend legalization)
//===----------------------------------------------------------------------===//

def FabricLowerProtocols : Pass<"fabric-lower-protocols", "::mlir::ModuleOp"> {
  let summary =
      "Legalize a folded QLX protocol graph for gadget-only backends";
  let description = [{
    Reifies each `fabric.protocol` as a semantically equivalent
    `fabric.gadget`, replaces `fabric.protocol_return` with `fabric.return`,
    and marks the requested root as the backend entry point. Calls remain
    folded and keep their symbol names, so existing gadget-inlining backends
    can consume P2N without embedding QLX selection policy.

    This is a representation legalization, not QEC selection or protocol
    synthesis. The source module is expected to have already selected every
    implementation. A synthetic device symbol is attached only because the
    legacy entry-gadget ABI requires one; device semantics remain in the
    linked QLX device/architecture declarations.

    Because a QLX build retains earlier profiles for evidence and replay,
    the pass also projects the cloned target module to the backend closure:
    codes, optional legacy device/region declarations, and gadgets.
    It never mutates the caller's build.
  }];
  let options = [
    Option<"rootSymbol", "root-symbol", "std::string",
           /*default=*/"\"\"",
           "Root fabric.protocol or fabric.gadget symbol to mark as entry">
  ];
  let dependentDialects = ["::qlx::fabric::FabricDialect"];
}

//===----------------------------------------------------------------------===//
// FabricInlineRealisations
//===----------------------------------------------------------------------===//

def FabricInlineRealisations
    : Pass<"fabric-inline-realisations", "::mlir::ModuleOp"> {
  let summary = "Inline referenced Fabric realization circuits into gadgets";
  let description = [{
    Clones each referenced `fabric.circuit` body into the signature-compatible
    `fabric.gadget` that names it through `realization`, remapping block
    arguments and preserving the gadget specification/objective attributes.
    The source circuit remains available for other gadgets and provenance.
  }];
  let dependentDialects = ["::qlx::fabric::FabricDialect"];
}

//===----------------------------------------------------------------------===//
// FabricCount (Analysis / annotation)
//===----------------------------------------------------------------------===//

def FabricCount : Pass<"fabric-count", "::mlir::ModuleOp"> {
  let summary =
      "Static counter: emit a typed per-region/per-protocol result";
  let description = [{
    Tier 1 of the resource-estimation stack. Walks a Fabric module and
    emits a versioned `qlx.estimate_result` summarizing logical-qubit, gate,
    syndrome-round, success-predicate, and protocol counts when `device=` identifies the
    selected immutable closure. During migration it also mirrors the data in
    the legacy module `fabric.counts` attribute; omitting `device=` produces
    only that compatibility annotation, not a typed Tier-1 result. Sub-second;
    does not model physical events, schedules, or wall-clock time.

    Output schema (top level):
    ```
    fabric.counts = {
      patches_peak = i64,
      logical_qubits_peak = i64,
      per_region = {<region_name> = {
        role, code, patches,
        gate_counts = {<gate> = i64, ...},
        rounds_by_kind = {syndrome = i64, idle = i64, ...},
        inject_count, transport_in, transport_out,
      }, ...},
      per_protocol = {<name> = {
        kind = "production" | "transport" | "injection",
        op_count = i64,
      }, ...},
      transversal_edges = [[src_region, dst_region], ...],
    }
    ```

    Walk semantics:
      - `fabric.gadget` is `IsolatedFromAbove`. The pass walks only the
        entry gadget at the module level; non-entry gadgets are reached
        through `fabric.call` and their counts accumulate at each call
        site (lazy inlining).
      - `fabric.repeat`: gate counts inside the body multiply by `count`;
        nested repeats compose multiplicatively.
      - `fabric.idle`: `rounds` adds to `rounds_by_kind.idle`; not a gate.
      - `fabric.if`: statically counts both branches as an upper bound,
        suitable for a structural estimate where only one branch executes.
      - `fabric.transport`: counted in both src and dst regions
        (`transport_out` / `transport_in`) and once under `per_protocol`.
      - `logical_qubits_peak`: maximum live logical qubits under the folded
        structural walk, weighting each patch by its resolved code `k`;
        repeat-local allocations are reused per iteration.
      - `patches_peak`: maximum live encoded patch owners under the same walk,
        kept separate from protected logical capacity for `k != 1` codes.
      - Protocol attributes resolve from `FlatSymbolRefAttr` (a
        `fabric.gadget` symbol) or `#fabric.spec_only<"name">`; kind
        derives from the owning op.
  }];
  let dependentDialects = [
    "::qlx::fabric::FabricDialect",
    "::qlx::QLXDialect"
  ];
  let options = [
    Option<"rootSymbol", "root", "std::string", /*default=*/"\"\"",
           "Selected entry fabric.gadget; inferred only when exactly one entry exists">,
    Option<"deviceSymbol", "device", "std::string", /*default=*/"\"\"",
           "Selected qlx.device required for typed Tier-1 result emission; omitted only for the legacy fabric.counts annotation">,
    Option<"resultSymbol", "result", "std::string",
           /*default=*/"\"static_estimate\"",
           "Symbol name for the emitted qlx.estimate_result">
  ];
}

#endif // QLX_DIALECT_FABRIC_TRANSFORMS_PASSES_TD
