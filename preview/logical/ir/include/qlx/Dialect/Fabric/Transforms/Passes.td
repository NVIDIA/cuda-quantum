/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_FABRIC_TRANSFORMS_PASSES_TD
#define QLX_DIALECT_FABRIC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FabricToPhys (P2 -> P3 progressive physical projection)
//===----------------------------------------------------------------------===//

def FabricToPhys : Pass<"fabric-to-phys", "::mlir::ModuleOp"> {
  let summary = "Project a selected Fabric call graph to native physical IR";
  let description = [{
    Consumes one fully selected P2 gadget/protocol closure and the linked
    physical device stack from the same live ModuleOp. The pass performs a
    read-only, parallelizable preflight over reachable callable bodies before
    making any mutation, then commits physical resources and a `phys.graph` in
    deterministic source order.

    The initial native contract supports patch allocation, X/Z preparation,
    release, and folded patch-only calls. Unsupported operations fail before
    graph emission. Later slices extend this same pass; Python remains
    responsible for device/provider policy and never serializes the module as
    an internal interchange.
  }];
  let options = [
    Option<"rootSymbol", "root-symbol", "std::string", /*default=*/"\"\"",
           "Selected fabric.protocol or fabric.gadget root">,
    Option<"rootProfileSymbol", "root-profile-symbol", "std::string",
           /*default=*/"\"\"",
           "Optional selected gadget profile attached to the root">,
    Option<"deviceSymbol", "device-symbol", "std::string", /*default=*/"\"\"",
           "Selected qlx.device owning the physical machine">,
    Option<"graphSymbol", "graph-symbol", "std::string", /*default=*/"\"\"",
           "Output phys.graph symbol; defaults to <root>_physical">,
    Option<"parallelPlanning", "parallel-planning", "bool",
           /*default=*/"true", "Parallelize independent read-only preflight">,
    Option<"preflightOnly", "preflight-only", "bool",
           /*default=*/"false", "Validate support without mutating the module">,
    Option<"deferOutputVerification", "defer-output-verification", "bool",
           /*default=*/"false",
           "Defer the full module verification to the progressive Build boundary">
  ];
  let dependentDialects = [
    "::qlx::fabric::FabricDialect",
    "::qlx::phys::PhysDialect",
    "::qlx::QLXDialect"
  ];
}

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
    round, injection, transport, and protocol counts when `device=` identifies the
    selected immutable closure. During migration it also mirrors the data in
    the legacy module `fabric.counts` attribute; omitting `device=` produces
    only that compatibility annotation, not a typed Tier-1 result. Sub-second;
    does not model noise, schedule, or wall-clock time. See
    `design/resource-estimation-proposal.md` §4.1.

    Output schema (top level):
    ```
    fabric.counts = {
      logical_qubits_peak = i64,
      per_region = {<region_name> = {
        role, code, distance, patches,
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
      - `cflow.repeat`: gate counts inside the body multiply by `count`;
        nested repeats compose multiplicatively.
      - `fabric.idle`: `rounds` adds to `rounds_by_kind.idle`; not a gate.
      - `cflow.if`: statically counts both branches as an upper bound,
        suitable for feed-forward schedules where only one branch executes.
      - `fabric.transport`: counted in both src and dst regions
        (`transport_out` / `transport_in`) and once under `per_protocol`.
      - `logical_qubits_peak`: maximum live allocations under the folded
        structural walk; repeat-local allocations are reused per iteration.
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

//===----------------------------------------------------------------------===//
// FabricEstimateAnalytical
//===----------------------------------------------------------------------===//

def FabricEstimateAnalytical
    : Pass<"fabric-estimate-analytical", "::mlir::ModuleOp"> {
  let summary = "Compose Tier-1 Fabric counts with an analytical QEC model";
  let description = [{
    Consumes one exact `qlx.estimate_result` from `fabric-count`, resolves the
    selected device, refinement, physical-machine, operating-point, and code
    evidence retained in the P2 module, and emits a versioned Tier-2
    analytical result. Probability aggregation uses stable survival arithmetic.
    The supplied Tier-1 data is recomputed from the selected executable/device
    closure and must match exactly; producer labels alone are not trusted.
    Strict distance mode requires an exact or lower-bound `fabric.code_profile`
    with a matching distance claim and nonempty evidence, referenced from the
    code metadata; free-form provenance strings are not authenticated evidence.
    Physical capacity is restricted to the carriers and simultaneous-patch
    capacity exposed by each active binding's `phys.patch_topology`.
    Bindings that share one physical resource fail closed until Tier 1 carries
    resource-qualified joint-liveness evidence.
    Native v1 supports the closed code-scaling model plus deterministic,
    guaranteed compact factory models. Folded resource demand is authenticated
    against the selected stream, producer metadata, logical factory capacity,
    QEC-to-physical model, dedicated physical footprint, and operating
    point. Retry and probabilistic production semantics remain fail-closed.
    Missing or inconsistent closure never produces a partial result.
  }];
  let options = [
    Option<"rootSymbol", "root", "std::string", /*default=*/"\"\"",
           "Selected fabric.gadget; defaults to the Tier-1 result root">,
    Option<"countsSymbol", "counts", "std::string",
           /*default=*/"\"static_estimate\"",
           "Tier-1 qlx.estimate_result symbol">,
    Option<"resultSymbol", "result", "std::string",
           /*default=*/"\"analytical_estimate\"",
           "Tier-2 qlx.estimate_result symbol">,
    Option<"deviceSymbol", "device", "std::string", /*default=*/"\"\"",
           "Optional explicitly selected qlx.device symbol">,
    Option<"physicalError", "p-phys", "double", /*default=*/"-1.0",
           "Required physical error probability in [0,1]">,
    Option<"failureBudget", "failure-budget", "double", /*default=*/"-1.0",
           "Required total failure budget in (0,1]">,
    Option<"cycleTime", "cycle-time", "double", /*default=*/"1.0",
           "Positive time per logical fault site">,
    Option<"scalingPrefactor", "scaling-prefactor", "double",
           /*default=*/"0.1", "Nonnegative logical-error scaling prefactor">,
    Option<"scalingThreshold", "scaling-threshold", "double",
           /*default=*/"0.01", "Positive logical-error scaling threshold">,
    Option<"requireEstablished", "require-established-distance", "bool",
           /*default=*/"true", "Reject unestablished distance evidence">
  ];
  let dependentDialects = [
    "::qlx::fabric::FabricDialect",
    "::qlx::QLXDialect"
  ];
}

#endif // QLX_DIALECT_FABRIC_TRANSFORMS_PASSES_TD
