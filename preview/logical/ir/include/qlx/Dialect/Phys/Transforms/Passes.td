/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

//===- Passes.td - Phys dialect transform passes ---------------*- tablegen -*-===//

#ifndef QLX_DIALECT_PHYS_TRANSFORMS_PASSES_TD
#define QLX_DIALECT_PHYS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def PhysSchedule : Pass<"phys-schedule", "::mlir::ModuleOp"> {
  let summary = "Construct one deterministic greedy-ASAP P3 schedule";
  let description = [{
    Schedules one verified `phys.graph` directly in native code. The pass
    preserves structured calls, shared call templates, repeats, branches, and
    bounded retry policy while deriving stable physical-resource, SSA, and
    clock-domain dependencies. It never reconstructs physical intent from P2
    and never prints or parses MLIR.
  }];
  let options = [
    Option<"graphSymbol", "graph", "std::string", /*default=*/"\"\"",
           "Selected phys.graph; inferred only when exactly one exists">,
    Option<"resultSymbol", "result", "std::string", /*default=*/"\"\"",
           "Output phys.schedule symbol; defaults to <graph>_schedule">,
    Option<"deferOutputVerification", "defer-output-verification", "bool",
           /*default=*/"false",
           "Defer whole-module verification to an immediate immutable Build boundary">
  ];
  let dependentDialects = ["::qlx::phys::PhysDialect"];
}

def PhysEstimateSchedule
    : Pass<"phys-estimate-schedule", "::mlir::ModuleOp"> {
  let summary = "Summarize one verified P3 physical schedule";
  let description = [{
    Produces the Tier-3 schedule-aware resource estimate directly from one
    `phys.schedule`. Compact event counts remain structural, while folded
    repeat and bounded-while ancestry scale leaf duration and resource
    occupancy. Mutually exclusive branches use the worst executable path.
    Bounded retries replay their verifier-authenticated causal slice and report
    first-attempt, expected, and maximum time/resource/qubit evidence. The
    matching P3 device is inferred from the graph architecture. Its native
    analytical lower tier is recomputed from the retained static, device, and
    model inputs and must match exactly; schema/producer labels are insufficient
    provenance.
    The pass does not construct or reconstruct a schedule from P2.
  }];
  let options = [
    Option<"scheduleSymbol", "schedule", "std::string", /*default=*/"\"\"",
           "Selected phys.schedule; inferred only when exactly one exists">,
    Option<"resultSymbol", "result", "std::string",
           /*default=*/"\"schedule_estimate\"",
           "Tier-3 qlx.estimate_result symbol">,
    Option<"lowerTierSymbol", "lower-tier", "std::string",
           /*default=*/"\"\"",
           "Required exact analytical qlx.estimate_result refined by this schedule">,
    Option<"terminationMode", "termination", "std::string",
           /*default=*/"\"program\"",
           "Termination costing: program or full_workload">
  ];
  let dependentDialects = ["::qlx::fabric::FabricDialect",
                           "::qlx::phys::PhysDialect", "::qlx::QLXDialect"];
}

#endif // QLX_DIALECT_PHYS_TRANSFORMS_PASSES_TD
