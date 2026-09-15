// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

// A portable row can be syntactically complete yet forge one partition of its
// causal claim.  ScheduleOp parses the rows, then the common independent
// semantic core reconstructs the physical-resource frontier and fails closed.
module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () {
    %state = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %delayed = phys.delay %state {
      duration_ns = 2.0 : f64, event_id = "delay"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %delayed {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
  // expected-error @+1 {{schedule event 'delay' resource_deps must exactly match the stable greedy-ASAP resource predecessors (scheduled=; expected=acquire)}}
  phys.schedule @tampered for @g {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order",
    optimization_status = "not_applicable",
    entries = [
      "acquire|acquire|0|0|qubits[0]|deps=|data_deps=|resource_deps=|domain_deps=",
      "delay|delay|0|2|qubits[0]|deps=acquire|data_deps=acquire|resource_deps=|domain_deps=",
      "release|release|2|0|qubits[0]|deps=delay|data_deps=delay|resource_deps=delay|domain_deps="
    ],
    makespan_ns = 2.0 : f64
  }
}
