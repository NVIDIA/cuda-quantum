// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit",
      count = 2 : i64,
      native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q1 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @events on @arch : () -> () {
    %q0 = phys.acquire [@q0] {event_id = "acquire0"}
      : !phys.state<@q0>
    %prepared0 = phys.prepare %q0 {event_id = "prepare0", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.barrier {domains = ["clock"], event_id = "tick"} : () -> ()
    %q1 = phys.acquire [@q1] {event_id = "acquire1"}
      : !phys.state<@q1>
    %prepared1 = phys.prepare %q1 {event_id = "prepare1", state = "zero"}
      : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.release %prepared0 {event_id = "release0"} : !phys.state<@q0>
    phys.release %prepared1 {event_id = "release1"} : !phys.state<@q1>
    phys.return
  }
  phys.schedule @events_schedule for @events {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64},
    tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire0|acquire|0|0|qubits[0]|deps=|data_deps=|resource_deps=|domain_deps=",
      "prepare0|prepare|0|1|qubits[0]|deps=acquire0|data_deps=acquire0|resource_deps=acquire0|domain_deps=",
      "tick|barrier|1|0|control:tick|deps=prepare0|data_deps=|resource_deps=|domain_deps=prepare0",
      "acquire1|acquire|1|0|qubits[1]|deps=tick|data_deps=|resource_deps=|domain_deps=tick",
      "prepare1|prepare|1|1|qubits[1]|deps=acquire1,tick|data_deps=acquire1|resource_deps=acquire1|domain_deps=tick",
      "release0|release|1|0|qubits[0]|deps=prepare0,tick|data_deps=prepare0|resource_deps=prepare0|domain_deps=tick",
      "release1|release|2|0|qubits[1]|deps=prepare1,tick|data_deps=prepare1|resource_deps=prepare1|domain_deps=tick"
    ],
    makespan_ns = 2.0 : f64
  }
}

// CHECK: phys.barrier {{.*}}domains = ["clock"]
// CHECK: "tick|barrier|1|0|control:tick|deps=prepare0|data_deps=|resource_deps=|domain_deps=prepare0"
// CHECK: "acquire1|acquire|1|0|qubits[1]|deps=tick|data_deps=|resource_deps=|domain_deps=tick"
