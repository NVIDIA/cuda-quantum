// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @events on @arch : () -> () {
    %q0 = phys.acquire [@q0] {event_id = "acquire0"}
      : !phys.state<@q0>
    %prepared0 = phys.prepare %q0 {event_id = "prepare0", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.barrier {domains = ["clock"], event_id = "tick"} : () -> ()
    phys.release %prepared0 {event_id = "release0"} : !phys.state<@q0>
    phys.return
  }
  // expected-error @+1 {{schedule event 'tick' domain_deps must exactly match its graph clock frontier}}
  phys.schedule @missing_predecessor for @events {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64},
    tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire0|acquire|0|0|qubits[0]|deps=|data_deps=|resource_deps=|domain_deps=",
      "prepare0|prepare|0|1|qubits[0]|deps=acquire0|data_deps=acquire0|resource_deps=acquire0|domain_deps=",
      "tick|barrier|0|0|control:tick|deps=|data_deps=|resource_deps=|domain_deps=",
      "release0|release|1|0|qubits[0]|deps=prepare0|data_deps=prepare0|resource_deps=prepare0|domain_deps="
    ],
    makespan_ns = 1.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @foreign_clock on @arch : () -> () {
    %q0 = phys.acquire [@q0] {event_id = "acquire0"}
      : !phys.state<@q0>
    "scf.execute_region"() ({
      phys.barrier {domains = ["clock"], event_id = "hidden_tick"} : () -> ()
      "scf.yield"() {event_id = "foreign_yield"} : () -> ()
    }) {event_id = "foreign"} : () -> ()
    %prepared0 = phys.prepare %q0 {event_id = "prepare0", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %prepared0 {event_id = "release0"} : !phys.state<@q0>
    phys.return
  }
  // expected-error @+1 {{graph event scheduling does not support region control scf.execute_region}}
  phys.schedule @foreign_clock_schedule for @foreign_clock {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64},
    tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire0|acquire|0|0|qubits[0]|deps=|data_deps=|resource_deps=|domain_deps=",
      "foreign|scf.execute_region|0|0|control:foreign|deps=|data_deps=|resource_deps=|domain_deps=",
      "hidden_tick|barrier|0|0|control:hidden_tick|deps=|data_deps=|resource_deps=|domain_deps=",
      "foreign_yield|scf.yield|0|0|control:foreign_yield|deps=|data_deps=|resource_deps=|domain_deps=",
      "prepare0|prepare|0|1|qubits[0]|deps=acquire0|data_deps=acquire0|resource_deps=acquire0|domain_deps=",
      "release0|release|1|0|qubits[0]|deps=prepare0|data_deps=prepare0|resource_deps=prepare0|domain_deps="
    ],
    makespan_ns = 1.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 2 : i64, native_actions = []
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
    phys.release %prepared0 {event_id = "release0"} : !phys.state<@q0>
    phys.release %q1 {event_id = "release1"} : !phys.state<@q1>
    phys.return
  }
  // expected-error @+1 {{schedule event 'acquire1' domain_deps must exactly match its graph clock frontier}}
  phys.schedule @missing_successor for @events {
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
      "acquire1|acquire|0|0|qubits[1]|deps=|data_deps=|resource_deps=|domain_deps=",
      "release0|release|1|0|qubits[0]|deps=prepare0,tick|data_deps=prepare0|resource_deps=prepare0|domain_deps=tick",
      "release1|release|0|0|qubits[1]|deps=acquire1|data_deps=acquire1|resource_deps=acquire1|domain_deps="
    ],
    makespan_ns = 1.0 : f64
  }
}
