// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: split-file %s %t
// RUN: qlx-opt %t/positive.mlir --phys-schedule='graph=pass_through result=pass_through_schedule' \
// RUN:   | FileCheck %s
// RUN: qlx-opt %t/tampered.mlir -verify-diagnostics

//--- positive.mlir

module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt attributes {
    metadata = {success_probability = "0.5",
                success_probability_evidence = "analysis:test"}
  } {}

  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = ["measure_z"]
    }
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit",
    resource_class = @qubits}
  phys.resource @q1 {index = 1 : i64, kind = "qubit",
    resource_class = @qubits}

  phys.graph @pass_through on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire0"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:3 = "phys.call"(%0#0, %0#1) <{
      callee = @attempt, event_id = "attempt0",
      instance = "attempt.pass_through", profile = @attempt_profile
    }> ({
    ^bb0(%active: !phys.state<@q0>, %pass_through: !phys.state<@q1>):
      %measured:2 = phys.measure @measure_z(%active) {
        event_id = "measure0", record_id = "attempt"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %pass_through, %measured#1
        : !phys.state<@q0>, !phys.state<@q1>,
          !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>,
         !phys.record<@attempt_bit>)
    %success = phys.condition %1#2 {event_id = "decision0"}
      : !phys.record<@attempt_bit> -> i1
    %2:2 = phys.retry %success carries (%1#0, %1#1) {
      attempt = @attempt, attempt_event = "attempt0",
      decision_event = "decision0", event_id = "retry0",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:test"
    } : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    phys.return %2#0, %2#1 : !phys.state<@q0>, !phys.state<@q1>
  }
}

// The call owns the state it actually changes, while the pass-through state
// retains acquire0 as its physical-resource producer at the retry frontier.
// CHECK: phys.schedule @pass_through_schedule
// CHECK-SAME: "retry0|retry|2|0|qubits[0],qubits[1]|deps=attempt0,decision0,acquire0|data_deps=attempt0,decision0|resource_deps=attempt0,acquire0|

//--- tampered.mlir

module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt attributes {
    metadata = {success_probability = "0.5",
                success_probability_evidence = "analysis:test"}
  } {}

  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = ["measure_z"]
    }
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit",
    resource_class = @qubits}
  phys.resource @q1 {index = 1 : i64, kind = "qubit",
    resource_class = @qubits}

  phys.graph @pass_through on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire0"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:3 = "phys.call"(%0#0, %0#1) <{
      callee = @attempt, event_id = "attempt0",
      instance = "attempt.pass_through", profile = @attempt_profile
    }> ({
    ^bb0(%active: !phys.state<@q0>, %pass_through: !phys.state<@q1>):
      %measured:2 = phys.measure @measure_z(%active) {
        event_id = "measure0", record_id = "attempt"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %pass_through, %measured#1
        : !phys.state<@q0>, !phys.state<@q1>,
          !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>,
         !phys.record<@attempt_bit>)
    %success = phys.condition %1#2 {event_id = "decision0"}
      : !phys.record<@attempt_bit> -> i1
    %2:2 = phys.retry %success carries (%1#0, %1#1) {
      attempt = @attempt, attempt_event = "attempt0",
      decision_event = "decision0", event_id = "retry0",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:test"
    } : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    phys.return %2#0, %2#1 : !phys.state<@q0>, !phys.state<@q1>
  }

  // q1 passes through attempt0 unchanged, so acquire0 remains one of retry0's
  // stable physical-resource predecessors.
  // expected-error @+1 {{schedule event 'retry0' resource_deps must exactly match the stable greedy-ASAP resource predecessors (scheduled=attempt0; expected=attempt0, acquire0)}}
  phys.schedule @tampered for @pass_through {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64}, tie_break = "stable_graph_order",
    optimization_status = "not_applicable",
    entries = [
      "acquire0|acquire|0|0|qubits[0],qubits[1]|deps=|data_deps=|resource_deps=|domain_deps=",
      "attempt0|call|0|1|qubits[0],qubits[1]|deps=acquire0|data_deps=acquire0|resource_deps=|domain_deps=|callee=attempt|instance=attempt.pass_through|profile=attempt_profile",
      "measure0|measure|0|1|qubits[0]|deps=acquire0|data_deps=acquire0|resource_deps=acquire0|domain_deps=|parent=attempt0|branch=body",
      "decision0|condition|1|1|control:decision0|deps=attempt0|data_deps=attempt0|resource_deps=|domain_deps=",
      "retry0|retry|2|0|qubits[0],qubits[1]|deps=attempt0,decision0|data_deps=attempt0,decision0|resource_deps=attempt0|domain_deps=|max_attempts=2|profile=attempt_profile|attempt=attempt|attempt_event=attempt0|decision_event=decision0|exhaustion=report_failure|success_probability=0.5|success_probability_source=attempt_profile|success_probability_evidence=analysis:test"
    ],
    makespan_ns = 2.0 : f64
  }
}
