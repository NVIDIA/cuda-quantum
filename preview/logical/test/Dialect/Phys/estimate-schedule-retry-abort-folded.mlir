// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=folded result=folded_schedule' \
// RUN:   --phys-estimate-schedule='schedule=folded_schedule lower-tier=analytical result=folded_estimate' | \
// RUN:   FileCheck %s
// RUN: qlx-opt %s --phys-schedule='graph=folded result=folded_schedule' \
// RUN:   --phys-estimate-schedule='schedule=folded_schedule lower-tier=analytical result=folded_estimate termination=full_workload' | \
// RUN:   FileCheck %s --check-prefix=FULL-WORKLOAD
// RUN: not qlx-opt %s --phys-schedule='graph=folded result=folded_schedule' \
// RUN:   --phys-estimate-schedule='schedule=folded_schedule lower-tier=analytical result=folded_estimate termination=unknown' 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD-TERMINATION
// RUN: qlx-opt %s --phys-schedule='graph=template_folded result=template_folded_schedule' \
// RUN:   --phys-estimate-schedule='schedule=template_folded_schedule lower-tier=analytical result=template_folded_estimate' | \
// RUN:   FileCheck %s --check-prefix=TEMPLATE-FOLDED
// RUN: sed 's/max_attempts = 1 : i64/max_attempts = 2 : i64/g' %s | \
// RUN:   qlx-opt --phys-schedule='graph=template_folded result=template_folded_schedule' \
// RUN:   --phys-estimate-schedule='schedule=template_folded_schedule lower-tier=analytical result=template_folded_estimate' | \
// RUN:   FileCheck %s --check-prefix=TEMPLATE-REPLAY
// RUN: qlx-opt %s --phys-schedule='graph=bounded_while result=bounded_while_schedule' \
// RUN:   --phys-estimate-schedule='schedule=bounded_while_schedule lower-tier=analytical result=bounded_while_estimate' | \
// RUN:   FileCheck %s --check-prefix=WHILE

module attributes {qlx.profiles = ["p3"]} {
  lvm.domain @estimate_logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @estimate_code {
    distance = 1 : i64,
    metadata = {distance_method = "fixture", distance_provenance = @estimate_code_evidence},
    partitions = {data = 1 : i64}
  }
  fabric.code_profile @estimate_code_evidence {
    code = @estimate_code, distance_claim = 1 : i64,
    distance_status = "exact", evidence = ["schedule-estimate-fixture@1"]
  }
  fabric.machine @estimate_qec {
    fabric.region @compute {
      code = @estimate_code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  fabric.gadget @estimate_source() { fabric.return }
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget @outer() {
    fabric.return
  }
  fabric.gadget_profile @attempt_profile for @attempt attributes {
    metadata = {
      success_probability = "0.5",
      success_probability_evidence = "analysis:abort-fold-test"
    }
  } {}

  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = ["measure_z"]
    }
    phys.qec_binding @estimate_binding {
      qec_region = @estimate_qec::@compute, resources = [@qubits]
    }
  }
  qlx.logical_to_qec @estimate_logical_to_qec {
    logical = @estimate_logical, qec = @estimate_qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @estimate_qec_to_physical {
    qec = @estimate_qec, physical = @arch,
    entries = [{qec = "compute", binding = "estimate_binding",
                resources = ["qubits"]}]
  }
  qlx.device @estimate_device {
    logical = @estimate_logical, qec = @estimate_qec, physical = @arch,
    logical_to_qec = @estimate_logical_to_qec,
    qec_to_physical = @estimate_qec_to_physical
  }
  qlx.estimate_result @static {
    assumptions = [], data = {}, device = @estimate_device,
    evidence = [@estimate_source], root = @estimate_source,
    metadata = {producer = "fixture", producer_version = "1"},
    schema = "qlx.fabric-counts/v1", tier = "static"
  }
  qlx.estimate_result @analytical {
    assumptions = [], data = {}, device = @estimate_device,
    evidence = [@estimate_source], lower_tier = @static,
    root = @estimate_source,
    metadata = {producer = "fixture", producer_version = "1"},
    schema = "qlx.fabric-estimate/v1", tier = "analytical"
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }

  // A trillion dynamic iterations remain one compact repeat.  Each iteration
  // reaches the next with probability 1/2, so the expected number visited is
  // the bounded geometric sum approaching two.  M=1 adds no replay time, but
  // abort still truncates the unconditional expectation below the static
  // first-attempt schedule.
  phys.graph @folded on @arch : () -> !phys.state<@q0>
      attributes {source_protocol = @estimate_source} {
    %q = phys.acquire [@q0] {event_id = "folded.acquire"}
      : !phys.state<@q0>
    %result = "cflow.repeat"(%q) <{
      count = 1000000000000 : i64, event_id = "folded.repeat"
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %attempted:2 = "phys.call"(%current) <{
        callee = @attempt, event_id = "folded.attempt",
        instance = "folded.attempt", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q0>):
        %measured:2 = phys.measure @measure_z(%attempt_state) {
          event_id = "folded.measure", record_id = "folded"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {
        event_id = "folded.decision"
      } : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "folded.attempt",
        decision_event = "folded.decision", event_id = "folded.retry",
        exhaustion = "abort", max_attempts = 1 : i64,
        profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:abort-fold-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %retried : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %result : !phys.state<@q0>
  }

  // The template occurrence and its canonical call both remain inside the
  // compact repeat.  Each has continuation probability 1/2, so one iteration
  // has continuation 1/4, expected elapsed 2 + (1/2)*2 = 3 ns, and expected
  // active time 1 + (1/2)*1 = 1.5 ns.  The trillion-fold geometric visit sum
  // approaches 4/3 without materializing any occurrence.
  phys.graph @template_folded on @arch : () -> !phys.state<@q0>
      attributes {source_protocol = @estimate_source} {
    %q = phys.acquire [@q0] {event_id = "template_folded.acquire"}
      : !phys.state<@q0>
    %result = "cflow.repeat"(%q) <{
      count = 1000000000000 : i64, event_id = "template_folded.repeat"
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %first = "phys.call"(%current) <{
        callee = @outer, event_id = "template_folded.outer0",
        instance = "template_folded.outer.0"
      }> ({
      ^bb0(%outer_state: !phys.state<@q0>):
        %attempted:2 = "phys.call"(%outer_state) <{
          callee = @attempt, event_id = "template_folded.attempt",
          instance = "template_folded.attempt", profile = @attempt_profile
        }> ({
        ^bb0(%attempt_state: !phys.state<@q0>):
          %measured:2 = phys.measure @measure_z(%attempt_state) {
            event_id = "template_folded.measure",
            record_id = "template_folded.0"
          } : (!phys.state<@q0>) ->
              (!phys.state<@q0>, !phys.record<@attempt_bit>)
          phys.yield %measured#0, %measured#1
            : !phys.state<@q0>, !phys.record<@attempt_bit>
        }) : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        %accepted = phys.condition %attempted#1 {
          event_id = "template_folded.decision"
        } : !phys.record<@attempt_bit> -> i1
        %retried = phys.retry %accepted carries (%attempted#0) {
          attempt = @attempt,
          attempt_event = "template_folded.attempt",
          decision_event = "template_folded.decision",
          event_id = "template_folded.retry", exhaustion = "abort",
          max_attempts = 1 : i64, profile = @attempt_profile,
          success_probability = 5.000000e-01 : f64,
          success_probability_source = @attempt_profile,
          success_probability_evidence = "analysis:abort-fold-test"
        } : (!phys.state<@q0>) -> !phys.state<@q0>
        phys.yield %retried : !phys.state<@q0>
      }) : (!phys.state<@q0>) -> !phys.state<@q0>
      %second = phys.call_template %first {
        callee = @outer, event_id = "template_folded.outer1",
        instance = "template_folded.outer.1",
        template_event = "template_folded.outer0"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %second : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %result : !phys.state<@q0>
  }

  // For max_iterations=2 the folded shape is
  // (condition ; body)^2 ; condition.  A reached condition costs 1 ns.  The
  // abort retry in each body costs 2 ns/1 active ns and continues with
  // probability 1/2.  Including the 1 ns initial predicate gives expected
  // elapsed 1 + 3*(1 + 1/2) + 1*(1/4) = 5.75 ns and active time 1.5 ns.
  phys.graph @bounded_while on @arch : () -> !phys.state<@q0>
      attributes {source_protocol = @estimate_source} {
    %q = phys.acquire [@q0] {event_id = "while.acquire"}
      : !phys.state<@q0>
    %go = "arith.constant"() {
      event_id = "while.go", value = true
    } : () -> i1
    %result:2 = "cflow.while"(%q, %go) <{
      event_id = "while.loop", max_iterations = 2 : i64
    }> ({
    ^bb0(%current: !phys.state<@q0>, %predicate: i1):
      %next_predicate = phys.xor %predicate, %predicate {
        event_id = "while.condition"
      } : i1
      "cflow.while_condition"(%next_predicate, %current, %next_predicate)
        : (i1, !phys.state<@q0>, i1) -> ()
    }, {
    ^bb0(%current: !phys.state<@q0>, %predicate: i1):
      %attempted:2 = "phys.call"(%current) <{
        callee = @attempt, event_id = "while.attempt",
        instance = "while.attempt", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q0>):
        %measured:2 = phys.measure @measure_z(%attempt_state) {
          event_id = "while.measure", record_id = "while.0"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {
        event_id = "while.decision"
      } : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "while.attempt",
        decision_event = "while.decision", event_id = "while.retry",
        exhaustion = "abort", max_attempts = 1 : i64,
        profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:abort-fold-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %retried, %predicate : !phys.state<@q0>, i1
    }) : (!phys.state<@q0>, i1) -> (!phys.state<@q0>, i1)
    phys.return %result#0 : !phys.state<@q0>
  }
}

// CHECK: qlx.estimate_result @folded_estimate
// CHECK-SAME: active_physical_qubit_time_ns = 1.000000e+12 : f64
// CHECK-SAME: active_resource_time_ns = 1.000000e+12 : f64
// CHECK-SAME: exhaustion_probability = 1.000000e+00 : f64
// CHECK-SAME: expected_active_physical_qubit_time_ns = 2.000000e+00 : f64
// CHECK-SAME: expected_active_resource_time_ns = 2.000000e+00 : f64
// CHECK-SAME: expected_makespan_ns = 4.000000e+00 : f64
// CHECK-SAME: makespan_ns = 2.000000e+12 : f64
// CHECK-SAME: maximum_active_physical_qubit_time_ns = 1.000000e+12 : f64
// CHECK-SAME: maximum_active_resource_time_ns = 1.000000e+12 : f64
// CHECK-SAME: maximum_makespan_ns = 2.000000e+12 : f64

// FULL-WORKLOAD: qlx.estimate_result @folded_estimate
// FULL-WORKLOAD-SAME: active_physical_qubit_time_ns = 1.000000e+12 : f64
// FULL-WORKLOAD-SAME: active_resource_time_ns = 1.000000e+12 : f64
// FULL-WORKLOAD-SAME: exhaustion_probability = 1.000000e+00 : f64
// FULL-WORKLOAD-SAME: expected_active_physical_qubit_time_ns = 1.000000e+12 : f64
// FULL-WORKLOAD-SAME: expected_active_resource_time_ns = 1.000000e+12 : f64
// FULL-WORKLOAD-SAME: expected_makespan_ns = 2.000000e+12 : f64
// FULL-WORKLOAD-SAME: makespan_ns = 2.000000e+12 : f64
// FULL-WORKLOAD-SAME: termination_semantics = "full_workload"

// BAD-TERMINATION: error: phys-estimate-schedule termination must be program or full_workload

// TEMPLATE-FOLDED: qlx.estimate_result @template_folded_estimate
// TEMPLATE-FOLDED-SAME: active_physical_qubit_time_ns = 2.000000e+12 : f64
// TEMPLATE-FOLDED-SAME: active_resource_time_ns = 2.000000e+12 : f64
// TEMPLATE-FOLDED-SAME: exhaustion_probability = 1.000000e+00 : f64
// TEMPLATE-FOLDED-SAME: expected_active_physical_qubit_time_ns = 2.000000e+00 : f64
// TEMPLATE-FOLDED-SAME: expected_active_resource_time_ns = 2.000000e+00 : f64
// TEMPLATE-FOLDED-SAME: expected_makespan_ns = 4.000000e+00 : f64
// TEMPLATE-FOLDED-SAME: makespan_ns = 4.000000e+12 : f64
// TEMPLATE-FOLDED-SAME: maximum_makespan_ns = 4.000000e+12 : f64

// With M=2, each reached retry costs 2 + 0.5*2 = 3 expected ns and succeeds
// with probability .75.  The second call consumes the first retry's state, so
// its launch is delayed by a replay rather than hiding that replay.  One body
// therefore has E[T]=3+.75*3=5.25 and continuation .75^2=.5625.  The compact
// trillion-fold result tends to 5.25/(1-.5625)=12 ns without materializing the
// canonical/template bodies.
// TEMPLATE-REPLAY: qlx.estimate_result @template_folded_estimate
// TEMPLATE-REPLAY-SAME: active_physical_qubit_time_ns = 2.000000e+12 : f64
// TEMPLATE-REPLAY-SAME: active_resource_time_ns = 2.000000e+12 : f64
// TEMPLATE-REPLAY-SAME: exhaustion_probability = 1.000000e+00 : f64
// TEMPLATE-REPLAY-SAME: expected_active_physical_qubit_time_ns = 6.000000e+00 : f64
// TEMPLATE-REPLAY-SAME: expected_active_resource_time_ns = 6.000000e+00 : f64
// TEMPLATE-REPLAY-SAME: expected_makespan_ns = 1.200000e+01 : f64
// TEMPLATE-REPLAY-SAME: makespan_ns = 4.000000e+12 : f64
// TEMPLATE-REPLAY-SAME: maximum_active_physical_qubit_time_ns = 4.000000e+12 : f64
// TEMPLATE-REPLAY-SAME: maximum_active_resource_time_ns = 4.000000e+12 : f64
// TEMPLATE-REPLAY-SAME: maximum_makespan_ns = 8.000000e+12 : f64

// WHILE: qlx.estimate_result @bounded_while_estimate
// WHILE-SAME: active_physical_qubit_time_ns = 2.000000e+00 : f64
// WHILE-SAME: active_resource_time_ns = 2.000000e+00 : f64
// WHILE-SAME: exhaustion_probability = 7.500000e-01 : f64
// WHILE-SAME: expected_active_physical_qubit_time_ns = 1.500000e+00 : f64
// WHILE-SAME: expected_active_resource_time_ns = 1.500000e+00 : f64
// WHILE-SAME: expected_makespan_ns = 5.750000e+00 : f64
// WHILE-SAME: makespan_ns = 8.000000e+00 : f64
// WHILE-SAME: maximum_makespan_ns = 8.000000e+00 : f64
