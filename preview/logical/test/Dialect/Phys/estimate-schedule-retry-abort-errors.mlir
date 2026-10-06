// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: not qlx-opt %s --phys-schedule='graph=conditional result=conditional_schedule' \
// RUN:   --phys-estimate-schedule='schedule=conditional_schedule lower-tier=analytical result=estimate' \
// RUN:   2>&1 | FileCheck %s --check-prefix=CONDITIONAL
// RUN: sed '0,/max_attempts = 2 : i64/s//max_attempts = 2 : i64, exhaustion = "abort"/' %S/estimate-schedule-retry-max-plus.mlir | \
// RUN:   not qlx-opt --phys-schedule='graph=slack result=slack_schedule' \
// RUN:   --phys-estimate-schedule='schedule=slack_schedule lower-tier=analytical result=estimate' \
// RUN:   2>&1 | FileCheck %s --check-prefix=CROSSING
// RUN: qlx-opt %s --phys-schedule='graph=committed result=committed_schedule' \
// RUN:   --phys-estimate-schedule='schedule=committed_schedule lower-tier=analytical result=committed_estimate' | \
// RUN:   FileCheck %s --check-prefix=COMMITTED
// RUN: sed '/event_id = "committed.abort.cut"/d' %s | \
// RUN:   not qlx-opt --phys-schedule='graph=committed result=committed_schedule' \
// RUN:   --phys-estimate-schedule='schedule=committed_schedule lower-tier=analytical result=committed_estimate' \
// RUN:   2>&1 | FileCheck %s --check-prefix=CROSSING

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
      success_probability_evidence = "analysis:abort-branch-test"
    }
  } {}

  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = ["measure_z"]
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
  phys.resource @q1 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }

  phys.graph @conditional on @arch : () -> !phys.state<@q0>
      attributes {source_protocol = @estimate_source} {
    %q = phys.acquire [@q0] {event_id = "conditional.acquire"}
      : !phys.state<@q0>
    %condition = "arith.constant"() {
      event_id = "conditional.condition", value = true
    } : () -> i1
    %selected = "cflow.if"(%condition) <{event_id = "conditional.if"}> ({
      %attempted:2 = "phys.call"(%q) <{
        callee = @attempt, event_id = "conditional.attempt",
        instance = "conditional.attempt", profile = @attempt_profile
      }> ({
      ^bb0(%current: !phys.state<@q0>):
        %measured:2 = phys.measure @measure_z(%current) {
          event_id = "conditional.measure", record_id = "conditional"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {
        event_id = "conditional.decision"
      } : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "conditional.attempt",
        decision_event = "conditional.decision",
        event_id = "conditional.retry", exhaustion = "abort",
        max_attempts = 2 : i64, profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:abort-branch-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %retried : !phys.state<@q0>
    }, {
      %fallback = phys.delay %q {
        duration_ns = 2.0 : f64, event_id = "conditional.fallback"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %fallback : !phys.state<@q0>
    }) : (i1) -> !phys.state<@q0>
    phys.return %selected : !phys.state<@q0>
  }

  // The q1 delay and the outer RUS wrapper launch together.  The operand-free
  // clock barrier inside the wrapper authenticates that q1 has finished before
  // the retry can abort.  Its complete metrics are committed, but its tail is
  // hidden beneath the wrapper prefix.  Removing the cut recreates a real
  // crossing and must continue to fail closed.
  phys.graph @committed on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        source_protocol = @estimate_source
      } {
    %q0, %q1 = phys.acquire [@q0, @q1] {event_id = "committed.acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %q1_done = phys.delay %q1 {
      duration_ns = 3.0 : f64, event_id = "committed.independent"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %q0_done = "phys.call"(%q0) <{
      callee = @outer, event_id = "committed.outer",
      instance = "committed.outer"
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %attempted:2 = "phys.call"(%current) <{
        callee = @attempt, event_id = "committed.attempt",
        instance = "committed.attempt", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q0>):
        %measured:2 = phys.measure @measure_z(%attempt_state) {
          event_id = "committed.measure", record_id = "committed"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {
        event_id = "committed.decision"
      } : !phys.record<@attempt_bit> -> i1
      phys.barrier {domains = ["clock"], event_id = "committed.abort.cut"} : () -> ()
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "committed.attempt",
        decision_event = "committed.decision", event_id = "committed.retry",
        exhaustion = "abort", max_attempts = 1 : i64,
        profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:abort-branch-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %retried : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %q0_done, %q1_done
      : !phys.state<@q0>, !phys.state<@q1>
  }
}

// CONDITIONAL: missing evidence: abort-aware expectation cannot compose retry
// CONDITIONAL-SAME: beneath conditional control without authenticated branch probabilities

// CROSSING: missing evidence: independent scheduled work crosses an abort
// CROSSING-SAME: continuation cut

// COMMITTED: qlx.estimate_result @committed_estimate
// COMMITTED-SAME: expected_makespan_ns = 3.000000e+00
// COMMITTED-SAME: maximum_makespan_ns = 3.000000e+00
