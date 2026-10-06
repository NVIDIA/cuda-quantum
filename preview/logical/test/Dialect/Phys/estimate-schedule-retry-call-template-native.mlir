// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=templated result=templated_schedule' \
// RUN:   --phys-estimate-schedule='schedule=templated_schedule lower-tier=analytical result=templated_estimate' \
// RUN:   | FileCheck %s --check-prefix=TEMPLATED
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=templated result=templated_schedule' \
// RUN:   --phys-estimate-schedule='schedule=templated_schedule lower-tier=analytical result=templated_estimate' \
// RUN:   | FileCheck %s --check-prefix=TEMPLATED
// RUN: qlx-opt %s --phys-schedule='graph=explicit result=explicit_schedule' \
// RUN:   --phys-estimate-schedule='schedule=explicit_schedule lower-tier=analytical result=explicit_estimate' \
// RUN:   | FileCheck %s --check-prefix=EXPLICIT
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=explicit result=explicit_schedule' \
// RUN:   --phys-estimate-schedule='schedule=explicit_schedule lower-tier=analytical result=explicit_estimate' \
// RUN:   | FileCheck %s --check-prefix=EXPLICIT
// RUN: qlx-opt %s --phys-schedule='graph=large_timestamp result=large_timestamp_schedule' \
// RUN:   --phys-estimate-schedule='schedule=large_timestamp_schedule lower-tier=analytical result=large_timestamp_estimate' \
// RUN:   | FileCheck %s --check-prefix=LARGE-TIMESTAMP
// RUN: sed 's/8.16351190862635E+08/8.16351190362635E+08/' %s | \
// RUN:   not qlx-opt --phys-schedule='graph=large_timestamp result=large_timestamp_schedule' \
// RUN:   --phys-estimate-schedule='schedule=large_timestamp_schedule lower-tier=analytical result=large_timestamp_estimate' \
// RUN:   2>&1 | FileCheck %s --check-prefix=TRUE-CHAIN

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
  fabric.gadget @leaf() { fabric.return }
  fabric.gadget @attempt() { fabric.return }
  fabric.gadget @outer() { fabric.return }
  fabric.gadget_profile @attempt_profile for @attempt attributes {
    metadata = {
      success_probability = "0.5",
      success_probability_evidence = "analysis:nested-template-test"
    }
  } {}

  phys.machine @arch {
    phys.resource_class @qubits {
      count = 4 : i64, kind = "qubit", native_actions = ["measure_z"]
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
  phys.resource @q2 {
    index = 2 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q3 {
    index = 3 : i64, kind = "qubit", resource_class = @qubits
  }

  // The retry replay contains @leaf1, a template inside @attempt0.  Reusing
  // @outer0 on q1 creates the two-level virtual expansion
  // [leaf1, attempt0, outer1].  Its replay metrics must use only that q1-mapped
  // occurrence subtree, not the authored q0 occurrence or another parent.
  phys.graph @templated on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        source_protocol = @estimate_source
      } {
    %q0, %q1 = phys.acquire [@q0, @q1] {event_id = "t.acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %leaf0 = "phys.call"(%q0) <{
      callee = @leaf, event_id = "t.leaf0", instance = "t.leaf.0"
    }> ({
    ^bb0(%state: !phys.state<@q0>):
      %prepared = phys.prepare %state {event_id = "t.prepare0", state = "zero"}
        : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %prepared : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    %q1_ready = phys.delay %q1 {
      duration_ns = 1.0 : f64, event_id = "t.q1_ready"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %outer0 = "phys.call"(%leaf0) <{
      callee = @outer, event_id = "t.outer0", instance = "t.outer.0"
    }> ({
    ^bb0(%outer_state: !phys.state<@q0>):
      %attempted:2 = "phys.call"(%outer_state) <{
        callee = @attempt, event_id = "t.attempt0",
        instance = "t.attempt.0", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q0>):
        %leaf1 = phys.call_template %attempt_state {
          callee = @leaf, event_id = "t.leaf1", instance = "t.leaf.1",
          template_event = "t.leaf0"
        } : (!phys.state<@q0>) -> !phys.state<@q0>
        %measured:2 = phys.measure @measure_z(%leaf1) {
          event_id = "t.measure0", record_id = "t.measure.0"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {event_id = "t.decision0"}
        : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "t.attempt0",
        decision_event = "t.decision0", event_id = "t.retry0",
        max_attempts = 2 : i64, profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:nested-template-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %retried : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    %outer1 = phys.call_template %q1_ready {
      callee = @outer, event_id = "t.outer1", instance = "t.outer.1",
      state_aliases = [{alias = @q1, template = @q0}],
      template_event = "t.outer0"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.return %outer0, %outer1 : !phys.state<@q0>, !phys.state<@q1>
  }

  // The canonical retry starts at a large absolute timestamp.  Its q1-mapped
  // call-template occurrence is authored to begin exactly when the independent
  // q2 retry ends, while an independent q3 retry genuinely overlaps only that
  // q1 occurrence.  Projecting the local retry offset before adding the
  // occurrence start preserves the intended singleton plus two-way layer.
  // Adding absolute times first loses one ULP, falsely overlaps q1 with q2,
  // and turns the intervals into an invalid three-node overlap chain.
  phys.graph @large_timestamp on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>, !phys.state<@q2>, !phys.state<@q3>)
      attributes {source_protocol = @estimate_source} {
    %q0, %q1, %q2, %q3 = phys.acquire [@q0, @q1, @q2, @q3] {
      event_id = "lt.acquire"
    } : !phys.state<@q0>, !phys.state<@q1>, !phys.state<@q2>, !phys.state<@q3>
    %q0_ready = phys.delay %q0 {
      duration_ns = 1.6327503817252797E+08 : f64, event_id = "lt.q0_ready"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %q1_ready = phys.delay %q1 {
      duration_ns = 8.16351190862635E+08 : f64, event_id = "lt.q1_ready"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %q2_ready = phys.delay %q2 {
      duration_ns = 8.16351188862635E+08 : f64, event_id = "lt.q2_ready"
    } : (!phys.state<@q2>) -> !phys.state<@q2>
    %q3_ready = phys.delay %q3 {
      duration_ns = 816351190.86263502 : f64, event_id = "lt.q3_ready"
    } : (!phys.state<@q3>) -> !phys.state<@q3>
    %outer0 = "phys.call"(%q0_ready) <{
      callee = @outer, event_id = "lt.outer0", instance = "lt.outer.0"
    }> ({
    ^bb0(%outer_state: !phys.state<@q0>):
      %attempted:2 = "phys.call"(%outer_state) <{
        callee = @attempt, event_id = "lt.attempt0",
        instance = "lt.attempt.0", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q0>):
        %measured:2 = phys.measure @measure_z(%attempt_state) {
          event_id = "lt.measure0", record_id = "lt.measure.0"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {event_id = "lt.decision0"}
        : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "lt.attempt0",
        decision_event = "lt.decision0", event_id = "lt.retry0",
        max_attempts = 2 : i64, profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:nested-template-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %retried : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    %outer1 = phys.call_template %q1_ready {
      callee = @outer, event_id = "lt.outer1", instance = "lt.outer.1",
      state_aliases = [{alias = @q1, template = @q0}],
      template_event = "lt.outer0"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %outer2 = "phys.call"(%q2_ready) <{
      callee = @outer, event_id = "lt.outer2", instance = "lt.outer.2"
    }> ({
    ^bb0(%outer_state: !phys.state<@q2>):
      %attempted:2 = "phys.call"(%outer_state) <{
        callee = @attempt, event_id = "lt.attempt2",
        instance = "lt.attempt.2", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q2>):
        %measured:2 = phys.measure @measure_z(%attempt_state) {
          event_id = "lt.measure2", record_id = "lt.measure.2"
        } : (!phys.state<@q2>) ->
            (!phys.state<@q2>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q2>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q2>) ->
          (!phys.state<@q2>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {event_id = "lt.decision2"}
        : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "lt.attempt2",
        decision_event = "lt.decision2", event_id = "lt.retry2",
        max_attempts = 2 : i64, profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:nested-template-test"
      } : (!phys.state<@q2>) -> !phys.state<@q2>
      phys.yield %retried : !phys.state<@q2>
    }) : (!phys.state<@q2>) -> !phys.state<@q2>
    %outer3 = "phys.call"(%q3_ready) <{
      callee = @outer, event_id = "lt.outer3", instance = "lt.outer.3"
    }> ({
    ^bb0(%outer_state: !phys.state<@q3>):
      %attempted:2 = "phys.call"(%outer_state) <{
        callee = @attempt, event_id = "lt.attempt3",
        instance = "lt.attempt.3", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q3>):
        %measured:2 = phys.measure @measure_z(%attempt_state) {
          event_id = "lt.measure3", record_id = "lt.measure.3"
        } : (!phys.state<@q3>) ->
            (!phys.state<@q3>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q3>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q3>) ->
          (!phys.state<@q3>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {event_id = "lt.decision3"}
        : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "lt.attempt3",
        decision_event = "lt.decision3", event_id = "lt.retry3",
        max_attempts = 2 : i64, profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:nested-template-test"
      } : (!phys.state<@q3>) -> !phys.state<@q3>
      phys.yield %retried : !phys.state<@q3>
    }) : (!phys.state<@q3>) -> !phys.state<@q3>
    phys.return %outer0, %outer1, %outer2, %outer3
      : !phys.state<@q0>, !phys.state<@q1>, !phys.state<@q2>, !phys.state<@q3>
  }

  // Fully authored equivalent used as the metric/time oracle.
  phys.graph @explicit on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        source_protocol = @estimate_source
      } {
    %q0, %q1 = phys.acquire [@q0, @q1] {event_id = "e.acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %leaf0 = "phys.call"(%q0) <{
      callee = @leaf, event_id = "e.leaf0", instance = "e.leaf.0"
    }> ({
    ^bb0(%state: !phys.state<@q0>):
      %prepared = phys.prepare %state {event_id = "e.prepare0", state = "zero"}
        : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %prepared : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    %q1_ready = phys.delay %q1 {
      duration_ns = 1.0 : f64, event_id = "e.q1_ready"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %outer0 = "phys.call"(%leaf0) <{
      callee = @outer, event_id = "e.outer0", instance = "e.outer.0"
    }> ({
    ^bb0(%outer_state: !phys.state<@q0>):
      %attempted:2 = "phys.call"(%outer_state) <{
        callee = @attempt, event_id = "e.attempt0",
        instance = "e.attempt.0", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q0>):
        %leaf = "phys.call"(%attempt_state) <{
          callee = @leaf, event_id = "e.inner0", instance = "e.inner.0"
        }> ({
        ^bb0(%leaf_state: !phys.state<@q0>):
          %prepared = phys.prepare %leaf_state {
            event_id = "e.inner_prepare0", state = "zero"
          } : (!phys.state<@q0>) -> !phys.state<@q0>
          phys.yield %prepared : !phys.state<@q0>
        }) : (!phys.state<@q0>) -> !phys.state<@q0>
        %measured:2 = phys.measure @measure_z(%leaf) {
          event_id = "e.measure0", record_id = "e.measure.0"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {event_id = "e.decision0"}
        : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "e.attempt0",
        decision_event = "e.decision0", event_id = "e.retry0",
        max_attempts = 2 : i64, profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:nested-template-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %retried : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    %outer1 = "phys.call"(%q1_ready) <{
      callee = @outer, event_id = "e.outer1", instance = "e.outer.1"
    }> ({
    ^bb0(%outer_state: !phys.state<@q1>):
      %attempted:2 = "phys.call"(%outer_state) <{
        callee = @attempt, event_id = "e.attempt1",
        instance = "e.attempt.1", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q1>):
        %leaf = "phys.call"(%attempt_state) <{
          callee = @leaf, event_id = "e.inner1", instance = "e.inner.1"
        }> ({
        ^bb0(%leaf_state: !phys.state<@q1>):
          %prepared = phys.prepare %leaf_state {
            event_id = "e.inner_prepare1", state = "zero"
          } : (!phys.state<@q1>) -> !phys.state<@q1>
          phys.yield %prepared : !phys.state<@q1>
        }) : (!phys.state<@q1>) -> !phys.state<@q1>
        %measured:2 = phys.measure @measure_z(%leaf) {
          event_id = "e.measure1", record_id = "e.measure.1"
        } : (!phys.state<@q1>) ->
            (!phys.state<@q1>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q1>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q1>) ->
          (!phys.state<@q1>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {event_id = "e.decision1"}
        : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "e.attempt1",
        decision_event = "e.decision1", event_id = "e.retry1",
        max_attempts = 2 : i64, profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:nested-template-test"
      } : (!phys.state<@q1>) -> !phys.state<@q1>
      phys.yield %retried : !phys.state<@q1>
    }) : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.return %outer0, %outer1 : !phys.state<@q0>, !phys.state<@q1>
  }
}

// TEMPLATED: qlx.estimate_result @templated_estimate
// TEMPLATED-SAME: exhaustion_probability = 4.375000e-01 : f64
// TEMPLATED-SAME: expected_active_physical_qubit_time_ns = 8.000000e+00 : f64
// TEMPLATED-SAME: expected_active_resource_time_ns = 8.000000e+00 : f64
// TEMPLATED-SAME: expected_makespan_ns = 6.250000e+00 : f64
// TEMPLATED-SAME: makespan_ns = 4.000000e+00 : f64
// TEMPLATED-SAME: maximum_active_physical_qubit_time_ns = 1.000000e+01 : f64
// TEMPLATED-SAME: maximum_active_resource_time_ns = 1.000000e+01 : f64
// TEMPLATED-SAME: maximum_makespan_ns = 7.000000e+00 : f64
// TEMPLATED-SAME: peak_active_physical_qubits = 2 : i64

// EXPLICIT: qlx.estimate_result @explicit_estimate
// EXPLICIT-SAME: exhaustion_probability = 4.375000e-01 : f64
// EXPLICIT-SAME: expected_active_physical_qubit_time_ns = 8.000000e+00 : f64
// EXPLICIT-SAME: expected_active_resource_time_ns = 8.000000e+00 : f64
// EXPLICIT-SAME: expected_makespan_ns = 6.250000e+00 : f64
// EXPLICIT-SAME: makespan_ns = 4.000000e+00 : f64
// EXPLICIT-SAME: maximum_active_physical_qubit_time_ns = 1.000000e+01 : f64
// EXPLICIT-SAME: maximum_active_resource_time_ns = 1.000000e+01 : f64
// EXPLICIT-SAME: maximum_makespan_ns = 7.000000e+00 : f64
// EXPLICIT-SAME: peak_active_physical_qubits = 2 : i64

// LARGE-TIMESTAMP: qlx.estimate_result @large_timestamp_estimate
// LARGE-TIMESTAMP-SAME: exhaustion_probability = 0.68359375 : f64
// LARGE-TIMESTAMP-SAME: peak_active_physical_qubits = 4 : i64

// TRUE-CHAIN: missing evidence: overlapping retries must form one pairwise-overlapping parallel layer
