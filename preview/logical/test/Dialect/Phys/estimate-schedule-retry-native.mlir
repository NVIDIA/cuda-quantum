// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=parallel result=parallel_schedule' \
// RUN:   --phys-estimate-schedule='schedule=parallel_schedule lower-tier=analytical result=estimate' \
// RUN:   | FileCheck %s
// RUN: env QLX_PROFILE_SCHEDULE_ESTIMATE=1 qlx-opt %s \
// RUN:   --phys-schedule='graph=parallel result=parallel_schedule' \
// RUN:   --phys-estimate-schedule='schedule=parallel_schedule lower-tier=analytical result=estimate' \
// RUN:   2>&1 | FileCheck %s --check-prefix=PROFILE
// RUN: not qlx-opt %s --phys-schedule='graph=causal result=causal_schedule' \
// RUN:   --phys-estimate-schedule='schedule=causal_schedule lower-tier=analytical result=estimate' \
// RUN:   2>&1 | FileCheck %s --check-prefix=CAUSAL
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=parallel result=parallel_schedule' \
// RUN:   --phys-estimate-schedule='schedule=parallel_schedule lower-tier=analytical result=estimate' \
// RUN:   | FileCheck %s

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
  fabric.gadget_profile @attempt_profile for @attempt attributes {
    metadata = {success_probability = "0.5",
                success_probability_evidence = "analysis:test"}
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
  phys.resource @q0 {index = 0 : i64, kind = "qubit",
    resource_class = @qubits}
  phys.resource @q1 {index = 1 : i64, kind = "qubit",
    resource_class = @qubits}
  phys.graph @parallel on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        source_protocol = @estimate_source
      } {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire0"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:2 = "phys.call"(%0#0) <{
      callee = @attempt, event_id = "attempt0",
      instance = "attempt.left", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "measure0", record_id = "left"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q0>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %2:2 = "phys.call"(%0#1) <{
      callee = @attempt, event_id = "attempt1",
      instance = "attempt.right", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q1>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "measure1", record_id = "right"
      } : (!phys.state<@q1>) ->
          (!phys.state<@q1>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q1>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q1>) ->
        (!phys.state<@q1>, !phys.record<@attempt_bit>)
    %left = phys.condition %1#1 {event_id = "decision0"}
      : !phys.record<@attempt_bit> -> i1
    %right = phys.condition %2#1 {event_id = "decision1"}
      : !phys.record<@attempt_bit> -> i1
    %3 = phys.retry %left carries (%1#0) {
      attempt = @attempt, attempt_event = "attempt0",
      decision_event = "decision0", event_id = "retry0",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:test"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %4 = phys.retry %right carries (%2#0) {
      attempt = @attempt, attempt_event = "attempt1",
      decision_event = "decision1", event_id = "retry1",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:test"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.return %3, %4 : !phys.state<@q0>, !phys.state<@q1>
  }
  phys.graph @causal on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        source_protocol = @estimate_source
      } {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "causal_acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:2 = "phys.call"(%0#0) <{
      callee = @attempt, event_id = "causal_attempt0",
      instance = "causal.left", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "causal_measure0", record_id = "causal_left"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q0>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %2:2 = "phys.call"(%0#1) <{
      callee = @attempt, event_id = "causal_attempt1",
      instance = "causal.right", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q1>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "causal_measure1", record_id = "causal_right"
      } : (!phys.state<@q1>) ->
          (!phys.state<@q1>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q1>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q1>) ->
        (!phys.state<@q1>, !phys.record<@attempt_bit>)
    %left = phys.condition %1#1 {event_id = "causal_decision0"}
      : !phys.record<@attempt_bit> -> i1
    %right_raw = phys.condition %2#1 {event_id = "causal_decision1_raw"}
      : !phys.record<@attempt_bit> -> i1
    %3 = phys.retry %left carries (%1#0) {
      attempt = @attempt, attempt_event = "causal_attempt0",
      decision_event = "causal_decision0", event_id = "causal_retry0",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:test"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %after:2 = phys.measure @measure_z(%3) {
      event_id = "causal_after_retry0", record_id = "causal_foreign"
    } : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %foreign = phys.condition %after#1 {event_id = "causal_foreign_condition"}
      : !phys.record<@attempt_bit> -> i1
    %right = phys.xor %foreign, %right_raw {
      event_id = "causal_decision1"
    } : i1
    %4 = phys.retry %right carries (%2#0) {
      attempt = @attempt, attempt_event = "causal_attempt1",
      decision_event = "causal_decision1", event_id = "causal_retry1",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:test"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.return %after#0, %4 : !phys.state<@q0>, !phys.state<@q1>
  }
}

// CHECK: qlx.estimate_result @estimate
// CHECK-SAME: exhaustion_probability = 4.375000e-01 : f64
// CHECK-SAME: expected_makespan_ns = 3.500000e+00 : f64
// CHECK-SAME: makespan_ns = 2.000000e+00 : f64
// CHECK-SAME: maximum_makespan_ns = 4.000000e+00 : f64
// CHECK-SAME: physical_qubits = 2 : i64
// CHECK-SAME: schema = "qlx.schedule-estimate/v2"

// PROFILE: phys-estimate-schedule retry-slice-work retries=2
// PROFILE: phys-estimate-schedule retry-causality singleton-groups=0 parallel-groups=1 hidden-layers=0 contributing-layers=1 owner-propagation-rows={{[1-9][0-9]*}} dependency-edges={{[1-9][0-9]*}} controlled-events={{[1-9][0-9]*}}

// CAUSAL: missing evidence: parallel retry replay slices must be causally
// CAUSAL-SAME: independent
