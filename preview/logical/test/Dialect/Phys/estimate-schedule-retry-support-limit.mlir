// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: not qlx-opt %s \
// RUN:   --phys-schedule='graph=parallel result=parallel_schedule' \
// RUN:   --phys-estimate-schedule='schedule=parallel_schedule lower-tier=analytical result=estimate' \
// RUN:   2>&1 | FileCheck %s
// RUN: env QLX_PROFILE_SCHEDULE_ESTIMATE=1 qlx-opt %s \
// RUN:   --phys-schedule='graph=serial result=serial_schedule' \
// RUN:   --phys-estimate-schedule='schedule=serial_schedule lower-tier=analytical result=serial_estimate' \
// RUN:   2>&1 | FileCheck %s --check-prefix=SERIAL

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
                success_probability_evidence = "analysis:limit-test"}
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
      max_attempts = 1000000000000 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:limit-test"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %4 = phys.retry %right carries (%2#0) {
      attempt = @attempt, attempt_event = "attempt1",
      decision_event = "decision1", event_id = "retry1",
      max_attempts = 1000000000000 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:limit-test"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.return %3, %4 : !phys.state<@q0>, !phys.state<@q1>
  }
  phys.graph @serial on @arch : () -> !phys.state<@q0> attributes {
    source_protocol = @estimate_source
  } {
    %0 = phys.acquire [@q0] {event_id = "serial_acquire"}
      : !phys.state<@q0>
    %1:2 = "phys.call"(%0) <{
      callee = @attempt, event_id = "serial_attempt",
      instance = "attempt.serial", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "serial_measure", record_id = "serial"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q0>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %accepted = phys.condition %1#1 {event_id = "serial_decision"}
      : !phys.record<@attempt_bit> -> i1
    %2 = phys.retry %accepted carries (%1#0) {
      attempt = @attempt, attempt_event = "serial_attempt",
      decision_event = "serial_decision", event_id = "serial_retry",
      max_attempts = 1000000000000 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:limit-test"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %2 : !phys.state<@q0>
  }
}

// CHECK: missing evidence: exact parallel retry order-statistic support exceeds
// CHECK-SAME: the bounded 262144-point estimator limit

// A trillion-attempt singleton remains exact and bounded independently of M:
// it performs no parallel-owner ancestry propagation or support expansion.
// SERIAL: phys-estimate-schedule retry-slice-work retries=1
// SERIAL: phys-estimate-schedule retry-causality singleton-groups=1 parallel-groups=0 hidden-layers=0 contributing-layers=1 owner-propagation-rows=0 dependency-edges=0 controlled-events=0
// SERIAL: qlx.estimate_result @serial_estimate
// SERIAL-SAME: expected_makespan_ns = 4.000000e+00 : f64
// SERIAL-SAME: makespan_ns = 2.000000e+00 : f64
// SERIAL-SAME: maximum_makespan_ns = 2.000000e+12 : f64
