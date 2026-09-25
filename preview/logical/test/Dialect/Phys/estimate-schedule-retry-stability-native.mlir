// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=stable result=stable_schedule' \
// RUN:   --phys-estimate-schedule='schedule=stable_schedule lower-tier=analytical result=estimate' \
// RUN:   | FileCheck %s
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=stable result=stable_schedule' \
// RUN:   --phys-estimate-schedule='schedule=stable_schedule lower-tier=analytical result=estimate' \
// RUN:   | FileCheck %s
// RUN: qlx-opt %s --phys-schedule='graph=near_one result=near_one_schedule' \
// RUN:   --phys-estimate-schedule='schedule=near_one_schedule lower-tier=analytical result=near_one_estimate' \
// RUN:   | FileCheck %s --check-prefix=NEAR
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=near_one result=near_one_schedule' \
// RUN:   --phys-estimate-schedule='schedule=near_one_schedule lower-tier=analytical result=near_one_estimate' \
// RUN:   | FileCheck %s --check-prefix=NEAR

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
  fabric.gadget_profile @epsilon_profile for @attempt attributes {
    metadata = {
      success_probability = "5.551115123125783e-17",
      success_probability_evidence = "analysis:epsilon-test"
    }
  } {}
  fabric.gadget_profile @near_one_profile for @attempt attributes {
    metadata = {
      success_probability = "0.9999999999999999",
      success_probability_evidence = "analysis:near-one-test"
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
  phys.resource @q0 {index = 0 : i64, kind = "qubit",
    resource_class = @qubits}
  phys.graph @stable on @arch : () -> !phys.state<@q0> attributes {
    source_protocol = @estimate_source
  } {
    %0 = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %1:2 = "phys.call"(%0) <{
      callee = @attempt, event_id = "attempt0",
      instance = "attempt.epsilon", profile = @epsilon_profile
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "measure0", record_id = "epsilon"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q0>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %accepted0 = phys.condition %1#1 {event_id = "decision0"}
      : !phys.record<@attempt_bit> -> i1
    %2 = phys.retry %accepted0 carries (%1#0) {
      attempt = @attempt, attempt_event = "attempt0",
      decision_event = "decision0", event_id = "retry0",
      max_attempts = 2 : i64, profile = @epsilon_profile,
      success_probability = 5.551115123125783e-17 : f64,
      success_probability_source = @epsilon_profile,
      success_probability_evidence = "analysis:epsilon-test"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %2 : !phys.state<@q0>
  }
  phys.graph @near_one on @arch : () -> !phys.state<@q0> attributes {
    source_protocol = @estimate_source
  } {
    %0 = phys.acquire [@q0] {event_id = "acquire1"} : !phys.state<@q0>
    %1:2 = "phys.call"(%0) <{
      callee = @attempt, event_id = "attempt1",
      instance = "attempt.near_one", profile = @near_one_profile
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "measure1", record_id = "near_one"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q0>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %accepted1 = phys.condition %1#1 {event_id = "decision1"}
      : !phys.record<@attempt_bit> -> i1
    %2 = phys.retry %accepted1 carries (%1#0) {
      attempt = @attempt, attempt_event = "attempt1",
      decision_event = "decision1", event_id = "retry1",
      max_attempts = 2 : i64, profile = @near_one_profile,
      success_probability = 9.999999999999999e-01 : f64,
      success_probability_source = @near_one_profile,
      success_probability_evidence = "analysis:near-one-test"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %2 : !phys.state<@q0>
  }
}

// The sub-epsilon probability must contribute one expected replay rather than
// collapsing `(1 - (1 - p)^M) / p` to zero.
// CHECK: qlx.estimate_result @estimate
// CHECK-SAME: active_physical_qubit_time_ns = 1.000000e+00 : f64
// CHECK-SAME: active_resource_time_ns = 1.000000e+00 : f64
// CHECK-SAME: expected_active_physical_qubit_time_ns = 2.000000e+00 : f64
// CHECK-SAME: expected_active_resource_time_ns = 2.000000e+00 : f64
// CHECK-SAME: expected_makespan_ns = 4.000000e+00 : f64
// CHECK-SAME: expected_utilization = 5.000000e-01 : f64
// CHECK-SAME: makespan_ns = 2.000000e+00 : f64
// CHECK-SAME: maximum_active_physical_qubit_time_ns = 2.000000e+00 : f64
// CHECK-SAME: maximum_active_resource_time_ns = 2.000000e+00 : f64
// CHECK-SAME: maximum_makespan_ns = 4.000000e+00 : f64
// CHECK-SAME: maximum_utilization = 5.000000e-01 : f64
// CHECK-SAME: utilization = 5.000000e-01 : f64

// Near one, the completion probability rounds to one but the complementary
// exhaustion probability remains representable and must not be erased.
// NEAR: qlx.estimate_result @near_one_estimate
// NEAR-SAME: exhaustion_probability = 1.232595{{[0-9]+}}E-32 : f64
// NEAR-SAME: expected_makespan_ns = 2.000000e+00 : f64
// NEAR-SAME: makespan_ns = 2.000000e+00 : f64
// NEAR-SAME: maximum_makespan_ns = 4.000000e+00 : f64
