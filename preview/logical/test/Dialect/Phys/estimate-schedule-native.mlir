// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=events result=events_schedule' \
// RUN:   --phys-estimate-schedule='schedule=events_schedule lower-tier=analytical result=estimate' \
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
  phys.machine @arch {
    phys.resource_class @atoms {
      count = 2 : i64, kind = "qubit",
      native_actions = [@cz, @measure_z]
    }
    phys.qec_binding @estimate_binding {
      qec_region = @estimate_qec::@compute, resources = [@atoms]
    }
  }
  qlx.logical_to_qec @estimate_logical_to_qec {
    logical = @estimate_logical, qec = @estimate_qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @estimate_qec_to_physical {
    qec = @estimate_qec, physical = @arch,
    entries = [{qec = "compute", binding = "estimate_binding",
                resources = ["atoms"]}]
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
    resource_class = @atoms}
  phys.resource @q1 {index = 1 : i64, kind = "qubit",
    resource_class = @atoms}
  phys.graph @events on @arch : () ->
      (!phys.record<@bit>, !phys.record<@bit>) attributes {
        source_protocol = @estimate_source
      } {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire0"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:2 = phys.prepare %0#0, %0#1 {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    %2:2 = phys.apply @cz(%1#0, %1#1) {event_id = "cz"}
      : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    %3 = phys.measure @measure_z(%2#0) {
      destructive, event_id = "m0", record_id = "r0"
    } : (!phys.state<@q0>) -> !phys.record<@bit>
    %4 = phys.measure @measure_z(%2#1) {
      destructive, event_id = "m1", record_id = "r1"
    } : (!phys.state<@q1>) -> !phys.record<@bit>
    phys.return %3, %4 : !phys.record<@bit>, !phys.record<@bit>
  }
}

// CHECK: qlx.estimate_result @estimate
// CHECK-SAME: active_physical_qubit_time_ns = 6.000000e+00 : f64
// CHECK-SAME: active_resource_time_ns = 6.000000e+00 : f64
// CHECK-SAME: expected_makespan_ns = 3.000000e+00 : f64
// CHECK-SAME: maximum_makespan_ns = 3.000000e+00 : f64
// CHECK-SAME: peak_active_physical_qubits = 2 : i64
// CHECK-SAME: peak_concurrency = 2 : i64
// CHECK-SAME: physical_qubits = 2 : i64
// CHECK-SAME: physical_resources = 2 : i64
// CHECK-SAME: schema = "qlx.schedule-estimate/v2"
// CHECK-SAME: tier = "schedule"
