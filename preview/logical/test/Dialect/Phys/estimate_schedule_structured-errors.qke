// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-count{root=p2 device=device result=static},fabric-estimate-analytical{root=p2 counts=static device=device result=analytical p-phys=0.001 failure-budget=0.2},phys-estimate-schedule{schedule=branch_schedule lower-tier=analytical result=scheduled})' | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @code {
    distance = 1 : i64,
    metadata = {distance_method = "fixture", distance_provenance = @code_evidence},
    partitions = {data = 1 : i64}
  }
  fabric.code_profile @code_evidence {
    code = @code, distance_claim = 1 : i64, distance_status = "exact",
    evidence = ["structured-estimate-test@1"]
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  fabric.protocol @p2 : () -> () {
    fabric.call @work() : () -> ()
    fabric.protocol_return
  }
  fabric.gadget @work {entry} on @qec() {
    %patch = fabric.alloc {code = @code, region = @compute} : !fabric.patch<@code>
    fabric.dealloc %patch : !fabric.patch<@code>
    fabric.return
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@qubits]
    }
  }
  phys.operating_point @point for @arch {
    calibration = {identity = "structured-estimate-test@1"}
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{qec = "compute", binding = "compute_binding",
                resources = ["qubits"]}]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical,
    operating_point = @point
  }
  phys.graph @branch on @arch : () -> () attributes {source_protocol = @p2} {
    %condition = "arith.constant"() {event_id = "condition", value = true} : () -> i1
    "cflow.if"(%condition) <{event_id = "if0"}> ({
      phys.barrier {domains = ["clock"], event_id = "then_tick"} : () -> ()
      cflow.yield
    }, {
      cflow.yield
    }) : (i1) -> ()
    phys.return
  }
  phys.schedule @branch_schedule for @branch {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64},
    tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "condition|arith.constant|0|1|control:condition|deps=|data_deps=|resource_deps=|domain_deps=",
      "if0|if|1|0|control:if0|deps=condition|data_deps=condition|resource_deps=|domain_deps=|condition=condition",
      "then_tick|barrier|1|0|control:then_tick|deps=|data_deps=|resource_deps=|domain_deps=|parent=if0|branch=then|condition=condition"
    ],
    makespan_ns = 1.0 : f64
  }
}

// The v2 estimator accounts for mutually exclusive branches by retaining the
// worst-case executable branch rather than superposing branch occupancy.
// CHECK: qlx.estimate_result @scheduled
// CHECK-SAME: data = {
// CHECK-DAG: active_physical_qubit_time_ns = 0.000000e+00 : f64
// CHECK-DAG: active_resource_time_ns = 0.000000e+00 : f64
// CHECK-DAG: event_count = 3 : i64
// CHECK-DAG: event_counts = {arith.constant = 1 : i64, barrier = 1 : i64, if = 1 : i64}
// CHECK-DAG: expected_active_resource_time_ns = 0.000000e+00 : f64
// CHECK-DAG: expected_makespan_ns = 1.000000e+00 : f64
// CHECK-DAG: expected_utilization = 0.000000e+00 : f64
// CHECK-DAG: makespan_ns = 1.000000e+00 : f64
// CHECK-DAG: maximum_active_resource_time_ns = 0.000000e+00 : f64
// CHECK-DAG: maximum_makespan_ns = 1.000000e+00 : f64
// CHECK-DAG: maximum_utilization = 0.000000e+00 : f64
// CHECK-DAG: peak_active_physical_qubits = 0 : i64
// CHECK-DAG: peak_concurrency = 1 : i64
// CHECK-DAG: physical_qubits = 0 : i64
// CHECK-DAG: physical_resources = 0 : i64
// CHECK-DAG: utilization = 0.000000e+00 : f64
// CHECK-SAME: device = @device
// CHECK-SAME: lower_tier = @analytical
// CHECK-SAME: schema = "qlx.schedule-estimate/v2"
// CHECK-SAME: tier = "schedule"
