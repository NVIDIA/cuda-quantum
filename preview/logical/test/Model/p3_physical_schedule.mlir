// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

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
    evidence = ["schedule-test@1"]
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @code,
      floorplan = #fabric.floorplan<direct, [1]>,
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
    phys.resource_class @atoms {
      kind = "qubit",
      count = 2 : i64,
      native_actions = [@cz, @measure_z]
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@atoms]
    }
  }
  phys.operating_point @point for @arch {
    calibration = {identity = "schedule-test@1"}
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical,
    qec = @qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec,
    physical = @arch,
    entries = [{qec = "compute", binding = "compute_binding",
                resources = ["atoms"]}]
  }
  qlx.device @device {
    logical = @logical,
    qec = @qec,
    physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical,
    operating_point = @point
  }
  qlx.estimate_result @counterfeit_static {
    assumptions = [], data = {}, device = @device, evidence = [@p2],
    root = @p2, metadata = {producer = "fabric-count", producer_version = "1"},
    schema = "qlx.fabric-counts/v1", tier = "static"
  }
  qlx.estimate_result @counterfeit_analytical {
    assumptions = [], data = {}, device = @device, evidence = [@p2],
    lower_tier = @counterfeit_static, root = @p2,
    metadata = {producer = "fixture", producer_version = "1"},
    schema = "qlx.fabric-estimate/v1", tier = "analytical"
  }
  qlx.estimate_result @spoofed_analytical {
    assumptions = [], data = {
      cycle_time = 1.000000e+00 : f64,
      failure_budget = 2.000000e-01 : f64,
      p_phys = 1.000000e-03 : f64,
      require_established_distance = true,
      scaling_prefactor = 1.000000e-01 : f64,
      scaling_threshold = 1.000000e-02 : f64
    }, device = @device, evidence = [@p2], lower_tier = @counterfeit_static,
    root = @p2,
    metadata = {producer = "fabric-estimate-analytical", producer_version = "1"},
    schema = "qlx.fabric-estimate/v1", tier = "analytical"
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @atoms}
  phys.resource @q1 {index = 1 : i64, kind = "qubit", resource_class = @atoms}
  phys.graph @events on @arch : () -> (!phys.record<@bit>, !phys.record<@bit>) attributes {source_protocol = @p2} {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire0"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:2 = phys.prepare %0#0, %0#1 {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>, !phys.state<@q1>) -> (!phys.state<@q0>, !phys.state<@q1>)
    %2:2 = phys.apply @cz(%1#0, %1#1) {event_id = "cz"}
      : (!phys.state<@q0>, !phys.state<@q1>) -> (!phys.state<@q0>, !phys.state<@q1>)
    %3 = phys.measure @measure_z(%2#0) {destructive, event_id = "m0", record_id = "r0"}
      : (!phys.state<@q0>) -> !phys.record<@bit>
    %4 = phys.measure @measure_z(%2#1) {destructive, event_id = "m1", record_id = "r1"}
      : (!phys.state<@q1>) -> !phys.record<@bit>
    phys.return %3, %4 : !phys.record<@bit>, !phys.record<@bit>
  }
  phys.schedule @events_schedule for @events {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64},
    tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire0|acquire|0|0|atoms[0],atoms[1]|deps=|data_deps=|resource_deps=",
      "prepare|prepare|0|1|atoms[0],atoms[1]|deps=acquire0|data_deps=acquire0|resource_deps=acquire0",
      "cz|apply|1|1|atoms[0],atoms[1]|deps=prepare|data_deps=prepare|resource_deps=prepare",
      "m0|measure|2|1|atoms[0]|deps=cz|data_deps=cz|resource_deps=cz",
      "m1|measure|2|1|atoms[1]|deps=cz|data_deps=cz|resource_deps=cz"
    ],
    makespan_ns = 3.0 : f64
  }
}

// CHECK: phys.graph @events on @arch
// CHECK: %[[ACQ:.*]]:2 = phys.acquire [@q0, @q1]
// CHECK: phys.prepare
// CHECK: phys.apply @cz
// CHECK: phys.measure @measure_z
// CHECK: phys.schedule @events_schedule for @events
// CHECK-SAME: makespan_ns = 3.000000e+00 : f64
