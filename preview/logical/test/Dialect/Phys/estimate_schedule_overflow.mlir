// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt %s --phys-estimate-schedule='schedule=events_schedule lower-tier=analytical result=estimate' 2>&1 | FileCheck %s
// RUN: sed -e 's/physical_unit_kind = "qubit"/physical_unit_kind = "atom"/' \
// RUN:   -e 's/physical_units = 9223372036854775807/physical_units = 1/' %s \
// RUN:   | not qlx-opt --phys-estimate-schedule='schedule=events_schedule lower-tier=analytical result=estimate' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONQUBIT

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
    phys.resource_class @patches {
      kind = "surface_code_patch",
      count = 2 : i64,
      granularity = "patch",
      physical_unit_kind = "qubit",
      physical_units = 9223372036854775807 : i64,
      footprint_evidence = "overflow-test@1",
      native_actions = []
    }
    phys.qec_binding @estimate_binding {
      qec_region = @estimate_qec::@compute, resources = [@patches]
    }
  }
  phys.resource @p0 {
    index = 0 : i64, kind = "surface_code_patch", resource_class = @patches
  }
  phys.resource @p1 {
    index = 1 : i64, kind = "surface_code_patch", resource_class = @patches
  }
  qlx.logical_to_qec @estimate_logical_to_qec {
    logical = @estimate_logical, qec = @estimate_qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @estimate_qec_to_physical {
    qec = @estimate_qec, physical = @arch,
    entries = [{qec = "compute", binding = "estimate_binding",
                resources = ["patches"]}]
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
  phys.graph @events on @arch : () -> () attributes {
    source_protocol = @estimate_source
  } {
    %0:2 = phys.acquire [@p0, @p1] {event_id = "acquire"}
      : !phys.state<@p0>, !phys.state<@p1>
    phys.release %0#0, %0#1 {event_id = "release"}
      : !phys.state<@p0>, !phys.state<@p1>
    phys.return
  }
  phys.schedule @events_schedule for @events {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {},
    tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire|acquire|0|0|patches[0],patches[1]|deps=|data_deps=|resource_deps=",
      "release|release|0|0|patches[0],patches[1]|deps=acquire|data_deps=acquire|resource_deps=acquire"
    ], makespan_ns = 0.0 : f64
  }
}

// CHECK: 'phys.schedule' op provisioned physical-qubit footprint overflows i64
// NONQUBIT: schedule estimate physical_qubits requires patch footprints to use physical_unit_kind = 'qubit'
