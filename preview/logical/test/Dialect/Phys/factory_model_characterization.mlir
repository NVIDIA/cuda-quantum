// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s --check-prefix=VALID
// RUN: sed 's/^      factory_source_code_distances = array<i64: 3>/      factory_source_code_distances = array<i64: 5>/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=DISTANCE
// RUN: sed 's/array<i64: 3>/array<i64: 5>/g' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=SELECTED-DISTANCE
// RUN: sed 's/^      factory_source_startup_cycles = "2"/      factory_source_startup_cycles = "1"/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=CADENCE
// RUN: sed 's/^    source_build_sha256 =/    removed_source_build_sha256 =/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=PARTIAL
// RUN: sed 's/^    policy = "single_shot"/    policy = "guaranteed"/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=POLICY
// RUN: sed 's/^    source_provider_sha256 = "sha256:222/source_provider_sha256 = "sha256:333/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=PRODUCER
// RUN: sed 's/timing = {surface_cycle_ns = "1000"}/timing = {surface_cycle_ns = "2000"}/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=TIMING
// RUN: sed 's/^    source_timing_profile =/    removed_source_timing_profile =/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=PARTIAL

module attributes {qlx.profiles = ["p3"], qlx.stages = ["p3"]} {
  lvm.domain @logical {
    lvm.space @factory {
      capabilities = [#lvm.capability<"qlx.machine/logical_factory">],
      capacity = 1 : i64
    }
    lvm.stream @magic {
      backing_region = @factory, capacity = 1 : i64,
      produced_by = @provider,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      producer_identity = "selected_t_factory",
      producer_semantics_sha256 = "sha256:2222222222222222222222222222222222222222222222222222222222222222",
      produces = @t_state
    }
    lvm.channel @supply {
      from = @factory, to = @magic,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
  qlx.action @produce_t : () -> !fabric.resource<@t_state> {
    kind = "produce_t_state"
  }
  fabric.protocol @provider : () -> !fabric.resource<@t_state> attributes {
    objective = @produce_t
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @t_state
    } : !fabric.resource<@t_state>
    fabric.protocol_return %state : !fabric.resource<@t_state>
  }
  fabric.code @code {
    distance = 3 : i64, partitions = {data = 9 : i64}
  }
  fabric.machine @qec {
    fabric.region @factory {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<factory>
    }
  }
  phys.machine @arch {
    phys.resource_class @factory_patches {
      kind = "surface_code_patch", count = 2 : i64,
      granularity = "patch",
      footprint_evidence = "distance-3 surface-code factory patches",
      physical_unit_kind = "qubit",
      physical_units = 32 : i64, native_actions = []
    }
    phys.qec_binding @factory_binding {
      qec_region = @qec::@factory, resources = [@factory_patches]
    }
  }
  phys.operating_point @point for @arch {
    timing = {surface_cycle_ns = "1000"}
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "factory", qec = "factory"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{
      qec = "factory", binding = "factory_binding",
      resources = ["factory_patches"],
      factory_startup_cycles = "2",
      factory_output_interval_cycles = "2",
      factory_model_policy = "single_shot",
      factory_model_evidence = "computation:test",
      factory_source_provider = "selected_t_factory",
      factory_source_provider_sha256 = "sha256:2222222222222222222222222222222222222222222222222222222222222222",
      factory_source_startup_cycles = "2",
      factory_source_output_interval_cycles = "2",
      factory_source_build_sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
      factory_source_schedule_sha256 = "1111111111111111111111111111111111111111111111111111111111111111",
      factory_source_operating_point = "source_point",
      factory_source_timing_profile = {cycle_ns = 1.000000e+03 : f64},
      factory_source_output_events = ["pack0"],
      factory_source_selection_events = ["select0"],
      factory_source_physical_units = 64 : i64,
      factory_source_physical_unit_kind = "qubit",
      factory_source_code_distances = array<i64: 3>
    }]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical, operating_point = @point
  }
  phys.factory_model @compiled_factory {
    buffer_capacity = 1 : i64,
    evidence = "computation:test",
    lane_count = 1 : i64,
    output_interval_ns = 2.000000e+03 : f64,
    operating_point = @point,
    physical_resource_class = @factory_patches,
    physical_units = 64 : i64,
    policy = "single_shot",
    provider = @provider,
    provider_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    qec_binding = @factory_binding,
    region = @logical::@factory,
    resource_kind = @t_state,
    source_provider = "selected_t_factory",
    source_provider_sha256 = "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    source_startup_cycles = "2",
    source_output_interval_cycles = "2",
    source_build_sha256 = "0000000000000000000000000000000000000000000000000000000000000000",
    source_schedule_sha256 = "1111111111111111111111111111111111111111111111111111111111111111",
    source_operating_point = "source_point",
    source_timing_profile = {cycle_ns = 1.000000e+03 : f64},
    source_output_events = ["pack0"],
    source_selection_events = ["select0"],
    source_physical_units = 64 : i64,
    source_physical_unit_kind = "qubit",
    source_code_distances = array<i64: 3>,
    startup_ns = 2.000000e+03 : f64,
    stream = @logical::@magic
  }
}

// VALID: phys.factory_model @compiled_factory
// VALID-SAME: source_code_distances = array<i64: 3>
// VALID-SAME: source_selection_events = ["select0"]
// DISTANCE: compiled factory source evidence must exactly match the retained device binding
// SELECTED-DISTANCE: compiled factory source code distances must include the selected factory binding code distance
// CADENCE: compiled factory source evidence must exactly match the retained device binding
// PARTIAL: compiled factory source evidence must appear as one complete tuple
// POLICY: guaranteed factory model cannot retain selection events
// PRODUCER: compiled factory source producer must match the retained P1 factory semantics commitment
// TIMING: compiled factory timing fact 'cycle_ns' does not match selected operating-point 'surface_cycle_ns'
