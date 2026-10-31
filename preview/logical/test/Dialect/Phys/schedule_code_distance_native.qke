// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --phys-schedule='graph=events result=scheduled' \
// RUN:   | FileCheck %s
// RUN: sed 's/prepare_d3_ns/prepare_d5_ns/' %s \
// RUN:   | not qlx-opt --phys-schedule='graph=events result=scheduled' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISSING
// RUN: sed 's/code_distance = 3 : i64, //' %s \
// RUN:   | not qlx-opt --phys-schedule='graph=events result=scheduled' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNBOUND
// RUN: sed 's/code_distance = 3 : i64/code_distance = 5 : i64/' %s \
// RUN:   | not qlx-opt --phys-schedule='graph=events result=scheduled' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FORGED
// RUN: sed -e '/timing_source =/d' \
// RUN:   -e 's/prepare_d3_ns = "3000.0"},/prepare_d3_ns = "3000.0"}/' %s \
// RUN:   | not qlx-opt 2>&1 | FileCheck %s --check-prefix=UNSOURCED

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @surface_3 {
    distance = 3 : i64,
    metadata = {distance_method = "fixture", distance_provenance = @profile},
    partitions = {data = 1 : i64}
  }
  fabric.code_profile @profile {
    code = @surface_3, distance_claim = 3 : i64,
    distance_status = "exact", evidence = ["distance-timing-test@1"]
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @surface_3,
      floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  phys.machine @arch {
    phys.resource_class @patches {
      kind = "surface_code_patch", count = 1 : i64, native_actions = []
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@patches]
    }
  }
  phys.operating_point @point for @arch {
    timing = {surface_cycle_ns = "1000.0", prepare_d3_ns = "3000.0"},
    timing_source = "distance-timing-test@1"
  }
  phys.resource @p0 {
    architecture = @arch, code_distance = 3 : i64, index = 0 : i64,
    kind = "surface_code_patch", qec_region = @qec::@compute,
    resource_class = @patches
  }
  phys.graph @events on @arch : () -> () attributes {
    operating_point = @point
  } {
    %0 = phys.acquire [@p0] {event_id = "acquire"} : !phys.state<@p0>
    %1 = phys.prepare %0 {event_id = "prepare", state = "zero"}
      : (!phys.state<@p0>) -> !phys.state<@p0>
    phys.release %1 {event_id = "release"} : !phys.state<@p0>
    phys.return
  }
}

// CHECK: phys.resource @p0
// CHECK-SAME: code_distance = 3 : i64
// CHECK: phys.schedule @scheduled for @events
// CHECK-SAME: "prepare|prepare|0|3000|patches[0]
// CHECK-SAME: makespan_ns = 3.000000e+03 : f64
// CHECK-SAME: timing_profile = {prepare_d3_ns = 3.000000e+03 : f64}

// MISSING: has no timing for 'prepare' at code distance 3
// UNBOUND: requires an authenticated code distance to select timing for 'prepare'
// FORGED: code_distance must equal the selected QEC region code distance
// UNSOURCED: distance-qualified timings require a nonempty timing_source
