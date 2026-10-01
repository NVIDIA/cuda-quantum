// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

// Verifies the #fabric.spec_only<"name"> attribute round-trips on every
// op that accepts a protocol attribute (region, interconnect,
// produce_resource, inject, transport, discard_resource).

fabric.code @surface_15 {
  distance = 15 : i64,
  partitions = {data = 225 : i64, sx = 112 : i64, sz = 112 : i64}
}

fabric.code @surface_25 {
  distance = 25 : i64,
  partitions = {data = 625 : i64, sx = 312 : i64, sz = 312 : i64}
}

fabric.machine @spec_only_dev {
  // CHECK: fabric.region @F0
  // CHECK-SAME: protocol = #fabric.spec_only<"distill-15to1-T">
  fabric.region @F0 {
    code = @surface_15,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<direct, [45]>,
    protocol = #fabric.spec_only<"distill-15to1-T">
  }
  // CHECK: fabric.region @C0
  fabric.region @C0 {
    code = @surface_25,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<checkerboard, [10, 10]>
  }
  // CHECK: fabric.interconnect
  // CHECK-SAME: protocol = #fabric.spec_only<"ls-handoff">
  fabric.interconnect @factory_to_compute {
    region_a = @F0, port_a = 0 : i64,
    region_b = @C0, port_b = 0 : i64,
    protocol = #fabric.spec_only<"ls-handoff">
  }
}
