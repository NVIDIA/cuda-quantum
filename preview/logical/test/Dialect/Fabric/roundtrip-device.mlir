// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

// CHECK: fabric.code @steane
// CHECK-SAME: distance = 3
// CHECK-SAME: partitions = {data = 7
fabric.code @steane {
  distance = 3 : i64,
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64},
  gate_map = {"h" = @steane_h, "cx" = @steane_cx}
}

// CHECK: fabric.code @surface_17
fabric.code @surface_17 {
  distance = 17 : i64,
  partitions = {data = 289 : i64, sx = 144 : i64, sz = 144 : i64}
}

// CHECK: fabric.machine @my_device
fabric.machine @my_device {
  // CHECK: fabric.region @Cs0
  // CHECK-SAME: code = @steane
  // CHECK-SAME: encoding = @steane_default
  // CHECK-SAME: epoch = @steane_initial
  fabric.region @Cs0 {
    code = @steane,
    encoding = @steane_default,
    epoch = @steane_initial,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<checkerboard, [5, 5]>
  }
  // CHECK: fabric.region @F0
  fabric.region @F0 {
    code = @steane,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<direct, [6]>
  }
  // CHECK: fabric.interconnect
  // CHECK-SAME: protocol = @teleport
  fabric.interconnect @compute_to_factory {
    region_a = @Cs0, port_a = 0 : i64,
    region_b = @F0,  port_b = 0 : i64,
    protocol = @teleport
  }
}
