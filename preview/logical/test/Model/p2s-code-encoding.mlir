// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2s"]} {
  // 2x3 Bacon-Shor subsystem code (qubit (i,j) = 3i+j): one X row-pair
  // stabilizer, two Z column-pair stabilizers, so s = 3 = n - k - r.
  fabric.code @subsystem {
    distance = 0 : i64,
    hx = [array<i64: 0, 1, 2, 3, 4, 5>],
    hz = [array<i64: 0, 1, 3, 4>, array<i64: 1, 2, 4, 5>],
    k = 1 : i64,
    lx = [array<i64: 0, 1, 2>],
    lz = [array<i64: 0, 3>],
    metadata = {distance_status = "unknown"},
    n = 6 : i64,
    partitions = {data = 6 : i64, gauge_ancilla = 2 : i64},
    r = 2 : i64
  }
  fabric.code_profile @subsystem_default_profile {
    code = @subsystem,
    distance_status = "unknown"
  }
  fabric.encoding_epoch_schema @subsystem_epoch_schema {
    phases = ["initial"], initial = "initial", transitions = []
  }
  fabric.encoding @subsystem_default_encoding {
    block = "block0",
    code = @subsystem,
    epoch_schema = @subsystem_epoch_schema,
    initial_epoch = @subsystem_initial_epoch,
    logical_ports = ["q0"],
    profile = @subsystem_default_profile
  }
  fabric.encoding_epoch @subsystem_initial_epoch {
    encoding = @subsystem_default_encoding,
    index = 0 : i64,
    phase = "initial",
    schema = @subsystem_epoch_schema
  }
  fabric.encoding @left_encoding {
    block = "left",
    code = @subsystem,
    logical_ports = ["q0"],
    profile = @subsystem_default_profile
  }
  fabric.encoding @right_encoding {
    block = "right",
    code = @subsystem,
    logical_ports = ["q0"],
    profile = @subsystem_default_profile
  }
  func.func private @distinct_encodings(
    !fabric.patch<@subsystem, @left_encoding>,
    !fabric.patch<@subsystem, @right_encoding>
  )
}

// CHECK: fabric.code @subsystem
// CHECK-SAME: k = 1 : i64
// CHECK-SAME: n = 6 : i64
// CHECK-SAME: r = 2 : i64
// CHECK: fabric.code_profile @subsystem_default_profile
// CHECK-SAME: code = @subsystem
// CHECK: fabric.encoding_epoch_schema @subsystem_epoch_schema
// CHECK: fabric.encoding @subsystem_default_encoding
// CHECK-SAME: code = @subsystem
// CHECK-SAME: epoch_schema = @subsystem_epoch_schema
// CHECK-SAME: initial_epoch = @subsystem_initial_epoch
// CHECK-SAME: logical_ports = ["q0"]
// CHECK-SAME: profile = @subsystem_default_profile
// CHECK: fabric.encoding_epoch @subsystem_initial_epoch
// CHECK-SAME: encoding = @subsystem_default_encoding
// CHECK-SAME: phase = "initial"
// CHECK: func.func private @distinct_encodings(!fabric.patch<@subsystem, @left_encoding>, !fabric.patch<@subsystem, @right_encoding>)
