// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2s", "p2n"]} {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  fabric.code_profile @profile {code = @c}
  fabric.encoding_epoch_schema @schema {
    phases = ["even", "odd"],
    initial = "even",
    periodic,
    closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
    transitions = ["even->odd", "odd->even"]
  }
  fabric.encoding @encoding {
    block = "block0",
    code = @c,
    epoch_schema = @schema,
    initial_epoch = @even,
    logical_ports = ["q0"],
    profile = @profile
  }
  fabric.encoding_epoch @even {
    encoding = @encoding, index = 0 : i64, phase = "even", schema = @schema
  }
  fabric.encoding_epoch @odd {
    encoding = @encoding, index = 0 : i64, phase = "odd", schema = @schema
  }
  fabric.protocol @cycle :
      (!fabric.patch<@c, @encoding, @even>) ->
      !fabric.patch<@c, @encoding, @even> {
  ^bb0(%arg0: !fabric.patch<@c, @encoding, @even>):
    %0 = fabric.epoch_transition %arg0 to @odd {
      evidence = "verified_even_to_odd", logical_map = {q0 = "q0"}
    } : (!fabric.patch<@c, @encoding, @even>) ->
        !fabric.patch<@c, @encoding, @odd>
    %1 = fabric.epoch_transition %0 to @even {
      evidence = "verified_odd_to_even", logical_map = {q0 = "q0"}
    } : (!fabric.patch<@c, @encoding, @odd>) ->
        !fabric.patch<@c, @encoding, @even>
    fabric.protocol_return %1 : !fabric.patch<@c, @encoding, @even>
  }
}

// CHECK: fabric.encoding_epoch_schema @schema
// CHECK: fabric.encoding_epoch @even
// CHECK: fabric.encoding_epoch @odd
// CHECK: fabric.epoch_transition %arg0 to @odd
// CHECK-SAME: evidence = "verified_even_to_odd"
// CHECK: fabric.epoch_transition %0 to @even
