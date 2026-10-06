// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2s", "p2a"]} {
  fabric.code @source {
    distance = 1 : i64,
    k = 1 : i64,
    n = 1 : i64,
    partitions = {data = 1 : i64},
    r = 0 : i64
  }
  fabric.code_profile @source_profile {
    code = @source,
    distance_claim = 1 : i64,
    distance_status = "claimed"
  }
  fabric.encoding @source_encoding {
    block = "block0",
    code = @source,
    logical_ports = ["q0"],
    profile = @source_profile
  }

  fabric.code @destination {
    distance = 1 : i64,
    k = 1 : i64,
    n = 3 : i64,
    partitions = {data = 3 : i64},
    r = 0 : i64
  }
  fabric.code_profile @destination_profile {
    code = @destination,
    distance_claim = 1 : i64,
    distance_status = "claimed"
  }
  fabric.encoding @destination_encoding {
    block = "block0",
    code = @destination,
    logical_ports = ["q0"],
    profile = @destination_profile
  }

  fabric.patch_transform @grow
      from @source_encoding to @destination_encoding {
    destination_roles = {
      active = array<i64: 0, 1, 2>,
      dormant = array<i64>,
      measured = array<i64>,
      reset = array<i64>,
      scratch = array<i64>
    },
    destination_support = array<i64: 0, 1, 2>,
    evidence = "verified_test_transform",
    frame_partitions = {data = 3 : i64},
    logical_map = array<i64: 0>,
    source_roles = {
      active = array<i64: 0>,
      dormant = array<i64: 1, 2>,
      measured = array<i64>,
      reset = array<i64>,
      scratch = array<i64>
    },
    source_support = array<i64: 0>
  }

  func.func @apply(
      %source: !fabric.patch<@source, @source_encoding>)
      -> !fabric.patch<@destination, @destination_encoding> {
    %frame = fabric.transform_begin %source using @grow
      : (!fabric.patch<@source, @source_encoding>)
        -> !fabric.patch_frame<@grow>
    %prepared = fabric.reset %frame data [1, 2]
      : !fabric.patch_frame<@grow>
    %measured, %bits = fabric.mz %prepared data [1, 2] {
      record = "projection"
    } : !fabric.patch_frame<@grow> -> tensor<2xi1>
    %accepted = fabric.all_zero %bits : tensor<2xi1> -> i1
    %destination = fabric.transform_end %measured using @grow
      : (!fabric.patch_frame<@grow>)
        -> !fabric.patch<@destination, @destination_encoding>
    return %destination : !fabric.patch<@destination, @destination_encoding>
  }
}

// CHECK: fabric.patch_transform @grow
// CHECK: %[[FRAME:.+]] = fabric.transform_begin
// CHECK: fabric.reset %[[FRAME]] data [1, 2]
// CHECK: fabric.all_zero
// CHECK: fabric.transform_end
