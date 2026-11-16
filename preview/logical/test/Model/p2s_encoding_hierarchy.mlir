// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s
// RUN: qlx-opt %s --fabric-count='root=hierarchical_round' | FileCheck %s --check-prefix=COUNT

module attributes {
  qlx.ir_version = "0.4-draft",
  qlx.model_version = "0.3.10-proposed",
  qlx.profiles = ["p0", "p2s"]
} {
  fabric.code @inner {
    distance = 3 : i64,
    k = 1 : i64,
    n = 7 : i64,
    partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64},
    r = 0 : i64
  }
  fabric.code_profile @inner_profile {
    code = @inner,
    distance_claim = 3 : i64,
    distance_status = "claimed"
  }
  fabric.encoding @inner_encoding {
    block = "block0",
    code = @inner,
    logical_ports = ["q0"],
    profile = @inner_profile
  }
  fabric.code @composite {
    distance = 0 : i64,
    k = 1 : i64,
    n = 49 : i64,
    partitions = {data = 49 : i64, sx = 24 : i64, sz = 24 : i64},
    r = 0 : i64
  }
  fabric.code_profile @composite_profile {
    code = @composite,
    distance_status = "unknown"
  }
  fabric.encoding @flat {
    block = "block0",
    code = @composite,
    logical_ports = ["q0"],
    profile = @composite_profile
  }
  fabric.encoding_hierarchy @hierarchy for @composite {
    carrier_map = ["0:0:0", "1:1:0", "2:2:0", "3:3:0", "4:4:0", "5:5:0", "6:6:0"],
    child = @inner_encoding,
    depth = 2 : i64,
    flat_encoding = @flat,
    multiplicity = 7 : i64,
    outer = @inner_encoding
  }
  fabric.encoding @structural {
    block = "block0",
    code = @composite,
    hierarchy = @hierarchy,
    logical_ports = ["q0"],
    profile = @composite_profile
  }
  fabric.encoding_projection @flatten from @structural to @flat {
    carrier_map = array<i64: 0, 1, 2, 3, 4, 5, 6>,
    evidence = "derived_concat_flattening",
    logical_map = array<i64: 0>
  }
  fabric.gadget @inner_round(%arg0: !fabric.patch<@inner>)
      -> !fabric.patch<@inner> {
    fabric.return %arg0 : !fabric.patch<@inner>
  }
  fabric.protocol @hierarchical_round :
      (!fabric.patch<@composite>) -> !fabric.patch<@composite> {
  ^bb0(%arg0: !fabric.patch<@composite>):
    %0 = fabric.encoding_unpack %arg0 using @hierarchy group @inner
      : (!fabric.patch<@composite>)
        -> !fabric.patch_bundle<@hierarchy, @inner>
    %1 = fabric.map_children @inner_round to %0 group @inner
      : (!fabric.patch_bundle<@hierarchy, @inner>)
        -> !fabric.patch_bundle<@hierarchy, @inner>
    %2 = fabric.encoding_pack %1 as @structural
      : (!fabric.patch_bundle<@hierarchy, @inner>)
        -> !fabric.patch<@composite>
    fabric.protocol_return %2 : !fabric.patch<@composite>
  }
}

// CHECK: fabric.encoding_hierarchy @hierarchy for @composite
// CHECK-SAME: depth = 2 : i64
// CHECK-SAME: multiplicity = 7 : i64
// CHECK: fabric.encoding @structural
// CHECK-SAME: hierarchy = @hierarchy
// CHECK: fabric.encoding_projection @flatten from @structural to @flat
// CHECK: fabric.encoding_unpack %{{.*}} using @hierarchy group @inner
// CHECK: fabric.map_children @inner_round to %{{.*}} group @inner
// CHECK: fabric.encoding_pack %{{.*}} as @structural
// COUNT: fabric.counts = {
// COUNT-DAG: gadget_calls = {inner_round = 7 : i64}
// COUNT-DAG: hierarchy_depths = {hierarchy = 2 : i64}
// COUNT-DAG: logical_qubits_peak = 1 : i64
