// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --fabric-count='root=moving' | FileCheck %s

fabric.code @code {
  distance = 1 : i64, partitions = {data = 1 : i64}
}

fabric.gadget @handoff(%patch: !fabric.patch<@code>)
    -> !fabric.patch<@code> {
  %out = fabric.h %patch data : !fabric.patch<@code>
  fabric.return %out : !fabric.patch<@code>
}

fabric.protocol @moving : (!fabric.patch<@code>) -> !fabric.patch<@code> {
^bb0(%patch: !fabric.patch<@code>):
  %out = fabric.relocate %patch using @handoff from @machine::@memory
      to @machine::@compute {
        transition = "teleport", continuity_witness = "worldline0",
        step = 0 : i64
      } : !fabric.patch<@code>
  fabric.protocol_return %out : !fabric.patch<@code>
}

// CHECK: fabric.counts = {
// CHECK-DAG: gadget_calls = {handoff = 1 : i64}
// CHECK-DAG: operation_counts = {h = 1 : i64, relocate = 1 : i64}
// CHECK-DAG: source_facets = ["qec_spec", "qec_realization", "protocol_network"]
