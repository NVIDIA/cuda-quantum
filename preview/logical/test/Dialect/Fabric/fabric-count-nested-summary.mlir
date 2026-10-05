// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: env QLX_PROFILE_SCHEDULE_ESTIMATE=1 qlx-opt %s --fabric-count 2>&1 | FileCheck %s

fabric.code @two {
  distance = 2 : i64,
  partitions = {data = 4 : i64, sx = 0 : i64, sz = 0 : i64},
  n = 4 : i64, k = 2 : i64, r = 0 : i64,
  hx = [array<i64: 0, 1, 2, 3>], hz = [array<i64: 0, 1, 2, 3>],
  lx = [array<i64: 0, 1>, array<i64: 0, 2>],
  lz = [array<i64: 0, 2>, array<i64: 0, 1>]
}
fabric.code @one {distance = 1 : i64, partitions = {data = 1 : i64}}
fabric.machine @dev {
  fabric.region @C0 {
    code = @two, floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
  fabric.region @C1 {
    code = @one, floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}

fabric.gadget @scratch(%p: !fabric.patch<@two>) -> !fabric.patch<@two> {
  %a = fabric.alloc {code = @two, region = @C0} : !fabric.patch<@two>
  %b = fabric.alloc {code = @one, region = @C1} : !fabric.patch<@one>
  %condition = arith.constant true
  cflow.if %condition {
    %t = fabric.alloc {code = @two, region = @C0} : !fabric.patch<@two>
    fabric.dealloc %t : !fabric.patch<@two>
    cflow.yield
  } else {
    %f0 = fabric.alloc {code = @two, region = @C0} : !fabric.patch<@two>
    %f1 = fabric.alloc {code = @two, region = @C0} : !fabric.patch<@two>
    fabric.dealloc %f0 : !fabric.patch<@two>
    fabric.dealloc %f1 : !fabric.patch<@two>
    cflow.yield
  }
  fabric.tick
  fabric.dealloc %a : !fabric.patch<@two>
  fabric.dealloc %b : !fabric.patch<@one>
  fabric.tick
  %out = fabric.h %p data : !fabric.patch<@two>
  fabric.return %out : !fabric.patch<@two>
}

// Two wrappers share a scratch closure at different caller counts and loop
// multiplicities. A repeated wrapper should reuse its composed summary too.
fabric.gadget @left(%p: !fabric.patch<@two>) -> !fabric.patch<@two> {
  %a = fabric.call @scratch(%p) : (!fabric.patch<@two>) -> !fabric.patch<@two>
  %b = fabric.call @scratch(%a) : (!fabric.patch<@two>) -> !fabric.patch<@two>
  fabric.return %b : !fabric.patch<@two>
}
fabric.gadget @right(%p: !fabric.patch<@two>) -> !fabric.patch<@two> {
  %out = cflow.repeat 3 iter(%arg : !fabric.patch<@two> = %p) {
    %next = fabric.call @scratch(%arg) : (!fabric.patch<@two>) -> !fabric.patch<@two>
    cflow.yield %next : !fabric.patch<@two>
  }
  fabric.return %out : !fabric.patch<@two>
}
fabric.gadget @entry {entry} on @dev() {
  %p = fabric.alloc {code = @two, region = @C0} : !fabric.patch<@two>
  %a = fabric.call @left(%p) : (!fabric.patch<@two>) -> !fabric.patch<@two>
  %background = fabric.alloc {code = @two, region = @C0} : !fabric.patch<@two>
  %b = fabric.call @right(%a) : (!fabric.patch<@two>) -> !fabric.patch<@two>
  fabric.dealloc %background : !fabric.patch<@two>
  %c = fabric.call @left(%b) : (!fabric.patch<@two>) -> !fabric.patch<@two>
  fabric.dealloc %c : !fabric.patch<@two>
  fabric.return
}
// CHECK: callable-summary-probes=3 callable-summary-hits=6
// CHECK: fabric.counts = {
// CHECK-SAME: gadget_calls = {left = 2 : i64, right = 1 : i64, scratch = 7 : i64}
// CHECK-SAME: logical_qubits_peak = 11 : i64
// CHECK-SAME: patches_peak = 6 : i64
// CHECK-SAME: per_region = {C0 = {
// CHECK-SAME: gate_counts = {dealloc = 30 : i64, h = 7 : i64}
// CHECK-SAME: patches = 5 : i64
// CHECK-SAME: rounds_by_kind = {tick = 14 : i64}
// CHECK-SAME: C1 = {
// CHECK-SAME: gate_counts = {dealloc = 7 : i64}
// CHECK-SAME: patches = 1 : i64
// CHECK-SAME: rounds_by_kind = {tick = 7 : i64}
