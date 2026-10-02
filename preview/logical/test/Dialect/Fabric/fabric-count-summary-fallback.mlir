// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: env QLX_PROFILE_SCHEDULE_ESTIMATE=1 qlx-opt %s --fabric-count 2>&1 | FileCheck %s

// A balanced callable can consume and replace entry reservations. A callee
// that changes exit ownership still falls back, including a negative cache
// hit at a different entry count.
fabric.code @one {distance = 1 : i64, partitions = {data = 1 : i64}}
fabric.machine @dev {
  fabric.region @C0 {
    code = @one, floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}
fabric.gadget @replace(%p: !fabric.patch<@one>) -> !fabric.patch<@one> {
  fabric.dealloc %p : !fabric.patch<@one>
  %out = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  fabric.return %out : !fabric.patch<@one>
}
fabric.gadget @consume(%p: !fabric.patch<@one>) {
  fabric.dealloc %p : !fabric.patch<@one>
  fabric.return
}
fabric.gadget @entry {entry} on @dev() {
  %p = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  %p0 = fabric.call @replace(%p) : (!fabric.patch<@one>) -> !fabric.patch<@one>
  %b = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  %p1 = fabric.call @replace(%p0) : (!fabric.patch<@one>) -> !fabric.patch<@one>
  fabric.call @consume(%p1) : (!fabric.patch<@one>) -> ()
  fabric.call @consume(%b) : (!fabric.patch<@one>) -> ()
  fabric.return
}
// CHECK: callable-summary-probes=2 callable-summary-hits=2 callable-summary-negative-hits=1
// CHECK: fabric.counts = {
// CHECK-SAME: gadget_calls = {consume = 2 : i64, replace = 2 : i64}
// CHECK-SAME: logical_qubits_peak = 2 : i64
// CHECK-SAME: operation_counts = {alloc = 4 : i64, call = 4 : i64, dealloc = 4 : i64}
// CHECK-SAME: patches_peak = 2 : i64
// CHECK-SAME: per_region = {C0 = {
// CHECK-SAME: gate_counts = {dealloc = 4 : i64}
// CHECK-SAME: patches = 2 : i64
