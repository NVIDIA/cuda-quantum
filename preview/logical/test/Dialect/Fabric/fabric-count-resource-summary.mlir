// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: env QLX_PROFILE_SCHEDULE_ESTIMATE=1 qlx-opt %s --fabric-count='root=entry' 2>&1 | FileCheck %s

// Resource payloads temporarily increase live ownership. Reuse their summary
// with different caller reservations without scaling or losing the peak.
fabric.code @bare {distance = 1 : i64, partitions = {data = 1 : i64}}
fabric.machine @dev {
  fabric.region @C0 {
    code = @bare, floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}
fabric.gadget @roundtrip(%anchor: !fabric.patch<@bare, @enc, @epoch>,
                         %state: !fabric.resource<@t_state>)
    -> (!fabric.patch<@bare, @enc, @epoch>, !fabric.resource<@t_state>) {
  %next, %payload = fabric.unpack_resource %state like(%anchor)
      : (!fabric.resource<@t_state>, !fabric.patch<@bare, @enc, @epoch>)
        -> (!fabric.patch<@bare, @enc, @epoch>, !fabric.patch<@bare, @enc, @epoch>)
  %packed = fabric.pack_resource (%payload) as @t_state
      {payload_encodings = [@enc]}
      : (!fabric.patch<@bare, @enc, @epoch>) -> !fabric.resource<@t_state>
  fabric.return %next, %packed
      : !fabric.patch<@bare, @enc, @epoch>, !fabric.resource<@t_state>
}
fabric.protocol @entry :
    (!fabric.patch<@bare, @enc, @epoch>, !fabric.resource<@t_state>)
    -> (!fabric.patch<@bare, @enc, @epoch>, !fabric.resource<@t_state>) {
^bb0(%p: !fabric.patch<@bare, @enc, @epoch>, %state: !fabric.resource<@t_state>):
  %b0 = fabric.alloc {code = @bare, region = @C0} : !fabric.patch<@bare>
  %p0, %s0 = fabric.call @roundtrip(%p, %state)
      : (!fabric.patch<@bare, @enc, @epoch>, !fabric.resource<@t_state>)
        -> (!fabric.patch<@bare, @enc, @epoch>, !fabric.resource<@t_state>)
  %b1 = fabric.alloc {code = @bare, region = @C0} : !fabric.patch<@bare>
  %p1, %s1 = fabric.call @roundtrip(%p0, %s0)
      : (!fabric.patch<@bare, @enc, @epoch>, !fabric.resource<@t_state>)
        -> (!fabric.patch<@bare, @enc, @epoch>, !fabric.resource<@t_state>)
  fabric.dealloc %b0 : !fabric.patch<@bare>
  fabric.dealloc %b1 : !fabric.patch<@bare>
  fabric.protocol_return %p1, %s1
      : !fabric.patch<@bare, @enc, @epoch>, !fabric.resource<@t_state>
}
// CHECK: callable-summary-probes=1 callable-summary-hits=2
// CHECK: fabric.counts = {
// CHECK-SAME: logical_qubits_peak = 4 : i64
// CHECK-SAME: operation_counts = {alloc = 2 : i64, call = 2 : i64, dealloc = 2 : i64, pack_resource = 2 : i64, unpack_resource = 2 : i64}
// CHECK-SAME: patches_peak = 4 : i64
