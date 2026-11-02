// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2n"]} {
  fabric.code @BareQubit {
    distance = 1 : i64,
    partitions = {data = 1 : i64},
    k = 1 : i64
  }

  fabric.encoding @toric_encoding {
    code = @Toric, profile = @toric_profile, block = "data",
    logical_ports = ["q0", "q1"]
  }
  fabric.protocol @resource_roundtrip :
      (!fabric.patch<@Steane, @steane_encoding, @epoch0>,
       !fabric.resource<@t_state>)
      -> (!fabric.patch<@Steane, @steane_encoding, @epoch0>,
          !fabric.resource<@t_state>) {
  ^bb0(%anchor: !fabric.patch<@Steane, @steane_encoding, @epoch0>,
       %state: !fabric.resource<@t_state>):
    %next, %payload = fabric.unpack_resource %state like(%anchor)
      : (!fabric.resource<@t_state>,
         !fabric.patch<@Steane, @steane_encoding, @epoch0>)
        -> (!fabric.patch<@Steane, @steane_encoding, @epoch0>,
            !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>)
    %packed = fabric.pack_resource (%payload) as @t_state
      {payload_encodings = [@bare_encoding]}
      : (!fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>)
        -> !fabric.resource<@t_state>
    fabric.protocol_return %next, %packed
      : !fabric.patch<@Steane, @steane_encoding, @epoch0>,
        !fabric.resource<@t_state>
  }

  func.func @multi_payload(
      %left: !fabric.patch<@Toric, @toric_encoding, @epoch0>,
      %right: !fabric.patch<@Toric, @toric_encoding, @epoch0>,
      %state: !fabric.resource<@ccz_state>) {
    %next:4 = fabric.unpack_resource %state like(%left, %right) {
      payload_action = #qlx.action<ccz>,
      payload_logical_block_ids = ["left_block", "left_block", "right_block"],
      payload_logical_blocks = array<i64: 0, 0, 1>,
      payload_logical_ports = array<i64: 0, 1, 0>
    } : (!fabric.resource<@ccz_state>,
         !fabric.patch<@Toric, @toric_encoding, @epoch0>,
         !fabric.patch<@Toric, @toric_encoding, @epoch0>)
      -> (!fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>,
          !fabric.patch<@Toric, @toric_encoding, @epoch0>)
    return
  }

  fabric.protocol @raw_product_rotation :
      (!fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>,
       !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>,
       !fabric.resource<@raw_t_state>)
      -> (!fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>,
          !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>) {
  ^bb0(%left: !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>,
       %right: !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>,
       %raw: !fabric.resource<@raw_t_state>):
    %next:2 = fabric.resource_rotate_product %raw on %left, %right {
      patch_indices = array<i64: 0, 1>,
      logical_indices = array<i64: 0, 0>,
      pauli_product = "ZZ",
      angle = 7.8539816339744828e-01 : f64
    } : (!fabric.resource<@raw_t_state>,
         !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>,
         !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>)
      -> (!fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>,
          !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>)
    fabric.protocol_return %next#0, %next#1
      : !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>,
        !fabric.patch<@BareQubit, @bare_encoding, @bare_epoch0>
  }
}

// CHECK: fabric.unpack_resource
// CHECK-SAME: !fabric.resource<@t_state>
// CHECK: fabric.pack_resource
// CHECK-SAME: as @t_state
// CHECK: fabric.unpack_resource
// CHECK-SAME: payload_action = #qlx.action<ccz>
// CHECK-SAME: payload_logical_block_ids = ["left_block", "left_block", "right_block"]
// CHECK-SAME: payload_logical_blocks = array<i64: 0, 0, 1>
// CHECK-SAME: payload_logical_ports = array<i64: 0, 1, 0>
// CHECK: fabric.resource_rotate_product
// CHECK-SAME: pauli_product = "ZZ"
// CHECK-NOT: fabric.inject
