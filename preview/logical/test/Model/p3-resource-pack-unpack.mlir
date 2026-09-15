// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @Toric {
    distance = 1 : i64, n = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.encoding @toric_encoding {
    code = @Toric, profile = @toric_profile, block = "data",
    logical_ports = ["q0", "q1"]
  }
  func.func @resource_roundtrip(
      %state: !phys.resource_payload<@t_state>,
      %carrier: !phys.state<@magic0>) -> !phys.resource_payload<@t_state> {
    %payload = phys.unpack_resource %state into(%carrier)
      {encoding = @steane_encoding, event_id = "unpack0"}
      : (!phys.resource_payload<@t_state>, !phys.state<@magic0>)
        -> !phys.state<@magic0>
    %packed = phys.pack_resource(%payload) as @t_state
      {encoding = @steane_encoding, event_id = "pack1"}
      : (!phys.state<@magic0>) -> !phys.resource_payload<@t_state>
    return %packed : !phys.resource_payload<@t_state>
  }

  func.func @raw_product_rotation(
      %raw: !phys.resource_payload<@raw_t_state>,
      %left: !phys.state<@q0>,
      %right: !phys.state<@q1>)
      -> (!phys.state<@q0>, !phys.state<@q1>) {
    %next:2 = phys.resource_rotate_product %raw on %left, %right {
      paulis = ["Z", "Z"],
      angle = 7.8539816339744828e-01 : f64,
      event_id = "resource_rpp0"
    } : (!phys.resource_payload<@raw_t_state>,
         !phys.state<@q0>, !phys.state<@q1>)
      -> (!phys.state<@q0>, !phys.state<@q1>)
    return %next#0, %next#1 : !phys.state<@q0>, !phys.state<@q1>
  }

  func.func @multi_payload(
      %state: !phys.resource_payload<@ccz_state>,
      %left: !phys.state<@left0>,
      %right: !phys.state<@right0>) {
    %next:2 = phys.unpack_resource %state into(%left, %right) {
      encoding = @toric_encoding,
      payload_action = #qlx.action<ccz>,
      payload_carrier_segments = array<i64: 0, 1, 2>,
      payload_logical_block_ids = ["left_block", "left_block", "right_block"],
      payload_logical_blocks = array<i64: 0, 0, 1>,
      payload_logical_ports = array<i64: 0, 1, 0>
    } : (!phys.resource_payload<@ccz_state>,
         !phys.state<@left0>, !phys.state<@right0>)
      -> (!phys.state<@left0>, !phys.state<@right0>)
    return
  }
}

// CHECK: phys.unpack_resource
// CHECK-SAME: encoding = @steane_encoding
// CHECK: phys.pack_resource
// CHECK-SAME: as @t_state
// CHECK: phys.resource_rotate_product
// CHECK-SAME: paulis = ["Z", "Z"]
// CHECK: phys.unpack_resource
// CHECK-SAME: payload_action = #qlx.action<ccz>
// CHECK-SAME: payload_carrier_segments = array<i64: 0, 1, 2>
// CHECK-SAME: payload_logical_block_ids = ["left_block", "left_block", "right_block"]
// CHECK-SAME: payload_logical_blocks = array<i64: 0, 0, 1>
// CHECK-NOT: phys.consume_resource
