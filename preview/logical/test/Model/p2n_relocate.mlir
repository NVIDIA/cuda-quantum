// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2n", "p2s"]} {
  fabric.code @c {
    distance = 1 : i64, k = 1 : i64, n = 1 : i64, r = 0 : i64,
    metadata = {distance_status = "claimed"}, partitions = {data = 1 : i64}
  }
  fabric.code_profile @cp {
    code = @c, distance_claim = 1 : i64, distance_status = "claimed"
  }
  fabric.encoding @e {
    block = "data", code = @c, logical_ports = ["q0"], profile = @cp
  }
  fabric.encoding_epoch_schema @e_epochs {
    phases = ["static"], initial = "static", transitions = [],
    logical_maps = {}
  }
  fabric.encoding_epoch @e_initial {
    encoding = @e, schema = @e_epochs, phase = "static", index = 0 : i64
  }
  fabric.gadget @handoff(
      %p: !fabric.patch<@c, @e, @e_initial>)
      -> !fabric.patch<@c, @e, @e_initial> {
    fabric.return %p : !fabric.patch<@c, @e, @e_initial>
  }
  fabric.protocol @moving : (!fabric.patch<@c, @e, @e_initial>)
      -> !fabric.patch<@c, @e, @e_initial> {
  ^bb0(%p: !fabric.patch<@c, @e, @e_initial>):
    %0 = fabric.relocate %p using @handoff from @machine::@memory
      to @machine::@compute {
        transition = "teleport_to_compute",
        continuity_witness = "worldline0",
        step = 0 : i64
      } : !fabric.patch<@c, @e, @e_initial>
    fabric.protocol_return %0 : !fabric.patch<@c, @e, @e_initial>
  }
}

// CHECK: fabric.relocate
// CHECK-SAME: using @handoff
// CHECK-SAME: from @machine::@memory to @machine::@compute
// CHECK-SAME: continuity_witness = "worldline0"
// CHECK-SAME: transition = "teleport_to_compute"
