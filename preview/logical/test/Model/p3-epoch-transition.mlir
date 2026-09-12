// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @q {kind = "qubit", count = 1 : i64,
                            native_actions = []}
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @dynamic on @arch : (!phys.state<@q0>) -> () {
  ^bb0(%arg0: !phys.state<@q0>):
    %0 = phys.epoch_transition @even to @odd(%arg0) {
      event_id = "epoch0", evidence = "verified", logical_map = {q0 = "q0"}
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %0 {event_id = "release0"} : !phys.state<@q0>
    phys.return
  }
}

// CHECK: phys.epoch_transition @even to @odd(%arg0)
// CHECK-SAME: event_id = "epoch0"
// CHECK-SAME: evidence = "verified"
