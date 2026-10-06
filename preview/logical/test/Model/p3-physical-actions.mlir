// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.action @rydberg_cz {
    arity = 2 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22cz\22,\22parameters\22:{}}",
    controller_bindings = {lanes = "cz"}
  }
  phys.action @global_h {
    arity = 1 : i64,
    broadcast,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22h\22,\22parameters\22:{}}",
    controller_bindings = {lanes = "h"}
  }
  phys.machine @array {
    phys.resource_class @atoms {
      kind = "atom",
      count = 2 : i64,
      native_actions = [@rydberg_cz, @global_h]
    }
  }
  phys.resource @a0 {kind = "atom", resource_class = @atoms, index = 0 : i64}
  phys.resource @a1 {kind = "atom", resource_class = @atoms, index = 1 : i64}
  phys.graph @blockade on @array :
      (!phys.state<@a0>, !phys.state<@a1>) -> () {
  ^bb0(%a0: !phys.state<@a0>, %a1: !phys.state<@a1>):
    %n0, %n1 = phys.apply @rydberg_cz(%a0, %a1) {event_id = "cz0"}
      : (!phys.state<@a0>, !phys.state<@a1>)
        -> (!phys.state<@a0>, !phys.state<@a1>)
    %h0, %h1 = phys.apply @global_h(%n0, %n1) {event_id = "global_h0"}
      : (!phys.state<@a0>, !phys.state<@a1>)
        -> (!phys.state<@a0>, !phys.state<@a1>)
    phys.release %h0, %h1 {event_id = "release0"} :
      !phys.state<@a0>, !phys.state<@a1>
    phys.return
  }
}

// CHECK: phys.action @rydberg_cz
// CHECK-SAME: arity = 2 : i64
// CHECK-SAME: controller_bindings = {lanes = "cz"}
// CHECK-SAME: process =
// CHECK: phys.action @global_h
// CHECK-SAME: arity = 1 : i64
// CHECK-SAME: broadcast
// CHECK: phys.resource_class @atoms
// CHECK-SAME: native_actions = [@rydberg_cz, @global_h]
// CHECK: phys.apply @rydberg_cz
// CHECK: phys.apply @global_h
