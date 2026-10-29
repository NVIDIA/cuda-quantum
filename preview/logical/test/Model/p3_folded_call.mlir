// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  func.func @folded(%q: !phys.state<@q0>) -> !phys.state<@q0> {
    %0 = "phys.call"(%q) <{callee = @encoded_x,
                              instance = "root.encoded_x.call0",
                              event_id = "call0"}> ({
    ^bb0(%current: !phys.state<@q0>):
      %1 = phys.apply @x(%current) {event_id = "x0"}
        : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %1 : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    return %0 : !phys.state<@q0>
  }
}

// CHECK: "phys.call"
// CHECK-SAME: callee = @encoded_x
// CHECK-SAME: instance = "root.encoded_x.call0"
// CHECK: phys.apply @x
// CHECK: phys.yield
