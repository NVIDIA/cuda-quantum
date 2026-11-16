// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @q {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @bad on @arch : (!phys.state<@q0>) -> () {
  ^bb0(%arg0: !phys.state<@q0>):
    // expected-error @+1 {{source and destination epochs must differ}}
    %0 = phys.epoch_transition @even to @even(%arg0) {
      event_id = "epoch0", evidence = "invalid"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %0 : !phys.state<@q0>
    phys.return
  }
}
