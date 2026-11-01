// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=g result=g_schedule' | FileCheck %s
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=g result=g_schedule' | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }

  phys.graph @g on @arch : () -> () {
    %state = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %late = phys.delay %state {
      duration_ns = 120236038.5212108 : f64, event_id = "late"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %repeated = "cflow.repeat"(%late) <{
      count = 2 : i64, event_id = "repeat"
    }> ({
    ^bb0(%active: !phys.state<@q0>):
      %step = phys.delay %active {
        duration_ns = 8034820.83257162 : f64, event_id = "step"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %step : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %repeated {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}

// At this absolute start time, folding the body duration and then recovering
// it through (start + duration) - start changes the low bit.  The schedule
// stores the canonical product directly, exactly as independent replay does.
// CHECK: phys.schedule @g_schedule for @g
// CHECK-SAME: "repeat|repeat|
// CHECK-SAME: repeat_count=2
