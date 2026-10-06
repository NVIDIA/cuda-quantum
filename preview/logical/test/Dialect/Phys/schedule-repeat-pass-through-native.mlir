// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=repeat result=repeat_schedule' | FileCheck %s
// RUN: qlx-opt %s --mlir-disable-threading --phys-schedule='graph=repeat result=repeat_schedule' | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 2 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q1 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @repeat on @arch : () -> () {
    %acquired:2 = phys.acquire [@q0, @q1] {event_id = "acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %late = phys.delay %acquired#1 {
      duration_ns = 2.0 : f64, event_id = "late"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %repeated:2 = "cflow.repeat"(%acquired#0, %late) <{
      count = 3 : i64, event_id = "repeat"
    }> ({
    ^bb0(%active: !phys.state<@q0>, %untouched: !phys.state<@q1>):
      %step = phys.delay %active {
        duration_ns = 1.0 : f64, event_id = "step"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %step, %untouched
        : !phys.state<@q0>, !phys.state<@q1>
    }) : (!phys.state<@q0>, !phys.state<@q1>) ->
      (!phys.state<@q0>, !phys.state<@q1>)
    phys.release %repeated#0 {event_id = "release0"}
      : !phys.state<@q0>
    phys.release %repeated#1 {event_id = "release1"}
      : !phys.state<@q1>
    phys.return
  }
}

// CHECK: phys.schedule @repeat_schedule for @repeat
// CHECK-SAME: "acquire|acquire|0|0|qubits[0],qubits[1]|deps=|data_deps=|resource_deps=
// CHECK-SAME: "late|delay|0|2|qubits[1]|deps=acquire|data_deps=acquire|resource_deps=acquire
// CHECK-SAME: "repeat|repeat|2|3|qubits[0],qubits[1]|deps=acquire,late|data_deps=acquire,late|resource_deps={{.*}}repeat_count=3
// CHECK-SAME: "step|delay|2|1|qubits[0]|deps=acquire|data_deps=acquire|resource_deps=acquire{{.*}}parent=repeat|branch=body
// CHECK-SAME: "release0|release|5|0|qubits[0]|deps=repeat|data_deps=repeat|resource_deps=repeat
// CHECK-SAME: "release1|release|5|0|qubits[1]|deps=repeat,late|data_deps=repeat|resource_deps=late
// CHECK-SAME: makespan_ns = 5.000000e+00 : f64
