// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=condition_tail result=condition_tail_schedule' \
// RUN:   | FileCheck %s
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=condition_tail result=condition_tail_schedule' \
// RUN:   | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit",
    resource_class = @qubits}

  phys.graph @condition_tail on @arch : () -> () {
    %state = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %go = "arith.constant"() {event_id = "go", value = true} : () -> i1
    %result:2 = "cflow.while"(%state, %go) <{
      event_id = "loop", max_iterations = 2 : i64
    }> ({
    ^bb0(%current: !phys.state<@q0>, %predicate: i1):
      "cflow.repeat"() <{count = 1 : i64, event_id = "condition_tail"}> ({
        %side0 = phys.xor %predicate, %predicate {
          event_id = "condition_side0"
        } : i1
        %side1 = phys.xor %side0, %predicate {
          event_id = "condition_side1"
        } : i1
        cflow.yield
      }) : () -> ()
      "cflow.while_condition"(%predicate, %current, %predicate)
        : (i1, !phys.state<@q0>, i1) -> ()
    }, {
    ^bb0(%current: !phys.state<@q0>, %predicate: i1):
      %next = phys.delay %current {
        duration_ns = 1.0 : f64, event_id = "body_step"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %next, %predicate : !phys.state<@q0>, i1
    }) : (!phys.state<@q0>, i1) -> (!phys.state<@q0>, i1)
    phys.release %result#0 {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}

// The loop starts after go at 1 ns.  Its nested, non-yielded condition tail
// finishes at 3 ns, so the body starts there.  For two iterations the folded
// duration is condition(2) * 3 checks + body(1) * 2 iterations = 8 ns.
// CHECK: phys.schedule @condition_tail_schedule for @condition_tail
// CHECK-SAME: "loop|while|1|8|qubits[0]
// CHECK-SAME: "condition_tail|repeat|1|2|control:condition_tail
// CHECK-SAME: parent=loop|branch=condition
// CHECK-SAME: "condition_side1|xor|2|1|control:condition_side1
// CHECK-SAME: parent=condition_tail|branch=body
// CHECK-SAME: "body_step|delay|3|1|qubits[0]
// CHECK-SAME: parent=loop|branch=body
// CHECK-SAME: "release|release|9|0|qubits[0]
// CHECK-SAME: makespan_ns = 9.000000e+00 : f64
