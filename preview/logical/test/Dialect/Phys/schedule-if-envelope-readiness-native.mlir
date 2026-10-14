// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=events result=events_schedule' | FileCheck %s
// RUN: env MLIR_ENABLE_MULTITHREADING=0 qlx-opt %s --phys-schedule='graph=events result=events_schedule' | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @events on @arch : () -> !phys.state<@q0> {
    %q = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %delayed = phys.delay %q {
      duration_ns = 3.0 : f64, event_id = "delay"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %condition = "arith.constant"() {
      event_id = "condition", value = true
    } : () -> i1
    %selected = "cflow.if"(%condition) <{event_id = "if"}> ({
      %then = phys.delay %delayed {
        duration_ns = 1.0 : f64, event_id = "then"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %then : !phys.state<@q0>
    }, {
      cflow.yield %delayed : !phys.state<@q0>
    }) : (i1) -> !phys.state<@q0>
    phys.return %selected : !phys.state<@q0>
  }
}

// The structural branch envelope begins as soon as its predicate is ready.
// Its resourceful child independently waits for the captured state.
// CHECK: phys.schedule @events_schedule for @events
// CHECK-SAME: "if|if|1|3|qubits[0]|deps=condition|data_deps=condition|resource_deps=|domain_deps=|parent=|branch=|condition=condition|
// CHECK-SAME: "then|delay|3|1|qubits[0]|deps=delay|data_deps=delay|resource_deps=delay|domain_deps=|parent=if|branch=then|condition=condition|
// CHECK-SAME: makespan_ns = 4.000000e+00 : f64
