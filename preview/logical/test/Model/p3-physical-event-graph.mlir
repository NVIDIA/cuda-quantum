// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @grid {
    phys.resource_class @qubits {
      kind = "qubit",
      count = 64 : i64,
      native_actions = [@h, @cx, @mz, @reset]
    }
    phys.topology @coupling {
      kind = "grid",
      parameters = {rows = "8", columns = "8"}
    }
    phys.qec_binding @compute {
      qec_region = @patch_machine::@compute,
      resources = [@qubits],
      topology = @coupling
    }
  }
  phys.resource @q0 {
    kind = "qubit",
    resource_class = @qubits,
    index = 0 : i64
  }
  phys.graph @one_h on @grid : (!phys.state<@q0>) -> () {
  ^bb0(%arg0: !phys.state<@q0>):
    %0 = phys.reset %arg0 {state = "zero", event_id = "reset0"} : (!phys.state<@q0>) -> !phys.state<@q0>
    %1 = phys.apply @h(%0) {event_id = "h0"} : (!phys.state<@q0>) -> !phys.state<@q0>
    %2 = phys.delay %1 {duration_ns = 1.000000e+03 : f64, event_id = "idle0"} : (!phys.state<@q0>) -> !phys.state<@q0>
    %3 = phys.barrier %2 {domains = ["drive"], event_id = "barrier0"} : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.release %3 {event_id = "release0"} : !phys.state<@q0>
    phys.return
  }
}

// CHECK: phys.machine @grid
// CHECK: phys.resource_class @qubits
// CHECK: phys.topology @coupling
// CHECK: phys.graph @one_h on @grid
// CHECK: phys.reset
// CHECK: phys.apply @h
// CHECK: phys.delay
// CHECK: phys.barrier
// CHECK: phys.release
