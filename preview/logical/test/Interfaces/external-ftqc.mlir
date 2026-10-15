// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, test-ftqc-plugin
// RUN: qlx-opt --load-dialect-plugin=%test_ftqc_plugin %s | FileCheck %s

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @round on @arch : (!phys.state<@q0>) ->
      (!phys.state<@q0>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>):
    %next, %record = test_ftqc.measure_round %state {
      event_id = "round.measure", record_id = "round.result"
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q0>, !phys.record<@bit>
  }
}

// CHECK: test_ftqc.measure_round
// CHECK-SAME: event_id = "round.measure"
// CHECK-SAME: record_id = "round.result"
