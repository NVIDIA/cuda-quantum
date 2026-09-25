// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  phys.machine @bad {
    // expected-error @+1 {{count must be nonnegative}}
    phys.resource_class @qubits {kind = "qubit", count = -1 : i64, native_actions = []}
  }
}

// -----

module {
  func.func @bad_reset(%arg0: !phys.state<@q0>) {
    // expected-error @+1 {{state must be zero or plus}}
    %0 = phys.reset %arg0 {state = "unknown"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    return
  }
}

// -----

module {
  // expected-error @+1 {{requires physical states or clock/resource domains}}
  "phys.barrier"() : () -> ()
}

// -----

module {
  func.func @bad(%arg0: !phys.state<@q0>) {
    // expected-error @+1 {{duration_ns must be nonnegative}}
    %0 = phys.delay %arg0 {duration_ns = -1.000000e+00 : f64} : (!phys.state<@q0>) -> !phys.state<@q0>
    return
  }
}
