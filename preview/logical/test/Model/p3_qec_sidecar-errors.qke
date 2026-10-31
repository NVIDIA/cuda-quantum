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
    phys.resource_class @qubits {
      count = 0 : i64,
      kind = "qubit",
      native_actions = []
    }
  }
  phys.graph @empty on @arch : () -> () {
    phys.return
  }
  // expected-error @+1 {{expected must be false because success sidecars store mismatch bits}}
  phys.selection_sidecar @wrong_polarity for @empty {
    expected = true,
    records = []
  }
}
