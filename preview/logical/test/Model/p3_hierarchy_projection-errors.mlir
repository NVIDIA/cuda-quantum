// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 0 : i64,
      kind = "qubit",
      native_actions = []
    }
  }
  phys.graph @physical on @arch : () -> () {
  }
  // expected-error @+1 {{source_hierarchy must resolve to fabric.encoding_hierarchy}}
  phys.hierarchy_projection @wrong_source for @physical {
    source_hierarchy = @arch,
    entries = ["inner[0].data[0]=q0"]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 0 : i64,
      kind = "qubit",
      native_actions = []
    }
  }
  phys.graph @physical on @arch : () -> () {
  }
  // expected-error @+1 {{contains duplicate entry 'inner[0].data[0]=q0'}}
  phys.hierarchy_projection @duplicates for @physical {
    source_hierarchy = @hierarchy,
    entries = ["inner[0].data[0]=q0", "inner[0].data[0]=q0"]
  }
}
