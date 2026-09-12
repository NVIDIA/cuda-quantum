// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64,
      kind = "qubit",
      native_actions = ["h"]
    }
  }
  phys.graph @physical on @arch : () -> () {
  }
  phys.hierarchy_projection @mapping for @physical {
    source_hierarchy = @hierarchy,
    entries = ["inner[0].data[0]=q0"]
  }
}

// CHECK: phys.hierarchy_projection @mapping for @physical
// CHECK-SAME: entries = ["inner[0].data[0]=q0"]
// CHECK-SAME: source_hierarchy = @hierarchy
