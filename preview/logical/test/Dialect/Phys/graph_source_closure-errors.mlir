// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --split-input-file --verify-diagnostics

module attributes {qlx.profiles = ["p2n", "p3"]} {
  phys.machine @arch {
    phys.resource_class @q {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  // expected-error@+1 {{projected communication calls must exactly match retained communication calls in the source protocol}}
  phys.graph @graph on @arch : () -> () attributes {source_protocol = @missing} {
    phys.return
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.protocol @p2 : () -> () {
    fabric.call @p2() : () -> ()
    fabric.protocol_return
  }
  phys.machine @arch {
    phys.resource_class @q {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  // expected-error@+1 {{recursive Fabric call graph cannot define physical call instances}}
  phys.graph @graph on @arch : () -> () attributes {source_protocol = @p2} {
    phys.return
  }
}
