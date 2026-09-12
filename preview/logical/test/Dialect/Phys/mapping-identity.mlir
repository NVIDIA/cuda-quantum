// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --split-input-file --verify-diagnostics

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @graph on @arch : () -> () {
    phys.return
  }
  phys.mapping @identity {
    graph = @graph,
    source_graph = @source,
    initial = [{role = "patch.data[0]", resource = @q0}],
    final = [{role = "patch.data[0]", resource = @q0}]
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @graph on @arch : () -> () {
    phys.return
  }
  // expected-error @+1 {{initial and final mappings must contain the same carrier roles}}
  phys.mapping @mismatch {
    graph = @graph,
    source_graph = @source,
    initial = [{role = "patch.data[0]", resource = @q0}],
    final = []
  }
}
