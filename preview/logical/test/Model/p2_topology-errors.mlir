// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error @+1 {{interaction patches must name nodes in the patch graph}}
  fabric.patch_graph @bad {
    root = @protocol,
    nodes = [{id = "patch0"}],
    interactions = [
      {id = "interaction0", action = "cx", patches = ["patch0", "patch1"]}
    ]
  }
}

// -----

module {
  phys.machine @fixed {
    phys.resource_class @qubits {
      kind = "qubit", count = 3 : i64, native_actions = []
    }
    phys.topology @line {
      kind = "explicit", num_nodes = 3 : i64,
      edges = [array<i64: 0, 2>], strict
    }
    phys.patch_topology @patches {
      capacity = 2 : i64,
      carrier_groups = [array<i64: 0>, array<i64: 2>],
      categories = ["", ""],
      edges = [array<i64: 0, 1>],
      carrier_topology = @line,
      resource_class = @qubits
    }
  }
  fabric.patch_graph @graph {
    root = @protocol,
    nodes = [{id = "patch0"}],
    interactions = []
  }
  // expected-error @+1 {{assignment slot 2 is absent from the patch topology}}
  fabric.patch_mapping @bad {
    graph = @graph,
    assignments = [{id = "map.patch0", patch = "patch0", slot = 2 : i64,
                    topology = @fixed::@patches}]
  }
}

// -----

module {
  phys.machine @missing_count {
    // expected-error @+1 {{topology coordinates require num_nodes}}
    phys.topology @bad {
      kind = "explicit", coordinates = [array<i64: 0, 0>]
    }
  }
}

// -----

module {
  phys.machine @partial_coordinates {
    // expected-error @+1 {{coordinates must contain one entry per topology node}}
    phys.topology @bad {
      kind = "explicit", num_nodes = 2 : i64,
      coordinates = [array<i64: 0, 0>], strict
    }
  }
}

// -----

module {
  phys.machine @bad_coordinate_dimension {
    // expected-error @+1 {{coordinates must be two-element i64 arrays}}
    phys.topology @bad {
      kind = "explicit", num_nodes = 1 : i64,
      coordinates = [array<i64: 0, 0, 0>], strict
    }
  }
}

// -----

module {
  phys.machine @duplicate_coordinates {
    // expected-error @+1 {{coordinates must be unique}}
    phys.topology @bad {
      kind = "explicit", num_nodes = 2 : i64,
      coordinates = [array<i64: 0, 0>, array<i64: 0, 0>], strict
    }
  }
}

// -----

module {
  phys.machine @fixed {
    phys.resource_class @qubits {
      kind = "qubit", count = 3 : i64, native_actions = []
    }
    phys.topology @line {
      kind = "explicit", num_nodes = 3 : i64,
      edges = [array<i64: 0, 2>], strict
    }
    // expected-error @+1 {{edges must exactly match structural paths through unassigned carriers}}
    phys.patch_topology @bad {
      capacity = 2 : i64,
      carrier_groups = [array<i64: 0>, array<i64: 2>],
      categories = ["", ""], edges = [],
      carrier_topology = @line, resource_class = @qubits
    }
  }
}
