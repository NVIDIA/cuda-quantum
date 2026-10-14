// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2n", "p3"]} {
  phys.action @swap {arity = 2 : i64, process = "{\22kind\22:\22builtin\22,\22name\22:\22swap\22,\22parameters\22:{}}"}
  phys.action @cx {arity = 2 : i64, process = "{\22kind\22:\22builtin\22,\22name\22:\22cx\22,\22parameters\22:{}}"}

  fabric.patch_graph @workload {
    root = @protocol,
    nodes = [
      {id = "patch0", code = @rep, slot = 0 : i64,
       patch_topology = @fixed::@patches},
      {id = "patch1", code = @rep, slot = 1 : i64,
       patch_topology = @fixed::@patches}
    ],
    interactions = [
      {id = "interaction0", action = "cx", patches = ["patch0", "patch1"],
       slots = array<i64: 0, 1>}
    ]
  }
  fabric.patch_mapping @workload_map {
    graph = @workload,
    assignments = [
      {id = "map.patch0", patch = "patch0", slot = 0 : i64,
       topology = @fixed::@patches},
      {id = "map.patch1", patch = "patch1", slot = 1 : i64,
       topology = @fixed::@patches}
    ]
  }

  phys.machine @fixed {
    phys.resource_class @qubits {
      kind = "qubit", count = 3 : i64, native_actions = [@swap, @cx]
    }
    phys.topology @line {
      kind = "explicit",
      num_nodes = 3 : i64,
      edges = [
        array<i64: 0, 1>,
        array<i64: 1, 2>
      ],
      strict
    }
    phys.patch_topology @patches {
      capacity = 2 : i64,
      carrier_groups = [array<i64: 0>, array<i64: 2>],
      categories = ["data", "data"],
      edges = [array<i64: 0, 1>],
      carrier_topology = @line,
      resource_class = @qubits
    }
    phys.qec_binding @compute {
      qec_region = @machine::@compute,
      resources = [@qubits],
      topology = @line,
      patch_topology = @patches
    }
  }
  phys.resource @q0 {kind = "qubit", resource_class = @qubits, index = 0 : i64}
  phys.resource @q1 {kind = "qubit", resource_class = @qubits, index = 1 : i64}
  phys.resource @q2 {kind = "qubit", resource_class = @qubits, index = 2 : i64}
  phys.graph @routed on @fixed :
      (!phys.state<@q0>, !phys.state<@q1>, !phys.state<@q2>) -> () {
  ^bb0(%q0: !phys.state<@q0>, %q1: !phys.state<@q1>, %q2: !phys.state<@q2>):
    %s0, %s1 = phys.apply @swap(%q0, %q1) {
      resources = [@q0, @q1], topology = @line, event_id = "swap0"
    } : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    %c1, %c2 = phys.apply @cx(%s1, %q2) {
      resources = [@q1, @q2], topology = @line, event_id = "cx0"
    } : (!phys.state<@q1>, !phys.state<@q2>) ->
        (!phys.state<@q1>, !phys.state<@q2>)
    %r0, %r1 = phys.apply @swap(%s0, %c1) {
      resources = [@q0, @q1], topology = @line, event_id = "swap1"
    } : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    phys.release %r0, %r1, %c2 {event_id = "release0"} :
      !phys.state<@q0>, !phys.state<@q1>, !phys.state<@q2>
    phys.return
  }
  phys.mapping @carrier_map {
    graph = @routed,
    source_graph = @workload,
    initial = [
      {role = "patch0.data[0]", resource = @q0},
      {role = "patch1.data[0]", resource = @q2}
    ],
    final = [
      {role = "patch0.data[0]", resource = @q0},
      {role = "patch1.data[0]", resource = @q2}
    ]
  }
  phys.routing @route {
    graph = @routed,
    topology = @line,
    steps = [{event = "route0", action = "cx", path = array<i64: 0, 1, 2>}]
  }
}

// CHECK-NOT: fabric.block_layout
// CHECK-NOT: fabric.patch_site
// CHECK-NOT: fabric.patch_link
// CHECK: fabric.patch_graph @workload
// CHECK: fabric.patch_mapping @workload_map
// CHECK: phys.topology @line
// CHECK-SAME: strict
// CHECK: phys.patch_topology @patches
// CHECK: carrier_groups = [array<i64: 0>, array<i64: 2>]
// CHECK: edges = [array<i64: 0, 1>]
// CHECK: phys.apply @swap
// CHECK-SAME: topology = @line
// CHECK: phys.apply @cx
// CHECK: phys.mapping @carrier_map
// CHECK: phys.routing @route
