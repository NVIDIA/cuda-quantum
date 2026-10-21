// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under    //
// the terms of the Apache License 2.0 which accompanies this distribution.    //
// ========================================================================== //

// RUN: qlx-opt %s --fabric-count='root=analysis' | FileCheck %s
// Shared fixture for Python's native projection and the native resource counter.
// CHECK: module attributes {fabric.counts = {
// CHECK-SAME: operation_counts = {measure_product = 1 : i64}
// CHECK-SAME: success_count = 0 : i64
// CHECK: fabric.detector {records = ["measure.read.outcome"]}
// CHECK: fabric.observable {idx = 0 : i64, records = ["measure.read.outcome"]}

module {
  fabric.code @c {
    distance = 1 : i64, k = 1 : i64, n = 1 : i64,
    partitions = {data = 1 : i64}, r = 0 : i64,
    lx = [array<i64: 0>], lz = [array<i64: 0>],
    stabilizer_basis = dense<> : tensor<0x2xi1>,
    anti_stabilizers = dense<> : tensor<0x2xi1>,
    logical_x_basis = dense<[[true, false]]> : tensor<1x2xi1>,
    logical_z_basis = dense<[[false, true]]> : tensor<1x2xi1>,
    gauge_x_basis = dense<> : tensor<0x2xi1>,
    gauge_z_basis = dense<> : tensor<0x2xi1>,
    encoding_clifford = dense<[[true, false], [false, true]]> : tensor<2x2xi1>
  }
  fabric.code_profile @cp {
    code = @c,
    effective_stabilizers = dense<> : tensor<0x2xi1>,
    decomposition = dense<> : tensor<0x0xi1>,
    kept_from_effective = dense<> : tensor<0x0xi1>,
    effective_metachecks = dense<> : tensor<0x0xi1>
  }
  fabric.encoding @e {
    block = "block0", code = @c, logical_ports = ["q0"], profile = @cp
  }
  qlx.action @idle : (!qlx.logical_qubit) -> !qlx.logical_qubit {kind = "idle"}
  fabric.objective @objective implements @idle
      : (!qlx.logical_qubit) -> !qlx.logical_qubit
  fabric.gadget_spec @spec for @objective
      : (!fabric.patch<@c, @e>) -> !fabric.patch<@c, @e> {
    encodings = [@e],
    ports = [{
      data_width = 1 : i64, direction = "inout", encoding = @e,
      input_state = "initialized", logical_arity = 1 : i64,
      name = "block", output_state = "initialized", ownership = "borrow",
      scratch_width = 0 : i64
    }]
  }
  fabric.gadget @measure(%patch: !fabric.patch<@c, @e>)
      -> !fabric.patch<@c, @e> {
    %next, %outcome = fabric.measure_product %patch {
      logical_indices = array<i64: 0>, patch_indices = array<i64: 0>,
      pauli_product = "Z", record = "read"
    } : (!fabric.patch<@c, @e>) -> (!fabric.patch<@c, @e>, i1)
    fabric.return %next : !fabric.patch<@c, @e>
  } {spec = @spec}
  fabric.gadget_profile @analysis for @measure attributes {
    inputs = [{endpoint = 0 : i64, name = "block", profile = @cp}],
    outputs = [{endpoint = 0 : i64, name = "block", profile = @cp}]
  } {
    fabric.detector {records = ["measure.read.outcome"]}
    fabric.observable {idx = 0 : i64, records = ["measure.read.outcome"]}
    fabric.profile_end
  }
  fabric.gadget @repeated(%patch: !fabric.patch<@c, @e>)
      -> !fabric.patch<@c, @e> {
    %result = cflow.repeat 2 iter(%outer : !fabric.patch<@c, @e> = %patch) {
      %inner = cflow.repeat 2 iter(%current : !fabric.patch<@c, @e> = %outer) {
        %next = fabric.call @measure(%current)
            : (!fabric.patch<@c, @e>) -> !fabric.patch<@c, @e>
        cflow.yield %next : !fabric.patch<@c, @e>
      }
      cflow.yield %inner : !fabric.patch<@c, @e>
    }
    fabric.return %result : !fabric.patch<@c, @e>
  } {spec = @spec}
  fabric.gadget @recursive(%patch: !fabric.patch<@c, @e>)
      -> !fabric.patch<@c, @e> {
    %next = fabric.call @recursive(%patch)
        : (!fabric.patch<@c, @e>) -> !fabric.patch<@c, @e>
    fabric.return %next : !fabric.patch<@c, @e>
  } {spec = @spec}
  fabric.gadget @dynamic(%patch: !fabric.patch<@c, @e>, %condition: i1)
      -> !fabric.patch<@c, @e> {
    %next = cflow.if %condition -> !fabric.patch<@c, @e> {
      cflow.yield %patch : !fabric.patch<@c, @e>
    } else {
      cflow.yield %patch : !fabric.patch<@c, @e>
    }
    fabric.return %next : !fabric.patch<@c, @e>
  }
}
