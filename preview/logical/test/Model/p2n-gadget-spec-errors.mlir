// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  fabric.code @c {
    distance = 3 : i64, k = 1 : i64, n = 1 : i64,
    partitions = {data = 1 : i64}, r = 0 : i64
  }
  fabric.code_profile @c_profile {code = @c}
  fabric.encoding @c_encoding {
    block = "block0", code = @c, logical_ports = ["q0"], profile = @c_profile
  }
  qlx.action @logical : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1) {
    kind = "idle"
  }
  fabric.objective @objective implements @logical
      : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)

  // The spec is internally well formed, but it changes the independently
  // derived realization boundary from inout/transform to output/prepare.
  fabric.gadget_spec @direction_mismatch for @objective
      : (!fabric.patch<@c, @c_encoding>)
          -> (!fabric.patch<@c, @c_encoding>, i1) {
    encodings = [@c_encoding],
    outcome_map = {
      constants = array<i64: 0>, input_syndromes = [[]],
      records = ["flag.outcome"], roles = [[]],
      rows = dense<1> : tensor<1x1xi1>
    },
    ports = [{
      data_width = 1 : i64,
      direction = "output",
      encoding = @c_encoding,
      input_state = "uninitialized",
      logical_arity = 1 : i64,
      name = "block",
      output_state = "initialized",
      ownership = "produce",
      scratch_width = 0 : i64
    }],
    record_schema = ["flag.outcome"]
  }
  // expected-error @+1 {{referenced gadget spec ports do not match realization_boundary}}
  fabric.gadget @realization(%arg0: !fabric.patch<@c, @c_encoding>)
      -> (!fabric.patch<@c, @c_encoding>, i1) {
    %next, %ok = fabric.measure_product %arg0 {
      logical_indices = array<i64: 0>, patch_indices = array<i64: 0>,
      pauli_product = "Z", record = "flag"
    } : (!fabric.patch<@c, @c_encoding>)
        -> (!fabric.patch<@c, @c_encoding>, i1)
    fabric.return %next, %ok : !fabric.patch<@c, @c_encoding>, i1
  } {
    spec = @direction_mismatch,
    realization_boundary = {
      flows = [{
        inputs = array<i64: 0>, kind = "transform", outputs = array<i64: 0>
      }],
      ports = [{
        data_width = 1 : i64,
        direction = "inout",
        encoding = @c_encoding,
        input_state = "initialized",
        logical_arity = 1 : i64,
        name = "block",
        output_state = "initialized",
        ownership = "borrow",
        scratch_width = 0 : i64
      }]
    }
  }
}
