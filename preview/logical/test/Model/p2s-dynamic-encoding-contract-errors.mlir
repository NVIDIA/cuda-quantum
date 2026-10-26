// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  fabric.code @c {distance = 3 : i64, n = 3 : i64, k = 1 : i64, r = 0 : i64,
    partitions = {data = 3 : i64},
    hz = [array<i64: 0, 1>, array<i64: 1, 2>],
    lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>]}
  fabric.code_profile @dynamic {code = @c, dynamic_phases = [
    {name = "only",
     measured_gauges = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     instantaneous_stabilizers = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     input_epoch = "only", output_epoch = "only",
     logical_action = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
     logical_map = {q0 = "q0"}}],
    period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
    record_logicals = {names = ["q0"], gauge_pair_indices = array<i64>,
      x = dense<0> : tensor<1x1xi1>, z = dense<1> : tensor<1x1xi1>}}
  fabric.encoding_epoch_schema @schema {phases = ["other"], initial = "other",
    transitions = ["other->other"],
    logical_maps = {"other->other" = {q0 = "q0"}}, periodic,
    closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>}
  // expected-error @+1 {{epoch_schema phases contradict the dynamic code profile}}
  fabric.encoding @encoding {block = "bad", code = @c, profile = @dynamic,
    logical_ports = ["q0"], epoch_schema = @schema, initial_epoch = @epoch}
  fabric.encoding_epoch @epoch {encoding = @encoding, schema = @schema,
    phase = "other", index = 0 : i64}
}

// -----

module {
  fabric.code @c {distance = 3 : i64, n = 3 : i64, k = 1 : i64, r = 0 : i64,
    partitions = {data = 3 : i64},
    hz = [array<i64: 0, 1>, array<i64: 1, 2>],
    lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>]}
  fabric.code_profile @dynamic {code = @c, dynamic_phases = [
    {name = "only",
     measured_gauges = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     instantaneous_stabilizers = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     input_epoch = "only", output_epoch = "only",
     logical_action = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
     logical_map = {q0 = "q0"}}],
    period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
    record_logicals = {names = ["q0"], gauge_pair_indices = array<i64>,
      x = dense<0> : tensor<1x1xi1>, z = dense<1> : tensor<1x1xi1>}}
  fabric.encoding_epoch_schema @schema {phases = ["only"], initial = "only",
    transitions = ["only->only", "only->only"],
    logical_maps = {"only->only" = {q0 = "q0"}}, periodic,
    closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>}
  // expected-error @+1 {{epoch_schema transitions contradict the dynamic code profile}}
  fabric.encoding @encoding {block = "bad", code = @c, profile = @dynamic,
    logical_ports = ["q0"], epoch_schema = @schema, initial_epoch = @epoch}
  fabric.encoding_epoch @epoch {encoding = @encoding, schema = @schema,
    phase = "only", index = 0 : i64}
}

// -----

module {
  fabric.code @c {distance = 3 : i64, n = 3 : i64, k = 1 : i64, r = 0 : i64,
    partitions = {data = 3 : i64},
    hz = [array<i64: 0, 1>, array<i64: 1, 2>],
    lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>]}
  fabric.code_profile @dynamic {code = @c, dynamic_phases = [
    {name = "only",
     measured_gauges = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     instantaneous_stabilizers = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     input_epoch = "only", output_epoch = "only",
     logical_action = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
     logical_map = {q0 = "q0"}}],
    period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
    record_logicals = {names = ["q0"], gauge_pair_indices = array<i64>,
      x = dense<0> : tensor<1x1xi1>, z = dense<1> : tensor<1x1xi1>}}
  fabric.encoding_epoch_schema @schema {phases = ["only"], initial = "only",
    transitions = ["only->only"],
    logical_maps = {"only->only" = {q0 = "renamed"}}, periodic,
    closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>}
  // expected-error @+1 {{epoch_schema logical_maps contradict the dynamic code profile}}
  fabric.encoding @encoding {block = "bad", code = @c, profile = @dynamic,
    logical_ports = ["q0"], epoch_schema = @schema, initial_epoch = @epoch}
  fabric.encoding_epoch @epoch {encoding = @encoding, schema = @schema,
    phase = "only", index = 0 : i64}
}

// -----

module {
  fabric.code @c {distance = 3 : i64, n = 3 : i64, k = 1 : i64, r = 0 : i64,
    partitions = {data = 3 : i64},
    hz = [array<i64: 0, 1>, array<i64: 1, 2>],
    lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>]}
  fabric.code_profile @dynamic {code = @c, dynamic_phases = [
    {name = "only",
     measured_gauges = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     instantaneous_stabilizers = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     input_epoch = "only", output_epoch = "only",
     logical_action = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
     logical_map = {q0 = "q0"}}],
    period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
    record_logicals = {names = ["q0"], gauge_pair_indices = array<i64>,
      x = dense<0> : tensor<1x1xi1>, z = dense<1> : tensor<1x1xi1>}}
  fabric.encoding_epoch_schema @schema {phases = ["only"], initial = "only",
    transitions = ["only->only"],
    logical_maps = {"only->only" = {q0 = "q0"}}, periodic,
    closure = dense<[[0, 1], [1, 0]]> : tensor<2x2xi1>}
  // expected-error @+1 {{epoch_schema closure contradicts the dynamic code profile}}
  fabric.encoding @encoding {block = "bad", code = @c, profile = @dynamic,
    logical_ports = ["q0"], epoch_schema = @schema, initial_epoch = @epoch}
  fabric.encoding_epoch @epoch {encoding = @encoding, schema = @schema,
    phase = "only", index = 0 : i64}
}

// -----

module {
  fabric.code @c {distance = 3 : i64, n = 3 : i64, k = 1 : i64, r = 0 : i64,
    partitions = {data = 3 : i64},
    hz = [array<i64: 0, 1>, array<i64: 1, 2>],
    lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>]}
  fabric.code_profile @dynamic {code = @c, dynamic_phases = [
    {name = "only",
     measured_gauges = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     instantaneous_stabilizers = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
     input_epoch = "only", output_epoch = "only",
     logical_action = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
     logical_map = {q0 = "q0"}}],
    period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
    record_logicals = {names = ["q0"], gauge_pair_indices = array<i64>,
      x = dense<0> : tensor<1x1xi1>, z = dense<1> : tensor<1x1xi1>}}
  fabric.encoding_epoch_schema @schema {phases = ["only"], initial = "only",
    transitions = ["only->only"],
    logical_maps = {"only->only" = {q0 = "q0"}}, periodic,
    closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>}
  fabric.encoding @encoding {block = "good", code = @c, profile = @dynamic,
    logical_ports = ["q0"], epoch_schema = @schema, initial_epoch = @epoch}
  fabric.encoding_epoch @epoch {encoding = @encoding, schema = @schema,
    phase = "only", index = 0 : i64}
  fabric.protocol @bad_map : (!fabric.patch<@c, @encoding, @epoch>)
      -> !fabric.patch<@c, @encoding, @epoch> {
  ^bb0(%patch: !fabric.patch<@c, @encoding, @epoch>):
    // expected-error @+1 {{logical_map contradicts dynamic profile edge 'only->only'}}
    %next = fabric.epoch_transition %patch to @epoch {
      evidence = "wrong", logical_map = {q0 = "renamed"}
    } : (!fabric.patch<@c, @encoding, @epoch>)
        -> !fabric.patch<@c, @encoding, @epoch>
    fabric.protocol_return %next : !fabric.patch<@c, @encoding, @epoch>
  }
}
