// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK: fabric.objective @gauge_xx :
// CHECK: subsystem_fragment = {
// CHECK: outcome_result = 1 : i64
fabric.objective @gauge_xx
    : (!fabric.patch<@path4, @path4_default, @epoch0>)
      -> (!fabric.patch<@path4, @path4_default, @epoch0>, i1) {
  subsystem_fragment = {
    derivation = "body_exact",
    epoch_effect = "preserve",
    gauge_effect = "instrument",
    protected_effect = "instrument",
    steps = [{operation = "measure",
              outcome_result = 1 : i64,
              patch_indices = array<i64: 0, 0>,
              paulis = "XX",
              port_indices = array<i64: 0, 1>,
              port_kinds = ["gauge", "protected"],
              record = "mpp0",
              sign = 1 : i64}]
  }
}

// -----

// expected-error@+1 {{requires exactly one semantic source: logical or subsystem_fragment}}
fabric.objective @missing : () -> ()

// -----

// expected-error@+1 {{subsystem_fragment outcome_result must name an i1 result}}
fabric.objective @bad_result : () -> (i64) {
  subsystem_fragment = {
    derivation = "body_exact",
    epoch_effect = "preserve",
    gauge_effect = "instrument",
    protected_effect = "identity",
    steps = [{operation = "measure",
              outcome_result = 0 : i64,
              patch_indices = array<i64: 0>,
              paulis = "Z",
              port_indices = array<i64: 0>,
              port_kinds = ["gauge"],
              sign = 1 : i64}]
  }
}

// -----

module {
  fabric.code @c {
    distance = 3 : i64, n = 3 : i64, k = 1 : i64, r = 0 : i64,
    partitions = {data = 3 : i64},
    hz = [array<i64: 0, 1>, array<i64: 1, 2>],
    lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>]
  }
  fabric.code_profile @dynamic {
    code = @c,
    dynamic_phases = [
      {name = "only",
       measured_gauges = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
       instantaneous_stabilizers = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
       input_epoch = "only", output_epoch = "only",
       logical_action = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
       logical_map = {q0 = "q0"}}
    ],
    period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
    record_logicals = {
      names = ["q0"], gauge_pair_indices = array<i64>,
      x = dense<0> : tensor<1x1xi1>, z = dense<1> : tensor<1x1xi1>
    }
  }
  fabric.objective @cycle : () -> () {
    subsystem_fragment = {
      derivation = "body_exact",
      epoch_effect = "cycle",
      gauge_effect = "instrument",
      protected_effect = "unitary",
      profile = @dynamic,
      period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
      steps = [
        {operation = "measure_gauges",
         operators = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
         phase = "only", record = "only.g", input_epoch = "only"},
        {operation = "epoch_transition",
         from_epoch = "only", to_epoch = "only",
         logical_map = {q0 = "q0"}, evidence = "profile_exact",
         period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>}
      ]
    }
  }
}

// CHECK: fabric.objective @cycle :
// CHECK: epoch_effect = "cycle"
// CHECK: operation = "measure_gauges"
// CHECK: operation = "epoch_transition"

// -----

module {
  fabric.code @c {
    distance = 3 : i64, n = 3 : i64, k = 1 : i64, r = 0 : i64,
    partitions = {data = 3 : i64},
    hz = [array<i64: 0, 1>, array<i64: 1, 2>],
    lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>]
  }
  fabric.code_profile @dynamic {
    code = @c,
    dynamic_phases = [
      {name = "only",
       measured_gauges = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
       instantaneous_stabilizers = dense<[[0, 0, 0, 1, 1, 0]]> : tensor<1x6xi1>,
       input_epoch = "only", output_epoch = "only",
       logical_action = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
       logical_map = {q0 = "q0"}}
    ],
    period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
    record_logicals = {
      names = ["q0"], gauge_pair_indices = array<i64>,
      x = dense<0> : tensor<1x1xi1>, z = dense<1> : tensor<1x1xi1>
    }
  }
  // expected-error @+1 {{cyclic subsystem_fragment measurement step 0 contradicts dynamic profile phase 0}}
  fabric.objective @wrong_cycle_measurement : () -> () {
    subsystem_fragment = {
      derivation = "body_exact",
      epoch_effect = "cycle",
      gauge_effect = "instrument",
      protected_effect = "unitary",
      profile = @dynamic,
      period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>,
      steps = [
        {operation = "measure_gauges",
         operators = dense<[[0, 0, 0, 0, 1, 1]]> : tensor<1x6xi1>,
         phase = "only", record = "only.g", input_epoch = "only"},
        {operation = "epoch_transition",
         from_epoch = "only", to_epoch = "only",
         logical_map = {q0 = "q0"}, evidence = "profile_exact",
         period_closure = dense<[[1, 0], [0, 1]]> : tensor<2x2xi1>}
      ]
    }
  }
}
