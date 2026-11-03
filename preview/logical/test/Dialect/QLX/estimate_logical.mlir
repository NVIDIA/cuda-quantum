// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --qlx-estimate-logical='root=folded result=logical' | FileCheck %s

qlx.program @folded : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
      : !qlx.logical_qubit
  %r = cflow.repeat 5
      iter(%arg: !qlx.logical_qubit = %q) {
    %next = qlx.apply #qlx.action<t>(%arg)
        : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %next : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK: qlx.estimate_result @logical
// CHECK-SAME: data = {
// CHECK-SAME: action_depth_upper_bound = 7 : i64
// CHECK-SAME: actions = {qlx_standard_t = 5 : i64}
// CHECK-SAME: instruments = {qlx_standard_measure_z = 1 : i64, qlx_standard_prepare_zero = 1 : i64}
// CHECK-SAME: logical_qubits_peak = 1 : i64
// CHECK-SAME: synthesis_demand = {qlx_standard_t = 5 : i64}
// CHECK-SAME: root = @folded
// CHECK-SAME: schema = "qlx.logical-profile/v1"
// CHECK-SAME: tier = "logical"
