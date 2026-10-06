// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p0"]} {
  qlx.action @signed_x : (!qlx.logical_qubit) -> !qlx.logical_qubit {
    kind = "composite",
    clifford_action = #qlx.clifford_action<
      matrix = [1, 0, 0, 1], phases = [0, 1], ports = ["q"]>
  }
  qlx.action @empty_identity : () -> () {
    kind = "composite",
    clifford_action = #qlx.clifford_action<
      matrix = [], phases = [], ports = []>
  }
}

// CHECK: qlx.action @signed_x
// CHECK-SAME: clifford_action = #qlx.clifford_action<matrix = [1, 0, 0, 1], phases = [0, 1], ports = ["q"]>
// CHECK: qlx.action @empty_identity
// CHECK-SAME: clifford_action = #qlx.clifford_action<matrix = [], phases = [], ports = []>
