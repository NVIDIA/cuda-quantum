// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: not qlx-opt %s -split-input-file 2>&1 | FileCheck %s

module {
  qlx.program @bad_profile : () -> () attributes {qlx.profile = "p1"} {
    qlx.return
  }
}

// CHECK: error: 'qlx.program' op requires qlx.stage = "p0"

// -----

module {
  qlx.program @unknown_action : (!qlx.logical_qubit) -> (!qlx.logical_qubit)
      attributes {qlx.profile = "p0"} {
  ^bb0(%q: !qlx.logical_qubit):
    %out = qlx.apply @missing(%q) : (!qlx.logical_qubit) -> (!qlx.logical_qubit)
    qlx.return %out : !qlx.logical_qubit
  }
}

// CHECK: error: 'qlx.apply' op references unknown qlx.action @missing

// -----

module {
  qlx.action @bad_clifford : (!qlx.logical_qubit) -> !qlx.logical_qubit {
    kind = "composite",
    clifford_action = #qlx.clifford_action<
      matrix = [1, 0, 1, 0], phases = [0, 0], ports = ["q"]>
  }
}

// CHECK: error: matrix does not preserve the binary symplectic form
