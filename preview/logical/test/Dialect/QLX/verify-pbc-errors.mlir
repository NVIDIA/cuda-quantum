// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s --qlx-verify-pbc

/// (1) Cliffords must already be absorbed by qlx-to-pbc; any surviving gate
/// action means the program was never lowered.
// expected-error@+1 {{PBC form permits no Clifford/gate actions}}
module {
  qlx.program @bad_unabsorbed_clifford : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %h = qlx.apply #qlx.action<h>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%h)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

/// (2) Only canonical signed pi/4 is magic. A pi/2 rotation is a Clifford in
/// disguise and should have been absorbed instead of left as a rotation.
// expected-error@+1 {{PBC rotation is not canonical signed pi/4}}
module {
  qlx.program @bad_half_pi_rotation : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %angle = arith.constant 1.5707963267948966 : f64
    %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
        {parameters = {angle_pi_denom = 2 : i64, angle_pi_numer = 1 : i64,
                       sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%r)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

/// (3) PBC is a rotation phase followed by a measurement phase; a rotation
/// after the first measurement breaks that staging.
// expected-error@+1 {{PBC form requires all rotations before measurements}}
module {
  qlx.program @bad_rotation_after_measurement : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %m0:2 = qlx.instrument #qlx.instrument<mpp>(%q)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    %r = qlx.apply #qlx.action<pauli_rotation>(%m0#0, %angle)
        {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                       sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
    %m1:2 = qlx.instrument #qlx.instrument<mpp>(%r)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m1#1 : i1
  }
}

// -----

/// (4) The measured products must be simultaneously measurable. X then Z on
/// the same qubit anticommute, so no single measurement layer realizes them.
// expected-error@+1 {{PBC measured Pauli products do not pairwise commute}}
module {
  qlx.program @bad_noncommuting_measurements : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %m0:2 = qlx.instrument #qlx.instrument<mpp>(%q)
        {parameters = {sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    %m1:2 = qlx.instrument #qlx.instrument<mpp>(%m0#0)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m1#1 : i1
  }
}
