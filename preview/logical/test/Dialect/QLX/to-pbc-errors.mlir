// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s --qlx-to-pbc

/// Inputs outside the pass contract. Both are rejected on the offending op
/// rather than lowered approximately.

/// Rotations must already be synthesized: qlx-to-pbc commutes Cliffords, it
/// does not run the synthesizer itself.
qlx.program @bad_unsynthesized_rotation : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.3 : f64
  // expected-error@+1 {{qlx-to-pbc requires synthesized gates; run qlx-synthesize first}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// CCZ is neither a Clifford the frame can commute nor a T the pass can turn
/// into a pi/4 rotation, so there is no PBC form to produce.
qlx.program @bad_ccz : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %q2 = qlx.prepare "zero" {allocation = 2 : i64, value_index = 2 : i64} : !qlx.logical_qubit
  // expected-error@+1 {{qlx-to-pbc does not yet lower ccz}}
  %r0, %r1, %r2 = qlx.apply #qlx.action<ccz>(%q0, %q1, %q2)
      : (!qlx.logical_qubit, !qlx.logical_qubit, !qlx.logical_qubit)
     -> (!qlx.logical_qubit, !qlx.logical_qubit, !qlx.logical_qubit)
  %m = qlx.measure #qlx.pauli<Z> %r0 : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}
