// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s --qlx-verify-clifford-t

/// TODO: evaluate whether rejecting X here is the behavior we want. X is a
/// Clifford and needs no decomposition to be Clifford+T, so this diagnostic
/// is really enforcing the positive-generator normal form that
/// qlx-synthesize-rotations produces (it rewrites x into h,s,s,h) rather
/// than gate-set membership. The same applies to y, z, sdg, tdg and cz. If
/// the pass is broadened to true Clifford+T membership, this case flips to
/// an accepted one -- but the pauli_rotation case below must keep failing,
/// since an unsynthesized rotation is off the Clifford+T lattice.
qlx.program @bad_builtin_outside_basis : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  // expected-error@+1 {{remains outside the Clifford+T gate set after legalization}}
  %x = qlx.apply #qlx.action<x>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %x : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// A surviving rotation means synthesis never ran: its angle is off the
/// Clifford+T lattice until it has been approximated.
qlx.program @bad_unsynthesized_rotation : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %angle = arith.constant 0.3 : f64
  // expected-error@+1 {{remains outside the Clifford+T gate set after legalization}}
  %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
      {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
      : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// A symbol-referenced action carries no built-in semantics, so the verifier
/// cannot classify it and refuses it outright.
qlx.action @custom_gate : (!qlx.logical_qubit) -> !qlx.logical_qubit {kind = "unitary"}

qlx.program @bad_symbolic_action : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  // expected-error@+1 {{is not a built-in action legal in the Clifford+T gate set}}
  %r = qlx.apply @custom_gate(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}
