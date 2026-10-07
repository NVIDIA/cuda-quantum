// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file %s --qlx-to-pbc | FileCheck %s

/// No Clifford to absorb: the T becomes a Z-axis rotation and the Z
/// measurement keeps its basis (x_mask = 0, z_mask = 1 for both).
qlx.program @t_without_clifford : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %t = qlx.apply #qlx.action<t>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %t : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @t_without_clifford
//       CHECK:   qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 1 : i64

// -----

/// Source dispositions keep their grouping and reason. They remain separate
/// from the anonymous terminal cleanup of nondestructive MPP survivors.
qlx.program @preserve_discard_reasons : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %q2 = qlx.prepare "zero" {allocation = 2 : i64, value_index = 2 : i64} : !qlx.logical_qubit
  %t = qlx.apply #qlx.action<t>(%q0) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  qlx.discard %t {reason = "ancilla-cleanup"} : !qlx.logical_qubit
  qlx.discard %q1 {reason = "workspace-release"} : !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %q2 : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @preserve_discard_reasons
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//       CHECK:   qlx.discard {{.*}} {reason = "ancilla-cleanup"}
//       CHECK:   qlx.discard {{.*}} {reason = "workspace-release"}
//       CHECK:   qlx.discard {{.*}} : !qlx.logical_qubit
//       CHECK:   qlx.return

// -----

/// A widened rotation updates every qubit in its support. A following repeat
/// must take those current owners, including support qubits not targeted by
/// the source T action.
qlx.program @repeat_inits_follow_widened_rotation : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %c0, %c1 = qlx.apply #qlx.action<cx>(%q0, %q1)
      : (!qlx.logical_qubit, !qlx.logical_qubit)
     -> (!qlx.logical_qubit, !qlx.logical_qubit)
  %t = qlx.apply #qlx.action<t>(%c1) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %r0, %r1 = cflow.repeat 4
      iter(%a : !qlx.logical_qubit = %c0,
           %b : !qlx.logical_qubit = %t) {
    cflow.yield %a, %b : !qlx.logical_qubit, !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r1 : !qlx.logical_qubit -> i1
  qlx.discard %r0 : !qlx.logical_qubit
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @repeat_inits_follow_widened_rotation
//       CHECK:   %[[ROT:.*]]:2 = qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 3 : i64
//       CHECK:   cflow.repeat 4
//       CHECK:   iter(%[[ARG0:.*]]: !qlx.logical_qubit = %[[ROT]]#0,
//  CHECK-SAME:        %[[ARG1:.*]]: !qlx.logical_qubit = %[[ROT]]#1) {
//       CHECK:   qlx.instrument #qlx.instrument<mpp>

// -----

/// Pauli signs are preserved inside an identity-residual body. Conjugation by
/// the leading X turns the T axis into -Z even though the two X gates cancel.
qlx.program @signed_identity_residual_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 9
      iter(%arg : !qlx.logical_qubit = %q) {
    %x0 = qlx.apply #qlx.action<x>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%x0) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %x1 = qlx.apply #qlx.action<x>(%t) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %x1 : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @signed_identity_residual_repeat
//       CHECK:   cflow.repeat 9
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       sign = -1 : i64, x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64

// -----

/// Supported fixed points compose structurally: neither repeat is unrolled.
qlx.program @nested_identity_residual_repeats : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %outer = cflow.repeat 5
      iter(%outer_arg : !qlx.logical_qubit = %q) {
    %inner = cflow.repeat 7
        iter(%inner_arg : !qlx.logical_qubit = %outer_arg) {
      %h0 = qlx.apply #qlx.action<h>(%inner_arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
      %t = qlx.apply #qlx.action<t>(%h0) : (!qlx.logical_qubit) -> !qlx.logical_qubit
      %h1 = qlx.apply #qlx.action<h>(%t) : (!qlx.logical_qubit) -> !qlx.logical_qubit
      cflow.yield %h1 : !qlx.logical_qubit
    }
    cflow.yield %inner : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %outer : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @nested_identity_residual_repeats
//       CHECK:   cflow.repeat 5
//       CHECK:     cflow.repeat 7
//       CHECK:       qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:         x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>

// -----

/// The H is absorbed rather than emitted: conjugating by it turns both the
/// rotation and the measurement into their X-axis counterparts.
qlx.program @t_after_clifford : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %h = qlx.apply #qlx.action<h>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %t = qlx.apply #qlx.action<t>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %t : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @t_after_clifford
//       CHECK:   qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:     x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     x_mask = 1 : i64, z_mask = 0 : i64
/// PBC form admits no gate actions, so the H and T are gone.
//   CHECK-NOT:   #qlx.action<h>
//   CHECK-NOT:   #qlx.action<t>

// -----

/// A folded body whose Clifford residue is identity is lowered once and kept
/// folded. The two H gates cancel in the residual frame; the T site becomes
/// an X rotation template executed by cflow.repeat 1024 times.
qlx.program @identity_residual_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 1024
      iter(%arg : !qlx.logical_qubit = %q) {
    %h0 = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%h0) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %h1 = qlx.apply #qlx.action<h>(%t) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %h1 : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @identity_residual_repeat
//       CHECK:   %[[R:.*]] = cflow.repeat 1024
//       CHECK:     %[[ROT:.*]] = qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:     cflow.yield %[[ROT]] : !qlx.logical_qubit
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 1 : i64
//   CHECK-NOT:   #qlx.action<h>
//   CHECK-NOT:   #qlx.action<t>

// -----

/// Count one is exact even with a non-identity residual frame: the repeat is
/// preserved, its T becomes an X rotation, and the outgoing Z measurement is
/// conjugated to X by the absorbed H.
qlx.program @single_iteration_nonidentity_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 1
      iter(%arg : !qlx.logical_qubit = %q) {
    %h = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %t : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @single_iteration_nonidentity_repeat
//       CHECK:   cflow.repeat 1
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     x_mask = 1 : i64, z_mask = 0 : i64

// -----

/// A nonidentity residual is represented by one bounded phase period. H has
/// period two on the signed Pauli basis, so 1024 source iterations become 512
/// folded executions of the exact X,Z rotation-phase sequence.
qlx.program @period_two_even_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 1024
      iter(%arg : !qlx.logical_qubit = %q) {
    %h = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %t : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @period_two_even_repeat
//       CHECK:   %[[CYCLE:.*]] = cflow.repeat 512
//       CHECK:     %[[X:.*]] = qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:     %[[Z:.*]] = qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:     cflow.yield %[[Z]] : !qlx.logical_qubit
//       CHECK:   qlx.instrument #qlx.instrument<mpp>(%[[CYCLE]])
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 1 : i64
//   CHECK-NOT:   #qlx.action<h>
//   CHECK-NOT:   #qlx.action<t>

// -----

/// A large odd trip count becomes a folded period plus one count-one remainder.
/// The residual H in that remainder changes the terminal Z measurement to X.
qlx.program @period_two_odd_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 101
      iter(%arg : !qlx.logical_qubit = %q) {
    %h = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %t : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @period_two_odd_repeat
//       CHECK:   %[[CYCLE:.*]] = cflow.repeat 50
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:     %[[CYCLE_LAST:.*]] = qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:     cflow.yield %[[CYCLE_LAST]] : !qlx.logical_qubit
//       CHECK:   %[[REMAINDER:.*]] = cflow.repeat 1
//       CHECK:     %[[LAST:.*]] = qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:     cflow.yield %[[LAST]] : !qlx.logical_qubit
//       CHECK:   qlx.instrument #qlx.instrument<mpp>(%[[REMAINDER]])
//  CHECK-SAME:     x_mask = 1 : i64, z_mask = 0 : i64

// -----

/// Period detection includes Pauli signs. A residual X alternates the T axis
/// between -Z and +Z, returning to the signed canonical basis after two phases.
qlx.program @signed_period_two_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 100
      iter(%arg : !qlx.logical_qubit = %q) {
    %x = qlx.apply #qlx.action<x>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%x) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %t : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @signed_period_two_repeat
//       CHECK:   cflow.repeat 50
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       sign = -1 : i64, x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64

// -----

/// T-adjoint phases retain their negative rotation sign through periodic
/// rechunking; the H residual alternates their axes between X and Z.
qlx.program @adjoint_period_two_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 100
      iter(%arg : !qlx.logical_qubit = %q) {
    %h = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %tdg = qlx.apply #qlx.action<tdg>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %tdg : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @adjoint_period_two_repeat
//       CHECK:   cflow.repeat 50
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       sign = -1 : i64, x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       sign = -1 : i64, x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64

// -----

/// A CNOT residual also has period two. The first target-axis rotation is
/// conjugated onto both carried qubits; the second returns to target Z.
qlx.program @entangling_period_two_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %r0, %r1 = cflow.repeat 100
      iter(%a : !qlx.logical_qubit = %q0,
           %b : !qlx.logical_qubit = %q1) {
    %c0, %c1 = qlx.apply #qlx.action<cx>(%a, %b)
        : (!qlx.logical_qubit, !qlx.logical_qubit)
       -> (!qlx.logical_qubit, !qlx.logical_qubit)
    %t = qlx.apply #qlx.action<t>(%c1) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %c0, %t : !qlx.logical_qubit, !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r1 : !qlx.logical_qubit -> i1
  qlx.discard %r0 : !qlx.logical_qubit
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @entangling_period_two_repeat
//       CHECK:   cflow.repeat 50
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       x_mask = 0 : i64, z_mask = 3 : i64
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 1 : i64

// -----

/// Periodic normalization is innermost-first. The inner period-two loop is
/// split into phases, then the identity-residual outer loop remains folded.
qlx.program @nested_periodic_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %outer = cflow.repeat 3
      iter(%outer_arg : !qlx.logical_qubit = %q) {
    %inner = cflow.repeat 100
        iter(%inner_arg : !qlx.logical_qubit = %outer_arg) {
      %h = qlx.apply #qlx.action<h>(%inner_arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
      %t = qlx.apply #qlx.action<t>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
      cflow.yield %t : !qlx.logical_qubit
    }
    cflow.yield %inner : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %outer : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @nested_periodic_repeat
//       CHECK:   %[[OUTER:.*]] = cflow.repeat 3
//       CHECK:     %[[INNER:.*]] = cflow.repeat 50
//       CHECK:       qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:         x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:       qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:         x_mask = 0 : i64, z_mask = 1 : i64
//       CHECK:     cflow.yield %[[INNER]] : !qlx.logical_qubit
//       CHECK:   qlx.instrument #qlx.instrument<mpp>(%[[OUTER]])

// -----

/// A zero-count repeat has no frame effect, but its canonical body template is
/// retained so source T sites still correspond one-for-one with rotations.
qlx.program @zero_iteration_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 0
      iter(%arg : !qlx.logical_qubit = %q) {
    %h = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %t : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @zero_iteration_repeat
//       CHECK:   cflow.repeat 0
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       x_mask = 1 : i64, z_mask = 0 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 1 : i64

// -----

/// Entangling Clifford cancellation is checked over the full symplectic
/// basis, not qubit-by-qubit. The target T axis becomes Z(control) Z(target).
qlx.program @entangling_identity_residual_repeat : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %r0, %r1 = cflow.repeat 17
      iter(%a : !qlx.logical_qubit = %q0,
           %b : !qlx.logical_qubit = %q1) {
    %c0, %c1 = qlx.apply #qlx.action<cx>(%a, %b)
        : (!qlx.logical_qubit, !qlx.logical_qubit)
       -> (!qlx.logical_qubit, !qlx.logical_qubit)
    %t = qlx.apply #qlx.action<t>(%c1) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %d0, %d1 = qlx.apply #qlx.action<cx>(%c0, %t)
        : (!qlx.logical_qubit, !qlx.logical_qubit)
       -> (!qlx.logical_qubit, !qlx.logical_qubit)
    cflow.yield %d0, %d1 : !qlx.logical_qubit, !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r1 : !qlx.logical_qubit -> i1
  qlx.discard %r0 : !qlx.logical_qubit
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @entangling_identity_residual_repeat
//       CHECK:   cflow.repeat 17
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//  CHECK-SAME:       x_mask = 0 : i64, z_mask = 3 : i64
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
//  CHECK-SAME:     x_mask = 0 : i64, z_mask = 1 : i64

// -----

/// Independent periodic repeats are both normalized before the final program
/// analysis, including the second repeat's updated SSA init from the first.
qlx.program @sequential_periodic_repeats : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %first = cflow.repeat 100
      iter(%first_arg : !qlx.logical_qubit = %q) {
    %first_h = qlx.apply #qlx.action<h>(%first_arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %first_t = qlx.apply #qlx.action<t>(%first_h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %first_t : !qlx.logical_qubit
  }
  %second = cflow.repeat 100
      iter(%second_arg : !qlx.logical_qubit = %first) {
    %second_h = qlx.apply #qlx.action<h>(%second_arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %second_t = qlx.apply #qlx.action<t>(%second_h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %second_t : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %second : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// CHECK-LABEL: qlx.program @sequential_periodic_repeats
//       CHECK:   cflow.repeat 50
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//       CHECK:   cflow.repeat 50
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//       CHECK:     qlx.apply #qlx.action<pauli_rotation>
//       CHECK:   qlx.instrument #qlx.instrument<mpp>
