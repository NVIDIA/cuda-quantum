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
  // expected-error@+1 {{qlx-to-pbc does not yet lower ccz/ccx}}
  %r0, %r1, %r2 = qlx.apply #qlx.action<ccz>(%q0, %q1, %q2)
      : (!qlx.logical_qubit, !qlx.logical_qubit, !qlx.logical_qubit)
     -> (!qlx.logical_qubit, !qlx.logical_qubit, !qlx.logical_qubit)
  %m = qlx.measure #qlx.pauli<Z> %r0 : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// Preserve the current main-line fail-closed contract for the newer CCX
/// builtin when using the repeat-aware PBC lowering.
qlx.program @bad_ccx : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %q2 = qlx.prepare "zero" {allocation = 2 : i64, value_index = 2 : i64} : !qlx.logical_qubit
  // expected-error@+1 {{qlx-to-pbc does not yet lower ccz/ccx}}
  %r0, %r1, %r2 = qlx.apply #qlx.action<ccx>(%q0, %q1, %q2)
      : (!qlx.logical_qubit, !qlx.logical_qubit, !qlx.logical_qubit)
     -> (!qlx.logical_qubit, !qlx.logical_qubit, !qlx.logical_qubit)
  %m = qlx.measure #qlx.pauli<Z> %r0 : !qlx.logical_qubit -> i1
  qlx.discard %r1, %r2 : !qlx.logical_qubit, !qlx.logical_qubit
  qlx.return %m : i1
}

// -----

/// Logical idle is a memory workload, not an erasable Clifford identity.
qlx.program @bad_workload_idle : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  // expected-error@+1 {{qlx-to-pbc cannot erase workload-bearing idle actions}}
  %i = qlx.apply #qlx.action<idle>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %i : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// Builtin gates in the synthesized source are qubit-only operations. A
/// classical payload must fail during preflight instead of surviving until
/// source erasure with a live result.
qlx.program @bad_t_classical_payload : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %c = arith.constant 0.0 : f64
  // expected-error@+1 {{qlx-to-pbc requires this builtin action to have exactly 1 logical-qubit inputs and results and no classical payloads}}
  %t:2 = qlx.apply #qlx.action<t>(%q, %c)
      : (!qlx.logical_qubit, f64) -> (!qlx.logical_qubit, i1)
  qlx.discard %t#0 : !qlx.logical_qubit
  qlx.return %t#1 : i1
}

// -----

/// Folded multiplicity makes silent idle erasure especially consequential.
qlx.program @bad_repeated_workload_idle : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 4
      iter(%arg : !qlx.logical_qubit = %q) {
    // expected-error@+1 {{qlx-to-pbc cannot erase workload-bearing idle actions}}
    %i = qlx.apply #qlx.action<idle>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%i) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %t : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// The shared cflow dialect exposes a physical schedule join key, but PBC
/// normalization is a machine-free P0 transform and may split one repeat into
/// multiple chunks. It must not duplicate or invent physical event identity.
qlx.program @bad_repeat_event_id : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  // expected-error@+1 {{qlx-to-pbc does not accept physical schedule event_id on a P0 repeat}}
  %r = cflow.repeat 4 iter(%arg : !qlx.logical_qubit = %q) event_id = "scheduled.repeat" {
    %t = qlx.apply #qlx.action<t>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %t : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

qlx.program @bad_repeat_measurement : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 4
      iter(%arg : !qlx.logical_qubit = %q) {
    // expected-error@+1 {{qlx-to-pbc repeat bodies do not support measurement or feed-forward}}
    %m = qlx.measure #qlx.pauli<Z> %arg : !qlx.logical_qubit -> i1
    cflow.yield %arg : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

qlx.program @bad_repeat_capture : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 4
      iter(%arg : !qlx.logical_qubit = %q0) {
    // expected-error@+1 {{repeat bodies may use only explicitly carried logical qubits}}
    %c0, %c1 = qlx.apply #qlx.action<cx>(%arg, %q1)
        : (!qlx.logical_qubit, !qlx.logical_qubit)
       -> (!qlx.logical_qubit, !qlx.logical_qubit)
    cflow.yield %c0 : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.discard %q1 : !qlx.logical_qubit
  qlx.return %m : i1
}

// -----

/// Capturing the pre-loop owner is invalid even when it has the same global
/// logical identity as the explicit block argument.
qlx.program @bad_repeat_same_identity_capture : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 4
      iter(%arg : !qlx.logical_qubit = %q) {
    // expected-error@+1 {{apply input must consume the current logical-qubit SSA owner}}
    %t = qlx.apply #qlx.action<t>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %t : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

qlx.program @bad_duplicate_action_owner : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  // expected-error@+1 {{qlx-to-pbc requires distinct logical-qubit action operands}}
  %a, %b = qlx.apply #qlx.action<cx>(%q, %q)
      : (!qlx.logical_qubit, !qlx.logical_qubit)
     -> (!qlx.logical_qubit, !qlx.logical_qubit)
  %m = qlx.measure #qlx.pauli<Z> %a : !qlx.logical_qubit -> i1
  qlx.discard %b : !qlx.logical_qubit
  qlx.return %m : i1
}

// -----

/// PBC normalization must not invent an anonymous cleanup for a source owner
/// that the author left open.
qlx.program @bad_open_source_owner : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %t = qlx.apply #qlx.action<t>(%q0) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %t : !qlx.logical_qubit -> i1
  // expected-error@+1 {{qlx-to-pbc requires every source logical-qubit owner to be measured or explicitly discarded before return}}
  qlx.return %m : i1
}

// -----

/// This first slice is a terminal-measurement PBC form. It must fail before
/// rewriting instead of discarding and then returning the repeat result.
qlx.program @bad_repeat_logical_return : () -> !qlx.logical_qubit attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %r = cflow.repeat 2
      iter(%arg : !qlx.logical_qubit = %q) {
    %h0 = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %t = qlx.apply #qlx.action<t>(%h0) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %h1 = qlx.apply #qlx.action<h>(%t) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %h1 : !qlx.logical_qubit
  }
  // expected-error@+1 {{qlx-to-pbc does not support logical-qubit program returns}}
  qlx.return %r : !qlx.logical_qubit
}

// -----

/// Even without a source-level capture, a prior Clifford can widen a T axis
/// beyond the loop carries. The derived support must still be closed.
qlx.program @bad_repeat_derived_support_escape : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %c0, %c1 = qlx.apply #qlx.action<cx>(%q0, %q1)
      : (!qlx.logical_qubit, !qlx.logical_qubit)
     -> (!qlx.logical_qubit, !qlx.logical_qubit)
  %r = cflow.repeat 4
      iter(%arg : !qlx.logical_qubit = %c0) {
    %h0 = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    // expected-error@+1 {{qlx-to-pbc rotation support escapes the repeat carry set}}
    %t = qlx.apply #qlx.action<t>(%h0) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %h1 = qlx.apply #qlx.action<h>(%t) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %h1 : !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
  qlx.discard %c1 : !qlx.logical_qubit
  qlx.return %m : i1
}

// -----

qlx.program @bad_repeat_carry_permutation : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  // expected-error@+1 {{qlx-to-pbc requires position-preserving repeat yields}}
  %r0, %r1 = cflow.repeat 4
      iter(%a : !qlx.logical_qubit = %q0,
           %b : !qlx.logical_qubit = %q1) {
    cflow.yield %b, %a : !qlx.logical_qubit, !qlx.logical_qubit
  }
  %m = qlx.measure #qlx.pauli<Z> %r0 : !qlx.logical_qubit -> i1
  qlx.discard %r1 : !qlx.logical_qubit
  qlx.return %m : i1
}
