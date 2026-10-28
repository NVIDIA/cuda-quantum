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

/// A folded region is PBC form only when every action in it is already a
/// canonical Pauli rotation.
// expected-error@+1 {{PBC form permits no Clifford/gate actions}}
module {
  qlx.program @bad_repeat_clifford : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %r = cflow.repeat 8
        iter(%arg : !qlx.logical_qubit = %q) {
      %h = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
      cflow.yield %h : !qlx.logical_qubit
    }
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%r)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

/// A P0 certificate cannot admit the shared repeat's physical schedule join
/// key, even when the rotation-region shape is otherwise valid.
// expected-error@+1 {{PBC repeat cannot carry a physical schedule event_id in P0}}
module {
  qlx.program @bad_repeat_event_id : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %r = cflow.repeat 8 iter(%arg : !qlx.logical_qubit = %q) event_id = "scheduled.repeat" {
      cflow.yield %arg : !qlx.logical_qubit
    }
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%r)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.discard %m#0 : !qlx.logical_qubit
    qlx.return %m#1 : i1
  }
}

// -----

/// Repeats are part of the rotation phase and cannot follow an MPP.
// expected-error@+1 {{PBC form requires all rotations before measurements; found a repeat after a measurement}}
module {
  qlx.program @bad_repeat_after_measurement : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %m0:2 = qlx.instrument #qlx.instrument<mpp>(%q)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    %r = cflow.repeat 8
        iter(%arg : !qlx.logical_qubit = %m0#0) {
      cflow.yield %arg : !qlx.logical_qubit
    }
    %m1:2 = qlx.instrument #qlx.instrument<mpp>(%r)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m1#1 : i1
  }
}

// -----

/// Logical-qubit captures are hidden loop state and therefore invalid.
// expected-error@+1 {{PBC rotation input has no tracked qubit identity}}
module {
  qlx.program @bad_repeat_capture : () -> i1 attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %r = cflow.repeat 8
        iter(%arg : !qlx.logical_qubit = %q0) {
      %rot = qlx.apply #qlx.action<pauli_rotation>(%q1, %angle)
          {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                         sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
          : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
      cflow.yield %arg : !qlx.logical_qubit
    }
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%r)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.discard %q1 : !qlx.logical_qubit
    qlx.return %m#1 : i1
  }
}

// -----

/// Historical SSA values are not live loop owners after a rotation consumes
/// them.
// expected-error@+1 {{PBC rotation input must consume the current logical-qubit SSA owner}}
module {
  qlx.program @bad_repeat_stale_owner_reuse : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %r = cflow.repeat 8
        iter(%arg : !qlx.logical_qubit = %q) {
      %angle = arith.constant 0.78539816339744828 : f64
      %rot0 = qlx.apply #qlx.action<pauli_rotation>(%arg, %angle)
          {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                         sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
          : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
      %rot1 = qlx.apply #qlx.action<pauli_rotation>(%arg, %angle)
          {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                         sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
          : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
      cflow.yield %rot1 : !qlx.logical_qubit
    }
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%r)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

/// Pauli masks are bounded by the logical operand arity.
// expected-error@+1 {{PBC rotation Pauli masks exceed the logical operand arity}}
module {
  qlx.program @bad_rotation_mask_arity : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %rot = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
        {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                       sign = 1 : i64, x_mask = 2 : i64, z_mask = 0 : i64}}
        : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%rot)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC mpp Pauli masks exceed the logical operand arity}}
module {
  qlx.program @bad_mpp_mask_arity : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%q)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 2 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC rotation does not permit an identity-only Pauli product}}
module {
  qlx.program @bad_identity_rotation : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %rot = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
        {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                       sign = 1 : i64, x_mask = 0 : i64, z_mask = 0 : i64}}
        : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%rot)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC rotation requires i64 x_mask, z_mask, and sign parameters}}
module {
  qlx.program @bad_missing_rotation_mask : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %rot = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
        {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                       sign = 1 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%rot)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC rotation requires nonnegative Pauli masks}}
module {
  qlx.program @bad_negative_rotation_mask : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %rot = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
        {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                       sign = 1 : i64, x_mask = -1 : i64, z_mask = 0 : i64}}
        : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%rot)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC rotation must omit identity-only operand positions}}
module {
  qlx.program @bad_rotation_identity_spectator : () -> i1 attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %rot:2 = qlx.apply #qlx.action<pauli_rotation>(%q0, %q1, %angle)
        {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                       sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
        : (!qlx.logical_qubit, !qlx.logical_qubit, f64)
       -> (!qlx.logical_qubit, !qlx.logical_qubit)
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%rot#0)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.discard %rot#1 : !qlx.logical_qubit
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC mpp must omit identity-only operand positions}}
module {
  qlx.program @bad_mpp_identity_spectator : () -> i1 attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
    %m:3 = qlx.instrument #qlx.instrument<mpp>(%q0, %q1)
        {parameters = {sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
        : (!qlx.logical_qubit, !qlx.logical_qubit)
       -> (!qlx.logical_qubit, !qlx.logical_qubit, i1)
    qlx.discard %m#0, %m#1 : !qlx.logical_qubit, !qlx.logical_qubit
    qlx.return %m#2 : i1
  }
}

// -----

// expected-error@+1 {{PBC rotation requires its exact canonical parameter set}}
module {
  qlx.program @bad_rotation_extra_parameter : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %rot = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
        {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                       extra = 0 : i64, sign = 1 : i64,
                       x_mask = 1 : i64, z_mask = 0 : i64}}
        : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%rot)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC mpp requires its exact canonical parameter set}}
module {
  qlx.program @bad_mpp_extra_parameter : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%q)
        {parameters = {extra = 0 : i64, sign = 1 : i64,
                       x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC rotation logical operands must have distinct owners}}
module {
  qlx.program @bad_repeat_duplicate_rotation_owner : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %r = cflow.repeat 4
        iter(%arg : !qlx.logical_qubit = %q) {
      %angle = arith.constant 0.78539816339744828 : f64
      %rot:2 = qlx.apply #qlx.action<pauli_rotation>(%arg, %arg, %angle)
          {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                         sign = 1 : i64, x_mask = 1 : i64, z_mask = 2 : i64}}
          : (!qlx.logical_qubit, !qlx.logical_qubit, f64)
         -> (!qlx.logical_qubit, !qlx.logical_qubit)
      cflow.yield %rot#0 : !qlx.logical_qubit
    }
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%r)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC mpp logical operands must have distinct owners}}
module {
  qlx.program @bad_duplicate_mpp_owner : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %m:3 = qlx.instrument #qlx.instrument<mpp>(%q, %q)
        {parameters = {sign = 1 : i64, x_mask = 1 : i64, z_mask = 2 : i64}}
        : (!qlx.logical_qubit, !qlx.logical_qubit)
       -> (!qlx.logical_qubit, !qlx.logical_qubit, i1)
    qlx.return %m#2 : i1
  }
}

// -----

// expected-error@+1 {{PBC rotation requires exactly N logical-qubit inputs and results plus one trailing f64 angle input}}
module {
  qlx.program @bad_rotation_extra_result : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %rot:2 = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
        {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                       sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
        : (!qlx.logical_qubit, f64) -> (!qlx.logical_qubit, i1)
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%rot#0)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC mpp requires exactly N logical-qubit inputs, N logical-qubit results, and one trailing i1 outcome}}
module {
  qlx.program @bad_mpp_extra_input : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %extra = arith.constant 0.0 : f64
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%q, %extra)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit, f64) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC rotation is not canonical signed pi/4}}
module {
  qlx.program @bad_i32_exact_angle_metadata : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %angle = arith.constant 0.78539816339744828 : f64
    %rot = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
        {parameters = {angle_pi_denom = 4 : i32, angle_pi_numer = 1 : i32,
                       sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
        : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%rot)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.return %m#1 : i1
  }
}

// -----

// expected-error@+1 {{PBC form does not permit logical-qubit program returns}}
module {
  qlx.program @bad_repeat_logical_return : () -> !qlx.logical_qubit attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %r = cflow.repeat 4
        iter(%arg : !qlx.logical_qubit = %q) {
      %angle = arith.constant 0.78539816339744828 : f64
      %rot = qlx.apply #qlx.action<pauli_rotation>(%arg, %angle)
          {parameters = {angle_pi_denom = 4 : i64, angle_pi_numer = 1 : i64,
                         sign = 1 : i64, x_mask = 1 : i64, z_mask = 0 : i64}}
          : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
      cflow.yield %rot : !qlx.logical_qubit
    }
    qlx.return %r : !qlx.logical_qubit
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
    qlx.discard %m1#0 : !qlx.logical_qubit
    qlx.return %m1#1 : i1
  }
}

// -----

/// Terminal MPP begins the classical result phase; no later quantum workload
/// may be introduced.
// expected-error@+1 {{PBC form permits no logical-qubit preparation after the terminal measurement phase begins}}
module {
  qlx.program @bad_prepare_after_measurement : () -> i1 attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %m:2 = qlx.instrument #qlx.instrument<mpp>(%q0)
        {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.discard %m#0 : !qlx.logical_qubit
    %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
    qlx.discard %q1 : !qlx.logical_qubit
    qlx.return %m#1 : i1
  }
}

// -----

/// A classical return does not excuse a leaked logical owner.
// expected-error@+1 {{PBC form requires every logical-qubit owner to be measured and discarded before program return}}
module {
  qlx.program @bad_live_owner_at_return : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %out = arith.constant 0 : i1
    qlx.return %out : i1
  }
}
