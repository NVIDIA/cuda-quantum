// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -verify-diagnostics %s --qlx-verify-clifford-t | FileCheck %s

/// The accepted set is {h, s, t, cx, idle}: the positive generators that
/// qlx-synthesize-rotations emits, not the whole Clifford+T gate set. `idle`
/// is the non-obvious member, admitted because it is a semantic no-op rather
/// than a gate outside the basis. Running under -verify-diagnostics also
/// asserts the verifier stays quiet on this input.

qlx.program @clifford_t_basis : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64} : !qlx.logical_qubit
  %h = qlx.apply #qlx.action<h>(%q0) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %s = qlx.apply #qlx.action<s>(%h) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %t = qlx.apply #qlx.action<t>(%s) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %i = qlx.apply #qlx.action<idle>(%t) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %c0, %c1 = qlx.apply #qlx.action<cx>(%i, %q1)
      : (!qlx.logical_qubit, !qlx.logical_qubit)
     -> (!qlx.logical_qubit, !qlx.logical_qubit)
  qlx.discard %c1 : !qlx.logical_qubit
  %m = qlx.measure #qlx.pauli<Z> %c0 : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

/// The verifier does not rewrite anything, so the program is echoed as-is.
// CHECK-LABEL: qlx.program @clifford_t_basis
//       CHECK:   qlx.apply #qlx.action<h>
//       CHECK:   qlx.apply #qlx.action<s>
//       CHECK:   qlx.apply #qlx.action<t>
//       CHECK:   qlx.apply #qlx.action<idle>
//       CHECK:   qlx.apply #qlx.action<cx>
