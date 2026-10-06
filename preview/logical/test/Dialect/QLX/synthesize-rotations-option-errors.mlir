// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -verify-diagnostics %s --qlx-synthesize-rotations=precision=2.0

/// The `precision` option is validated before any rotation is inspected, so
/// this rejects the pipeline itself rather than a specific op. The explicit
/// `module` wrapper only exists to give that module-level diagnostic a source
/// location the verifier can be anchored to.

// expected-error@+1 {{default synthesis precision must be finite and in (0, 1)}}
module {
  qlx.program @unreachable_body : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %m = qlx.measure #qlx.pauli<Z> %q : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}
