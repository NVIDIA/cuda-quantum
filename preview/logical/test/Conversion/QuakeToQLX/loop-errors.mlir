// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file --verify-diagnostics %s

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__loop_local_owner_leak()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %carried = quake.null_wire
    %loop:2 = cc.loop while (
        (%w = %carried, %iv = %c0) -> (!quake.wire, i32)) {
      %go = arith.cmpi ne, %iv, %c2 : i32
      cc.condition %go(%w, %iv : !quake.wire, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32):
      %local = quake.null_wire
      // expected-error@+1 {{folded loop leaks an iteration-local quantum owner; measure or discard every body-local allocation before yielding}}
      cc.continue %w, %iv : !quake.wire, i32
    } step {
    ^bb0(%w: !quake.wire, %iv: i32):
      %next = arith.addi %iv, %c1 : i32
      cc.continue %w, %next : !quake.wire, i32
    }
    quake.sink %loop#0 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__reversed_slt()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %q = quake.null_wire
    // expected-error@+1 {{unsupported loop: condition must compare the induction directly to a constant trip count (affine-index loops unsupported)}}
    %loop:2 = cc.loop while (
        (%w = %q, %iv = %c0) -> (!quake.wire, i32)) {
      %go = arith.cmpi slt, %c5, %iv : i32
      cc.condition %go(%w, %iv : !quake.wire, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32):
      cc.continue %w, %iv : !quake.wire, i32
    } step {
    ^bb0(%w: !quake.wire, %iv: i32):
      %next = arith.addi %iv, %c1 : i32
      cc.continue %w, %next : !quake.wire, i32
    }
    quake.sink %loop#0 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__negative_slt_bound()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cm1 = arith.constant -1 : i32
    %q = quake.null_wire
    // expected-error@+1 {{unsupported loop: trip count must be nonnegative}}
    %loop:2 = cc.loop while (
        (%w = %q, %iv = %c0) -> (!quake.wire, i32)) {
      %go = arith.cmpi slt, %iv, %cm1 : i32
      cc.condition %go(%w, %iv : !quake.wire, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32):
      cc.continue %w, %iv : !quake.wire, i32
    } step {
    ^bb0(%w: !quake.wire, %iv: i32):
      %next = arith.addi %iv, %c1 : i32
      cc.continue %w, %next : !quake.wire, i32
    }
    quake.sink %loop#0 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__nonzero_loop_init()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %q = quake.null_wire
    // expected-error@+1 {{unsupported loop: induction must be initialized to 0}}
    %loop:2 = cc.loop while (
        (%w = %q, %iv = %c1) -> (!quake.wire, i32)) {
      %go = arith.cmpi slt, %iv, %c5 : i32
      cc.condition %go(%w, %iv : !quake.wire, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32):
      cc.continue %w, %iv : !quake.wire, i32
    } step {
    ^bb0(%w: !quake.wire, %iv: i32):
      %next = arith.addi %iv, %c1 : i32
      cc.continue %w, %next : !quake.wire, i32
    }
    quake.sink %loop#0 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__nonunit_loop_step()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c6 = arith.constant 6 : i32
    %q = quake.null_wire
    // expected-error@+1 {{unsupported loop: step must be `induction + 1`}}
    %loop:2 = cc.loop while (
        (%w = %q, %iv = %c0) -> (!quake.wire, i32)) {
      %go = arith.cmpi slt, %iv, %c6 : i32
      cc.condition %go(%w, %iv : !quake.wire, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32):
      cc.continue %w, %iv : !quake.wire, i32
    } step {
    ^bb0(%w: !quake.wire, %iv: i32):
      %next = arith.addi %iv, %c2 : i32
      cc.continue %w, %next : !quake.wire, i32
    }
    quake.sink %loop#0 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__dynamic_loop_bound()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %q = quake.null_wire
    // expected-error@+1 {{unsupported loop: condition must compare the induction directly to a constant trip count (affine-index loops unsupported)}}
    %loop:3 = cc.loop while (
        (%w = %q, %iv = %c0, %bound = %c5)
        -> (!quake.wire, i32, i32)) {
      %go = arith.cmpi slt, %iv, %bound : i32
      cc.condition %go(%w, %iv, %bound : !quake.wire, i32, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32, %bound: i32):
      cc.continue %w, %iv, %bound : !quake.wire, i32, i32
    } step {
    ^bb0(%w: !quake.wire, %iv: i32, %bound: i32):
      %next = arith.addi %iv, %c1 : i32
      cc.continue %w, %next, %bound : !quake.wire, i32, i32
    }
    quake.sink %loop#0 : !quake.wire
    return
  }
}
