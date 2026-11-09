// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file --verify-diagnostics %s

// The pass requires an explicitly marked CUDA-Q entry point.
// expected-error@+1 {{convert-quake-to-qlx found no func.func with the cudaq-entrypoint attribute}}
module {
  func.func @helper() {
    return
  }
}

// -----

// A reachable helper must use the supported scalar reference ABI.
module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    quake.apply @helper() : () -> ()
    return
  }
  // expected-error@+2 {{reachable helper without scalar quantum inputs is unsupported}}
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @helper() {
    return
  }
}

// -----

module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    return
  }
  // expected-error@+1 {{unexpected top-level operation at the Quake-to-P0 boundary}}
  %orphan = qlx.prepare "zero" {allocation = 99 : i64, value_index = 0 : i64}
      : !qlx.logical_qubit
  qlx.discard %orphan : !qlx.logical_qubit
}

// -----

module {
  // expected-error@+2 {{Quake-to-P0 requires a specialized entry point with no quantum arguments}}
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__argument(%arg0: !quake.wire)
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    quake.sink %arg0 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+2 {{Quake-to-P0 currently supports only i1 entry-point results}}
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__bad_result() -> i32
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__reference()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    // expected-error@+1 {{Quake-to-P0 supports only value-semantics !quake.wire values plus statically sized helper cable boundaries}}
    %0 = quake.alloca !quake.ref
    quake.h %0 : (!quake.ref) -> ()
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__negated_control()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    // expected-error@+1 {{negated controls are outside the Quake-to-P0 contract}}
    %2:2 = quake.x [%0 neg [true]] %1
        : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    quake.sink %2#0 : !quake.wire
    quake.sink %2#1 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__excessive_controls()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %3 = quake.null_wire
    // expected-error@+1 {{unsupported gate shape}}
    %4:4 = quake.x [%0, %1, %2] %3
        : (!quake.wire, !quake.wire, !quake.wire, !quake.wire)
       -> (!quake.wire, !quake.wire, !quake.wire, !quake.wire)
    quake.sink %4#0 : !quake.wire
    quake.sink %4#1 : !quake.wire
    quake.sink %4#2 : !quake.wire
    quake.sink %4#3 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__fredkin()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    // expected-error@+1 {{controlled swap (Fredkin) is outside the Quake-to-P0 contract}}
    %3:3 = quake.swap [%0] %1, %2
        : (!quake.wire, !quake.wire, !quake.wire)
       -> (!quake.wire, !quake.wire, !quake.wire)
    quake.sink %3#0 : !quake.wire
    quake.sink %3#1 : !quake.wire
    quake.sink %3#2 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__controlled_rotation()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %angle = arith.constant 5.000000e-01 : f64
    %0 = quake.null_wire
    %1 = quake.null_wire
    // expected-error@+1 {{controlled rotations are outside the Quake-to-P0 contract}}
    %2:2 = quake.rz (%angle) [%0] %1
        : (f64, !quake.wire, !quake.wire)
       -> (!quake.wire, !quake.wire)
    quake.sink %2#0 : !quake.wire
    quake.sink %2#1 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__phased_rx()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %theta = arith.constant 2.500000e-01 : f64
    %phase = arith.constant 5.000000e-01 : f64
    %0 = quake.null_wire
    // expected-error@+1 {{is outside the typed Quake-to-P0 conversion contract}}
    %1 = quake.phased_rx (%theta, %phase) %0
        : (f64, f64, !quake.wire) -> !quake.wire
    quake.sink %1 : !quake.wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__non_measurement_return() -> i1
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %true = arith.constant true
    // expected-error@+1 {{returns a value that is not a discriminated measurement}}
    return %true : i1
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__unsupported_loop_predicate()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %q0 = quake.null_wire
    // expected-error@+1 {{unsupported loop: condition must use `arith.cmpi ne`, or `arith.cmpi slt` with the induction on the left}}
    %loop:2 = cc.loop while (
        (%w = %q0, %iv = %c0) -> (!quake.wire, i32)) {
      %go = arith.cmpi sle, %iv, %c5 : i32
      cc.condition %go(%w, %iv : !quake.wire, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32):
      %h = quake.h %w : (!quake.wire) -> !quake.wire
      cc.continue %h, %iv : !quake.wire, i32
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
  // expected-error@+2 {{leaves 1 live quantum value(s); measure, sink, or return every wire}}
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__entry_owner_leak()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %wire = quake.null_wire
    return
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__nested_adaptive() -> i1
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %m, %w = quake.mz %0
        : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %condition = quake.discriminate %m : (!cc.measure_handle) -> i1
    %result = cc.if (%condition) ((%arg = %1)) -> (!quake.wire) {
      // expected-error@+1 {{nested adaptive control is outside the Quake-to-P0 contract; flatten or outline the nested region}}
      %inner = cc.if (%condition) ((%nested = %arg)) -> (!quake.wire) {
        %x = quake.x %nested : (!quake.wire) -> !quake.wire
        cc.continue %x : !quake.wire
      } else {
        cc.continue %nested : !quake.wire
      }
      cc.continue %inner : !quake.wire
    } else {
      cc.continue %arg : !quake.wire
    }
    quake.sink %w : !quake.wire
    quake.sink %result : !quake.wire
    return %condition : i1
  }
}

// -----

// A Python `tuple[bool, int]` result.
module {
  // expected-error@+2 {{Quake-to-P0 currently supports only i1 entry-point results and structs of i1}}
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__mixed_tuple() -> !cc.struct<"tuple" {i1, i64}>
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = cc.undef !cc.struct<"tuple" {i1, i64}>
    return %0 : !cc.struct<"tuple" {i1, i64}>
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__constant_member() -> !cc.struct<"tuple" {i1}>
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %true = arith.constant true
    %0 = cc.undef !cc.struct<"tuple" {i1}>
    // expected-error@+1 {{Quake-to-P0 supports cc.insert_value only of a discriminated measurement}}
    %1 = cc.insert_value %0[0], %true : (!cc.struct<"tuple" {i1}>, i1) -> !cc.struct<"tuple" {i1}>
    return %1 : !cc.struct<"tuple" {i1}>
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__array_member() -> i1
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %m, %w = quake.mz %0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %bit = quake.discriminate %m : (!cc.measure_handle) -> i1
    %1 = cc.undef !cc.array<i1 x 1>
    // expected-error@+1 {{Quake-to-P0 supports cc.insert_value only into a struct built from cc.undef}}
    %2 = cc.insert_value %1[0], %bit : (!cc.array<i1 x 1>, i1) -> !cc.array<i1 x 1>
    quake.sink %w : !quake.wire
    return %bit : i1
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__nested_member() -> i1
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %m, %w = quake.mz %0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %bit = quake.discriminate %m : (!cc.measure_handle) -> i1
    %1 = cc.undef !cc.struct<{!cc.struct<{i1}>}>
    // expected-error@+1 {{Quake-to-P0 supports cc.insert_value only at a top-level struct member}}
    %2 = cc.insert_value %1[0, 0], %bit : (!cc.struct<{!cc.struct<{i1}>}>, i1) -> !cc.struct<{!cc.struct<{i1}>}>
    quake.sink %w : !quake.wire
    return %bit : i1
  }
}

// -----

module {
  // expected-error@+1 {{failed to legalize operation 'func.func'}}
  func.func @__nvqpp__mlirgen__partial_tuple() -> !cc.struct<"tuple" {i1, i1}>
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %m, %w = quake.mz %0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %bit = quake.discriminate %m : (!cc.measure_handle) -> i1
    %1 = cc.undef !cc.struct<"tuple" {i1, i1}>
    %2 = cc.insert_value %1[0], %bit : (!cc.struct<"tuple" {i1, i1}>, i1) -> !cc.struct<"tuple" {i1, i1}>
    quake.sink %w : !quake.wire
    // expected-error@+1 {{returns a value that is not a discriminated measurement or a struct of them}}
    return %2 : !cc.struct<"tuple" {i1, i1}>
  }
}

// -----

// The importer drops only the host registration of an entry point in the
// module.
module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    return
  }
  // expected-error@+1 {{unexpected top-level operation at the Quake-to-P0 boundary}}
  llvm.mlir.global external constant @other.kernelName("other\00") {addr_space = 0 : i32}
}

// -----

module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    return
  }
  // expected-error@+1 {{unexpected top-level operation at the Quake-to-P0 boundary}}
  llvm.mlir.global_ctors ctors = [@initialize], priorities = [17 : i32], data = [#llvm.zero]
  llvm.func @initialize() {
    llvm.return
  }
}

// -----

// A registration constructor may only call the runtime's registration
// functions.
module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    return
  }
  llvm.func @observable_effect()
  // expected-error@+1 {{unexpected top-level operation at the Quake-to-P0 boundary}}
  llvm.func @entry.kernelRegFunc() {
    llvm.call @observable_effect() : () -> ()
    llvm.return
  }
  llvm.mlir.global_ctors ctors = [@entry.kernelRegFunc], priorities = [17 : i32], data = [#llvm.zero]
}

// -----

module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    return
  }
  func.func private @observable_effect()
  // expected-error@+1 {{unexpected top-level operation at the Quake-to-P0 boundary}}
  llvm.func @entry.kernelRegFunc() {
    func.call @observable_effect() : () -> ()
    llvm.return
  }
  llvm.mlir.global_ctors ctors = [@entry.kernelRegFunc], priorities = [17 : i32], data = [#llvm.zero]
}

// -----

// A registration constructor may only pass its own kernel's names.
module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    return
  }
  func.func private @cudaqRegisterKernelName(!cc.ptr<i8>)
  // expected-error@+1 {{unexpected top-level operation at the Quake-to-P0 boundary}}
  llvm.func @entry.kernelRegFunc() {
    %0 = llvm.mlir.addressof @other.kernelName : !llvm.ptr
    %1 = cc.cast %0 : (!llvm.ptr) -> !cc.ptr<i8>
    func.call @cudaqRegisterKernelName(%1) : (!cc.ptr<i8>) -> ()
    llvm.return
  }
  llvm.mlir.global external constant @other.kernelName("other\00") {addr_space = 0 : i32}
}

// -----

// A `.kernelName` string must be the constant name of its kernel.
module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    return
  }
  // expected-error@+1 {{unexpected top-level operation at the Quake-to-P0 boundary}}
  llvm.mlir.global external constant @entry.kernelName("other\00") {addr_space = 0 : i32}
}

// -----

module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    return
  }
  // expected-error@+1 {{unexpected top-level operation at the Quake-to-P0 boundary}}
  llvm.mlir.global internal constant @entry.kernelName(42 : i64) : i64
}

// -----

module {
  func.func @__nvqpp__mlirgen__entry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    return
  }
  // expected-error@+1 {{unexpected top-level operation at the Quake-to-P0 boundary}}
  llvm.mlir.global external @entry.kernelName("entry\00") {addr_space = 0 : i32}
}
