// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file %s | FileCheck %s --implicit-check-not=quake. --implicit-check-not=llvm. --implicit-check-not=func.

module {
  func.func @__nvqpp__mlirgen__measurement_axes() -> (i1, i1, i1)
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %mx, %wx = quake.mx %0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %my, %wy = quake.my %1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %mz, %wz = quake.mz %2 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %x = quake.discriminate %mx : (!cc.measure_handle) -> i1
    %y = quake.discriminate %my : (!cc.measure_handle) -> i1
    %z = quake.discriminate %mz : (!cc.measure_handle) -> i1
    quake.sink %wx : !quake.wire
    quake.sink %wy : !quake.wire
    quake.sink %wz : !quake.wire
    return %x, %y, %z : i1, i1, i1
  }
}

// CHECK-LABEL: qlx.program @measurement_axes : () -> (i1, i1, i1)
// CHECK: %[[MEAS_QX:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[MEAS_QY:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[MEAS_QZ:.*]] = qlx.prepare "zero" {allocation = 2 : i64
// CHECK: %[[MEAS_X:.*]] = qlx.measure <X> %[[MEAS_QX]]
// CHECK: %[[MEAS_Y:.*]] = qlx.measure <Y> %[[MEAS_QY]]
// CHECK: %[[MEAS_Z:.*]] = qlx.measure <Z> %[[MEAS_QZ]]
// CHECK: qlx.return %[[MEAS_X]], %[[MEAS_Y]], %[[MEAS_Z]] : i1, i1, i1

// -----

// CUDA-Q also accepts the legacy !quake.measure result form. Keep this case
// explicit: both upstream measurement representations have the same P0
// meaning and must cross the ingestion boundary.
module {
  func.func @__nvqpp__mlirgen__legacy_measurement() -> i1
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %m, %w = quake.mz %0 : (!quake.wire) -> (!quake.measure, !quake.wire)
    %bit = quake.discriminate %m : (!quake.measure) -> i1
    quake.sink %w : !quake.wire
    return %bit : i1
  }
}

// CHECK-LABEL: qlx.program @legacy_measurement : () -> i1
// CHECK: %[[LEGACY_Q:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[LEGACY_M:.*]] = qlx.measure <Z> %[[LEGACY_Q]]
// CHECK: qlx.return %[[LEGACY_M]] : i1

// -----

// CUDA-Q compiles this Python kernel for a Logical target to the module
// below, shown without module attributes and with a shorter unique suffix:
//
//   @cudaq.kernel
//   def flip() -> bool:
//       q = cudaq.qubit()
//       x(q)
//       return mz(q)
//
// A kernel that returns a value also gets host launch functions and a host
// runtime registration. P0 keeps only the program.
module {
  func.func @__nvqpp__mlirgen__flip..0x1() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.x %0 : (!quake.wire) -> !quake.wire
    %measOut, %wires = quake.mz %1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %2 = quake.discriminate %measOut : (!cc.measure_handle) -> i1
    quake.sink %wires : !quake.wire
    return %2 : i1
  }
  func.func private @cudaqRegisterArgsCreator(!cc.ptr<i8>, !cc.ptr<i8>)
  func.func private @cudaqRegisterKernelName(!cc.ptr<i8>)
  func.func private @malloc(i64) -> !cc.ptr<i8>
  func.func private @__nvqpp_vector_bool_free_temporary_initlists(!cc.ptr<i8>)
  func.func private @__nvqpp_cleanup_arrays()
  llvm.mlir.global external constant @flip..0x1.kernelName("flip..0x1\00") {addr_space = 0 : i32}
  func.func @flip..0x1.returnOffset() -> i64 {
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
  func.func @flip..0x1.thunk(%arg0: !cc.ptr<i8>, %arg1: i1) -> !cc.struct<{!cc.ptr<i8>, i64}> {
    %c0_i64 = arith.constant 0 : i64
    %0 = call @__nvqpp__mlirgen__flip..0x1() {no_inline} : () -> i1
    call @__nvqpp_cleanup_arrays() : () -> ()
    %1 = cc.cast %arg0 : (!cc.ptr<i8>) -> !cc.ptr<i1>
    cc.store %0, %1 : !cc.ptr<i1>
    %2 = cc.cast %c0_i64 : (i64) -> !cc.ptr<i8>
    %3 = cc.undef !cc.struct<{!cc.ptr<i8>, i64}>
    %4 = cc.insert_value %3[0], %2 : (!cc.struct<{!cc.ptr<i8>, i64}>, !cc.ptr<i8>) -> !cc.struct<{!cc.ptr<i8>, i64}>
    %5 = cc.insert_value %4[1], %c0_i64 : (!cc.struct<{!cc.ptr<i8>, i64}>, i64) -> !cc.struct<{!cc.ptr<i8>, i64}>
    return %5 : !cc.struct<{!cc.ptr<i8>, i64}>
  }
  func.func @flip..0x1.argsCreator(%arg0: !cc.ptr<!cc.ptr<i8>>, %arg1: !cc.ptr<!cc.ptr<i8>>) -> i64 {
    %c0_i64 = arith.constant 0 : i64
    %0 = cc.cast %c0_i64 : (i64) -> !cc.ptr<i8>
    %1 = cc.sizeof !cc.struct<{i1}> : i64
    %2 = call @malloc(%1) : (i64) -> !cc.ptr<i8>
    %3 = cc.cast %0 : (!cc.ptr<i8>) -> i64
    %4 = arith.cmpi ne, %3, %c0_i64 : i64
    cc.if(%4) {
      func.call @__nvqpp_vector_bool_free_temporary_initlists(%0) : (!cc.ptr<i8>) -> ()
    }
    cc.store %2, %arg1 : !cc.ptr<!cc.ptr<i8>>
    return %1 : i64
  }
  llvm.func @flip..0x1.kernelRegFunc() {
    %f = func.constant @flip..0x1.argsCreator : (!cc.ptr<!cc.ptr<i8>>, !cc.ptr<!cc.ptr<i8>>) -> i64
    %0 = llvm.mlir.addressof @flip..0x1.kernelName : !llvm.ptr
    %1 = cc.cast %0 : (!llvm.ptr) -> !cc.ptr<i8>
    func.call @cudaqRegisterKernelName(%1) : (!cc.ptr<i8>) -> ()
    %2 = cc.func_ptr %f : ((!cc.ptr<!cc.ptr<i8>>, !cc.ptr<!cc.ptr<i8>>) -> i64) -> !cc.ptr<i8>
    func.call @cudaqRegisterArgsCreator(%1, %2) : (!cc.ptr<i8>, !cc.ptr<i8>) -> ()
    llvm.return
  }
  llvm.mlir.global_ctors ctors = [@flip..0x1.kernelRegFunc], priorities = [17 : i32], data = [#llvm.zero]
}

// CHECK-LABEL: qlx.program @flip : () -> i1
// CHECK: %[[FLIP_Q:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[FLIP_X:.*]] = qlx.apply #qlx.action<x>(%[[FLIP_Q]])
// CHECK: %[[FLIP_M:.*]] = qlx.measure <Z> %[[FLIP_X]]
// CHECK: qlx.return %[[FLIP_M]] : i1

// -----

// The Python frontend returns `tuple[bool, ...]` as one `!cc.struct`, and P0
// returns one result per member, in member order. This is the entry point
// that CUDA-Q compiles for a Logical target from
//
//   @cudaq.kernel
//   def bell() -> tuple[bool, bool]:
//       q = cudaq.qvector(2)
//       h(q[0])
//       x.ctrl(q[0], q[1])
//       first = mz(q[0])
//       second = mz(q[1])
//       return second, first
module {
  func.func @__nvqpp__mlirgen__bell..0x1() -> !cc.struct<"tuple" {i1, i1}> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.h %0 : (!quake.wire) -> !quake.wire
    %3:2 = quake.x [%2] %1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    %measOut, %wires = quake.mz %3#0 name "first" : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %measOut_0, %wires_1 = quake.mz %3#1 name "second" : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %4 = quake.discriminate %measOut_0 : (!cc.measure_handle) -> i1
    %5 = quake.discriminate %measOut : (!cc.measure_handle) -> i1
    %6 = cc.undef !cc.struct<"tuple" {i1, i1}>
    %7 = cc.insert_value %6[0], %4 : (!cc.struct<"tuple" {i1, i1}>, i1) -> !cc.struct<"tuple" {i1, i1}>
    %8 = cc.insert_value %7[1], %5 : (!cc.struct<"tuple" {i1, i1}>, i1) -> !cc.struct<"tuple" {i1, i1}>
    quake.sink %wires : !quake.wire
    quake.sink %wires_1 : !quake.wire
    return %8 : !cc.struct<"tuple" {i1, i1}>
  }
}

// CHECK-LABEL: qlx.program @bell : () -> (i1, i1)
// CHECK: %[[BELL_FIRST:.*]] = qlx.measure <Z>
// CHECK: %[[BELL_SECOND:.*]] = qlx.measure <Z>
// CHECK: qlx.return %[[BELL_SECOND]], %[[BELL_FIRST]] : i1, i1
