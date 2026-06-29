/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Starting from C++, mimic the relevant JIT shape and check that
// parent-dependent nested loops over quantum data are unrolled before wire
// assignment.
// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --canonicalize --cc-loop-normalize --expand-measurements --cc-loop-unroll=unroll-only-wire-blocking-loops=true --add-dealloc --combine-quantum-alloc --canonicalize --factor-quantum-alloc --memtoreg --add-wireset --assign-wire-indices | FileCheck %s
// clang-format on

#include <cudaq.h>

// Triangular pair loops index quantum data with induction variables. Both loops
// must unroll so the controlled operations and whole-register measurement can
// lower to wires.
__qpu__ std::vector<bool> kernel() {
  constexpr int N = 4;
  cudaq::qvector q(N);
  // Expected to unroll: `i` indexes quantum data and bounds the inner loop.
  for (int i = 0; i < N - 1; ++i) {
    // Expected to unroll: `j` indexes quantum data.
    for (int j = i + 1; j < N; ++j) {
      x<cudaq::ctrl>(q[i], q[j]);
    }
  }
  return cudaq::to_bools(mz(q));
}

// CHECK-LABEL:   quake.wire_set @wires[2147483647] attributes {sym_visibility = "private"}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel._Z6kernelv() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-NOT:       cc.loop
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK-NOT:       quake.subveq
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : i64
// CHECK:           %[[Q0:.*]] = quake.borrow_wire @wires[0] : !quake.wire
// CHECK:           %[[Q1:.*]] = quake.borrow_wire @wires[1] : !quake.wire
// CHECK:           %[[Q2:.*]] = quake.borrow_wire @wires[2] : !quake.wire
// CHECK:           %[[Q3:.*]] = quake.borrow_wire @wires[3] : !quake.wire
// CHECK-NOT:       cc.loop
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK-NOT:       quake.subveq
// CHECK:           %[[X01:.*]]:2 = quake.x {{\[}}%[[Q0]]] %[[Q1]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[X02:.*]]:2 = quake.x {{\[}}%[[X01]]#0] %[[Q2]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[X03:.*]]:2 = quake.x {{\[}}%[[X02]]#0] %[[Q3]] : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[X12:.*]]:2 = quake.x {{\[}}%[[X01]]#1] %[[X02]]#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[X13:.*]]:2 = quake.x {{\[}}%[[X12]]#0] %[[X03]]#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           %[[X23:.*]]:2 = quake.x {{\[}}%[[X12]]#1] %[[X13]]#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK-NOT:       cc.loop
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK-NOT:       quake.subveq
// CHECK:           %[[BITS:.*]] = cc.alloca !cc.array<i8 x 4>
// CHECK:           %[[M0:.*]], %[[W0:.*]] = quake.mz %[[X03]]#0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[B0:.*]] = quake.discriminate %[[M0]] : (!cc.measure_handle) -> i1
// CHECK:           %[[BITSPTR:.*]] = cc.cast %[[BITS]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[BYTE0:.*]] = cc.cast unsigned %[[B0]] : (i1) -> i8
// CHECK:           cc.store %[[BYTE0]], %[[BITSPTR]] : !cc.ptr<i8>
// CHECK:           %[[M1:.*]], %[[W1:.*]] = quake.mz %[[X13]]#0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[B1:.*]] = quake.discriminate %[[M1]] : (!cc.measure_handle) -> i1
// CHECK:           %[[PTR1:.*]] = cc.compute_ptr %[[BITS]][1] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[BYTE1:.*]] = cc.cast unsigned %[[B1]] : (i1) -> i8
// CHECK:           cc.store %[[BYTE1]], %[[PTR1]] : !cc.ptr<i8>
// CHECK:           %[[M2:.*]], %[[W2:.*]] = quake.mz %[[X23]]#0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[B2:.*]] = quake.discriminate %[[M2]] : (!cc.measure_handle) -> i1
// CHECK:           %[[PTR2:.*]] = cc.compute_ptr %[[BITS]][2] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[BYTE2:.*]] = cc.cast unsigned %[[B2]] : (i1) -> i8
// CHECK:           cc.store %[[BYTE2]], %[[PTR2]] : !cc.ptr<i8>
// CHECK:           %[[M3:.*]], %[[W3:.*]] = quake.mz %[[X23]]#1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[B3:.*]] = quake.discriminate %[[M3]] : (!cc.measure_handle) -> i1
// CHECK:           %[[PTR3:.*]] = cc.compute_ptr %[[BITS]][3] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[BYTE3:.*]] = cc.cast unsigned %[[B3]] : (i1) -> i8
// CHECK:           cc.store %[[BYTE3]], %[[PTR3]] : !cc.ptr<i8>
// CHECK:           %[[COPY_SRC:.*]] = cc.cast %[[BITS]] : (!cc.ptr<!cc.array<i8 x 4>>) -> !cc.ptr<i8>
// CHECK:           %[[COPY:.*]] = call @__nvqpp_vectorCopyCtor(%[[COPY_SRC]], %[[CONSTANT_1]], %[[CONSTANT_0]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[RESULT:.*]] = cc.stdvec_init %[[COPY]], %[[CONSTANT_1]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK-NOT:       cc.loop
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK-NOT:       quake.subveq
// CHECK:           quake.return_wire %[[W0]] : !quake.wire
// CHECK:           quake.return_wire %[[W1]] : !quake.wire
// CHECK:           quake.return_wire %[[W2]] : !quake.wire
// CHECK:           quake.return_wire %[[W3]] : !quake.wire
// CHECK:           return %[[RESULT]] : !cc.stdvec<i1>
// CHECK:         }

// The inner loop unrolls first, exposing outer-induction-variable-dependent
// flattened indices. The same pass must then unroll the outer loop when it
// reaches a fixed point.
__qpu__ std::vector<bool> rectangular_flattened_index() {
  constexpr int N = 2;
  constexpr int M = 3;
  cudaq::qvector q(N * M);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
      x(q[i * M + j]);
  return cudaq::to_bools(mz(q));
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_rectangular_flattened_index._Z27rectangular_flattened_indexv() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-NOT:       cc.loop
// CHECK-NOT:       {{quake\.alloca|quake\.extract_ref|quake\.subveq|!quake\.ref|!quake\.veq}}
// CHECK:           %[[Q0:.*]] = quake.borrow_wire @wires[0] : !quake.wire
// CHECK:           %[[Q1:.*]] = quake.borrow_wire @wires[1] : !quake.wire
// CHECK:           %[[Q2:.*]] = quake.borrow_wire @wires[2] : !quake.wire
// CHECK:           %[[Q3:.*]] = quake.borrow_wire @wires[3] : !quake.wire
// CHECK:           %[[Q4:.*]] = quake.borrow_wire @wires[4] : !quake.wire
// CHECK:           %[[Q5:.*]] = quake.borrow_wire @wires[5] : !quake.wire
// CHECK-NOT:       cc.loop
// CHECK-NOT:       {{quake\.alloca|quake\.extract_ref|quake\.subveq|!quake\.ref|!quake\.veq}}
// CHECK:           %[[X0:.*]] = quake.x %[[Q0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X1:.*]] = quake.x %[[Q1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X2:.*]] = quake.x %[[Q2]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X3:.*]] = quake.x %[[Q3]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X4:.*]] = quake.x %[[Q4]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X5:.*]] = quake.x %[[Q5]] : (!quake.wire) -> !quake.wire
// CHECK-NOT:       cc.loop
// CHECK-NOT:       {{quake\.alloca|quake\.extract_ref|quake\.subveq|!quake\.ref|!quake\.veq}}
// CHECK:           %[[M0:.*]], %[[W0:.*]] = quake.mz %[[X0]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[M1:.*]], %[[W1:.*]] = quake.mz %[[X1]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[M2:.*]], %[[W2:.*]] = quake.mz %[[X2]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[M3:.*]], %[[W3:.*]] = quake.mz %[[X3]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[M4:.*]], %[[W4:.*]] = quake.mz %[[X4]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[M5:.*]], %[[W5:.*]] = quake.mz %[[X5]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK-NOT:       cc.loop
// CHECK-NOT:       {{quake\.alloca|quake\.extract_ref|quake\.subveq|!quake\.ref|!quake\.veq}}
// CHECK:           quake.return_wire %[[W0]] : !quake.wire
// CHECK:           quake.return_wire %[[W1]] : !quake.wire
// CHECK:           quake.return_wire %[[W2]] : !quake.wire
// CHECK:           quake.return_wire %[[W3]] : !quake.wire
// CHECK:           quake.return_wire %[[W4]] : !quake.wire
// CHECK:           quake.return_wire %[[W5]] : !quake.wire
// CHECK:           return
// CHECK:         }

// The middle loop does not access quantum data, but it separates the inner loop
// from the grandparent induction variable that bounds it. The outer and inner
// loops must unroll so the q[inner] access lowers to static wires, while the
// middle separator can stay rolled as fixed-wire loops.
__qpu__ std::vector<bool> grandparent_bound_separator() {
  constexpr int N = 4;
  cudaq::qvector q(N);
  // Expected to unroll: `outer` bounds the inner quantum-data loop.
  for (int outer = 0; outer < N; ++outer) {
    // Expected to stay rolled: this loop only repeats fixed-wire operations
    // after the surrounding dependent loops are unrolled.
    for (int middle = 0; middle < 2; ++middle) {
      // Expected to unroll: `inner` indexes quantum data and is bounded by
      // grandparent induction variable `outer`.
      for (int inner = 0; inner < outer; ++inner) {
        x(q[inner]);
      }
    }
  }
  return cudaq::to_bools(mz(q));
}

// The grandparent-dependent separator case should unroll the outer and inner
// loops while preserving the non-blocking middle loop as fixed-wire loops.
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_grandparent_bound_separator._Z27grandparent_bound_separatorv() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-NOT:       {{quake\.alloca|quake\.extract_ref|quake\.subveq|!quake\.ref|!quake\.veq}}
// CHECK:           %[[Q0:.*]] = quake.borrow_wire @wires[0] : !quake.wire
// CHECK:           %[[Q1:.*]] = quake.borrow_wire @wires[1] : !quake.wire
// CHECK:           %[[Q2:.*]] = quake.borrow_wire @wires[2] : !quake.wire
// CHECK:           %[[Q3:.*]] = quake.borrow_wire @wires[3] : !quake.wire
// CHECK-NOT:       {{quake\.alloca|quake\.extract_ref|quake\.subveq|!quake\.ref|!quake\.veq}}
// CHECK:           %[[MID0:.*]]:2 = cc.loop while
// CHECK-SAME:        !quake.wire
// CHECK:             quake.x %{{.*}} : (!quake.wire) -> !quake.wire
// CHECK:           } step {
// CHECK:           %[[MID1:.*]]:3 = cc.loop while
// CHECK-SAME:        !quake.wire, !quake.wire
// CHECK:             quake.x %{{.*}} : (!quake.wire) -> !quake.wire
// CHECK:             quake.x %{{.*}} : (!quake.wire) -> !quake.wire
// CHECK:           } step {
// CHECK:           %[[MID2:.*]]:4 = cc.loop while
// CHECK-SAME:        !quake.wire, !quake.wire, !quake.wire
// CHECK:             quake.x %{{.*}} : (!quake.wire) -> !quake.wire
// CHECK:             quake.x %{{.*}} : (!quake.wire) -> !quake.wire
// CHECK:             quake.x %{{.*}} : (!quake.wire) -> !quake.wire
// CHECK:           } step {
// CHECK-NOT:       cc.loop
// CHECK-NOT:       {{quake\.alloca|quake\.extract_ref|quake\.subveq|!quake\.ref|!quake\.veq}}
// CHECK:           %[[M0:.*]], %[[W0:.*]] = quake.mz %{{.*}} : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[M1:.*]], %[[W1:.*]] = quake.mz %{{.*}} : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[M2:.*]], %[[W2:.*]] = quake.mz %{{.*}} : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           %[[M3:.*]], %[[W3:.*]] = quake.mz %[[Q3]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK-NOT:       cc.loop
// CHECK-NOT:       {{quake\.alloca|quake\.extract_ref|quake\.subveq|!quake\.ref|!quake\.veq}}
// CHECK:           quake.return_wire %[[W0]] : !quake.wire
// CHECK:           quake.return_wire %[[W1]] : !quake.wire
// CHECK:           quake.return_wire %[[W2]] : !quake.wire
// CHECK:           quake.return_wire %[[W3]] : !quake.wire
// CHECK:           return
// CHECK:         }
