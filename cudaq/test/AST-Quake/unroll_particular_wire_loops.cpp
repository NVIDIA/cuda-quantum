/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Starting from C++, mimic the relevant JIT shape: an early classical-only
// mem2reg/loop-normalization cleanup from the standard prep pipeline, the
// target high-level measurement expansion, then the mid-level selective unroll
// and the single full quantum mem2reg that produces wires for assign-wire-indices.
// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --canonicalize --cc-loop-normalize --expand-measurements --cc-loop-unroll=unroll-only-wire-blocking-loops=true --add-dealloc --combine-quantum-alloc --canonicalize --factor-quantum-alloc --memtoreg --add-wireset --assign-wire-indices | FileCheck %s
// clang-format on

#include <cudaq.h>

// Fixed-wire loops can be represented as wire iter-arg loops, so they stay
// rolled.
__qpu__ bool keeps_wire_compatible_loop() {
  cudaq::qubit a, b;
  // Expected to stay rolled: the loop body uses fixed qubits.
  for (int r = 0; r < 5; r++) {
    h(a);
    x<cudaq::ctrl>(a, b);
  }
  return mz(a) ^ mz(b);
}

// clang-format off
// CHECK-LABEL:   quake.wire_set @wires[2147483647] attributes {sym_visibility = "private"}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_keeps_wire_compatible_loop
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK:           %[[A:.*]] = quake.borrow_wire @wires[0] : !quake.wire
// CHECK:           %[[B:.*]] = quake.borrow_wire @wires[1] : !quake.wire
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK:           %[[LOOP:.*]]:3 = cc.loop while
// CHECK-SAME:        !quake.wire
// CHECK:             %[[H:.*]] = quake.h %{{.*}} : (!quake.wire) -> !quake.wire
// CHECK:             %[[X:.*]]:2 = quake.x {{\[}}%[[H]]] %{{.*}} : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
// CHECK:           quake.return_wire
// CHECK:           quake.return_wire
// CHECK:           return
// clang-format on

// Loops that index quantum data by the induction variable block wire
// conversion, so they are unrolled.
__qpu__ bool unrolls_wire_blocking_loop() {
  cudaq::qvector q(3);
  // Expected to unroll: the induction variable indexes quantum data.
  for (int i = 0; i < 3; i++) {
    h(q[i]);
  }
  return mz(q[0]) ^ mz(q[1]) ^ mz(q[2]);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_unrolls_wire_blocking_loop
// CHECK-NOT:       cc.loop
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK:           %[[Q0:.*]] = quake.borrow_wire @wires[0] : !quake.wire
// CHECK:           %[[Q1:.*]] = quake.borrow_wire @wires[1] : !quake.wire
// CHECK:           %[[Q2:.*]] = quake.borrow_wire @wires[2] : !quake.wire
// CHECK-NOT:       cc.loop
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK:           %[[H0:.*]] = quake.h %[[Q0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H1:.*]] = quake.h %[[Q1]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[H2:.*]] = quake.h %[[Q2]] : (!quake.wire) -> !quake.wire
// CHECK:           quake.return_wire
// CHECK:           quake.return_wire
// CHECK:           quake.return_wire
// CHECK-NOT:       cc.loop
// CHECK:           return
// clang-format on

// Loops over measurement data do not access quantum data, so they stay rolled.
__qpu__ bool keeps_measurement_data_loop() {
  cudaq::qvector q(3);
  x(q[0]);
  x(q[2]);

  auto bits = mz(q);
  bool parity = false;
  // Expected to stay rolled: the loop indexes measurement data, not qubits.
  for (int i = 0; i < 3; i++) {
    parity ^= bits[i];
  }
  return parity;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_keeps_measurement_data_loop
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK:           %[[Q0:.*]] = quake.borrow_wire @wires[0] : !quake.wire
// CHECK:           %[[Q1:.*]] = quake.borrow_wire @wires[1] : !quake.wire
// CHECK:           %[[Q2:.*]] = quake.borrow_wire @wires[2] : !quake.wire
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK:           %[[X0:.*]] = quake.x %[[Q0]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[X2:.*]] = quake.x %[[Q2]] : (!quake.wire) -> !quake.wire
// CHECK:           %[[HANDLES:.*]] = cc.alloca !cc.array<!cc.measure_handle x 3>
// CHECK:           %[[M0:.*]], %[[W0:.*]] = quake.mz %[[X0]] name "bits" : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           cc.store %[[M0]], %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[M1:.*]], %[[W1:.*]] = quake.mz %[[Q1]] name "bits" : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           cc.store %[[M1]], %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[M2:.*]], %[[W2:.*]] = quake.mz %[[X2]] name "bits" : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK:           cc.store %[[M2]], %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK-NOT:       quake.alloca
// CHECK-NOT:       quake.unwrap
// CHECK-NOT:       quake.wrap
// CHECK-NOT:       quake.concat
// CHECK-NOT:       quake.extract_ref
// CHECK:           %[[LOOP:.*]]:2 = cc.loop while
// CHECK-SAME:        (i1, i32)
// CHECK:             %[[PTR:.*]] = cc.compute_ptr %[[HANDLES]]{{\[}}%{{.*}}] : (!cc.ptr<!cc.array<!cc.measure_handle x 3>>, i64) -> !cc.ptr<!cc.measure_handle>
// CHECK:             %[[HANDLE:.*]] = cc.load %[[PTR]] : !cc.ptr<!cc.measure_handle>
// CHECK:             quake.discriminate %[[HANDLE]] : (!cc.measure_handle) -> i1
// CHECK:             cc.continue
// CHECK:           quake.return_wire %[[W0]] : !quake.wire
// CHECK:           quake.return_wire %[[W1]] : !quake.wire
// CHECK:           quake.return_wire %[[W2]] : !quake.wire
// CHECK:           return %[[LOOP]]#0 : i1
// clang-format on
