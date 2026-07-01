/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Starting from C++, expand whole-register controls before the wire-assignment
// pipeline so value-semantic quantum code reaches indexed wires.
// RUN: cudaq-quake %s | cudaq-opt --expand-control-veqs --memtoreg=quantum=0 --canonicalize --cc-loop-normalize --expand-measurements --cc-loop-unroll=unroll-only-wire-blocking-loops=true --add-dealloc --combine-quantum-alloc --canonicalize --factor-quantum-alloc --memtoreg --add-wireset --assign-wire-indices | FileCheck %s
// clang-format on

#include <cudaq.h>

struct veq_control {
  void operator()() __qpu__ {
    cudaq::qvector ctrls(3);
    cudaq::qubit target;
    x<cudaq::ctrl>(ctrls, target); // Multi-controlled X, whole register as controls.
    mz(target);
  }
};

// The whole-register control should expand to one multi-control wire op with
// three static control wires and no `quake.alloca` or `quake.veq` in the kernel.
// CHECK-LABEL:   quake.wire_set @wires[2147483647] attributes {sym_visibility = "private"}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__veq_control() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           %[[BORROW_WIRE_0:.*]] = quake.borrow_wire @wires[0] : !quake.wire
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           %[[BORROW_WIRE_1:.*]] = quake.borrow_wire @wires[1] : !quake.wire
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           %[[BORROW_WIRE_2:.*]] = quake.borrow_wire @wires[2] : !quake.wire
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           %[[BORROW_WIRE_3:.*]] = quake.borrow_wire @wires[3] : !quake.wire
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           %[[X_0:.*]]:4 = quake.x {{\[}}%[[BORROW_WIRE_0]], %[[BORROW_WIRE_1]], %[[BORROW_WIRE_2]]] %[[BORROW_WIRE_3]] : (!quake.wire, !quake.wire, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire, !quake.wire, !quake.wire)
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           %[[VAL_0:.*]], %[[MZ_0:.*]] = quake.mz %[[X_0]]#3 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           quake.return_wire %[[X_0]]#0 : !quake.wire
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           quake.return_wire %[[X_0]]#1 : !quake.wire
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           quake.return_wire %[[X_0]]#2 : !quake.wire
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           quake.return_wire %[[MZ_0]] : !quake.wire
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:           return
// CHECK-NOT:       {{quake\.alloca|!quake\.veq|!quake\.ref}}
// CHECK:         }
