/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Tests documenting the difference between broadcast and control semantics
// for single-qubit gates and swap, with and without the <cudaq::ctrl> modifier.

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

// Broadcast: x applied to three individually addressed qubits.
// All three are targets; no controls are generated.
struct broadcast_individual {
  void operator()() __qpu__ {
    cudaq::qvector q(4);
    x(q[0], q[1], q[2]);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__broadcast_individual
// CHECK:           %[[Q:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[R0:.*]] = quake.extract_ref %[[Q]][0] : (!quake.veq<4>) -> !quake.ref
// CHECK:           %[[R1:.*]] = quake.extract_ref %[[Q]][1] : (!quake.veq<4>) -> !quake.ref
// CHECK:           %[[R2:.*]] = quake.extract_ref %[[Q]][2] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.x %[[R0]] : (!quake.ref) -> ()
// CHECK:           quake.x %[[R1]] : (!quake.ref) -> ()
// CHECK:           quake.x %[[R2]] : (!quake.ref) -> ()
// CHECK:           return

// Broadcast: x applied to every qubit in a register via an invariant loop.
struct broadcast_register {
  void operator()() __qpu__ {
    cudaq::qvector q(4);
    x(q);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__broadcast_register
// CHECK:           quake.alloca !quake.veq<4>
// CHECK:           cc.loop while
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()

// Broadcast: even with <cudaq::ctrl>, passing a single register still
// broadcasts (the ctrl modifier is ignored for a single-veq operand).
struct ctrl_broadcast_register {
  void operator()() __qpu__ {
    cudaq::qvector q(4);
    x<cudaq::ctrl>(q);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__ctrl_broadcast_register
// CHECK:           quake.alloca !quake.veq<4>
// CHECK:           cc.loop while
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.x [%

// Control: <cudaq::ctrl> with multiple refs — first N-1 are controls, last
// is the target.  Here q[0] and q[1] are controls; q[2] is the target.
struct ctrl_individual {
  void operator()() __qpu__ {
    cudaq::qvector q(4);
    x<cudaq::ctrl>(q[0], q[1], q[2]);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__ctrl_individual
// CHECK:           quake.alloca !quake.veq<4>
// CHECK:           quake.x [%{{.*}}, %{{.*}}] %{{.*}} : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:           return

// Implicit control: passing a qvector followed by a qubit selects the
// two-argument overload whose default modifier is ctrl, so no explicit
// <cudaq::ctrl> is required.
struct veq_implicit_ctrl {
  void operator()() __qpu__ {
    cudaq::qvector ctrls(4);
    cudaq::qubit target;
    x(ctrls, target);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__veq_implicit_ctrl
// CHECK:           %[[CTRLS:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[TGT:.*]] = quake.alloca !quake.ref
// CHECK:           quake.x [%[[CTRLS]]] %[[TGT]] : (!quake.veq<4>, !quake.ref) -> ()
// CHECK:           return

// Simple swap: two qubits, no controls.
struct simple_swap {
  void operator()() __qpu__ {
    cudaq::qvector q(4);
    swap(q[0], q[1]);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__simple_swap
// CHECK:           %[[Q:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[R0:.*]] = quake.extract_ref %[[Q]][0] : (!quake.veq<4>) -> !quake.ref
// CHECK:           %[[R1:.*]] = quake.extract_ref %[[Q]][1] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.swap %[[R0]], %[[R1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return

// Controlled swap: <cudaq::ctrl> with three individual qubits — the first
// qubit becomes the control, the remaining two are swapped.
struct ctrl_swap_individual {
  void operator()() __qpu__ {
    cudaq::qvector q(4);
    swap<cudaq::ctrl>(q[0], q[1], q[2]);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__ctrl_swap_individual
// CHECK:           %[[Q:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[R0:.*]] = quake.extract_ref %[[Q]][0] : (!quake.veq<4>) -> !quake.ref
// CHECK:           %[[R1:.*]] = quake.extract_ref %[[Q]][1] : (!quake.veq<4>) -> !quake.ref
// CHECK:           %[[R2:.*]] = quake.extract_ref %[[Q]][2] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.swap [%[[R0]]] %[[R1]], %[[R2]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:           return

// Controlled swap: a register of controls with two individual qubit targets.
// The dedicated three-argument overload swap(QuantumRegister, qubit, qubit)
// always treats the first argument as controls — no <cudaq::ctrl> needed.
struct ctrl_swap_veq {
  void operator()() __qpu__ {
    cudaq::qvector ctrl_reg(4);
    cudaq::qubit src, tgt;
    swap(ctrl_reg, src, tgt);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__ctrl_swap_veq
// CHECK:           %[[CTRLS:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[SRC:.*]] = quake.alloca !quake.ref
// CHECK:           %[[TGT:.*]] = quake.alloca !quake.ref
// CHECK:           quake.swap [%[[CTRLS]]] %[[SRC]], %[[TGT]] : (!quake.veq<4>, !quake.ref, !quake.ref) -> ()
// CHECK:           return
