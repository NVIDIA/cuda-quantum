/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct PortableControlFormTest {
  void operator()() __qpu__ {
     cudaq::qvector q(2);

    cx(q[0], q[1]);
    cy(q[0], q[1]);
    cz(q[0], q[1]);

    ch(q[0], q[1]);
    ct(q[0], q[1]);
    cs(q[0], q[1]);

    crx(M_PI_2, q[0], q[1]);
    cry(M_PI_2, q[0], q[1]);
    crz(M_PI_2, q[0], q[1]);
    cr1(M_PI_2, q[0], q[1]);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__PortableControlFormTest() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_90:.*]] = arith.constant 1.5707963267948966 : f64
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_1]]] %[[VAL_2]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.y [%[[VAL_3]]] %[[VAL_4]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.z [%[[VAL_5]]] %[[VAL_6]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_8:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h [%[[VAL_7]]] %[[VAL_8]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_10:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.t [%[[VAL_9]]] %[[VAL_10]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_11:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_12:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.s [%[[VAL_11]]] %[[VAL_12]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_14:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.rx (%[[VAL_90]]) [%[[VAL_14]]] %[[VAL_15]] : (f64, !quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_16:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_17:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.ry (%[[VAL_90]]) [%[[VAL_16]]] %[[VAL_17]] : (f64, !quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_18:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_19:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.rz (%[[VAL_90]]) [%[[VAL_18]]] %[[VAL_19]] : (f64, !quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_20:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_21:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.r1 (%[[VAL_90]]) [%[[VAL_20]]] %[[VAL_21]] : (f64, !quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }
