/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

// Allocate a struq of 4 qubits
struct Grouping1 {
  cudaq::qarray<4> ok;
};

__qpu__ void fremtredende1(Grouping1 &);

__qpu__ void seletoey1() {
  Grouping1 e1;
  fremtredende1(e1);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_seletoey1._Z9seletoey1v() attributes
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.struq<!quake.veq<4>>
// CHECK:           call @_Z13fremtredende1R9Grouping1(%[[VAL_0]]) : (!quake.struq<!quake.veq<4>>) -> ()
// CHECK:           return
// CHECK:         }

// Allocate a struq of 1 qubit.
struct Grouping2 {
  cudaq::qubit cuddly;
};

__qpu__ void fremtredende2(Grouping2 &);

__qpu__ void seletoey2() {
  Grouping2 e2;
  fremtredende2(e2);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_seletoey2._Z9seletoey2v() attributes
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.struq<!quake.ref>
// CHECK:           call @_Z13fremtredende2R9Grouping2(%[[VAL_0]]) : (!quake.struq<!quake.ref>) -> ()
// CHECK:           return
// CHECK:         }

// Allocate a struq with 4 qubits, grouped in a pair of fields.
struct Grouping3 {
  cudaq::qubit cozy;
  cudaq::qarray<3> cute;
};

__qpu__ void fremtredende3(Grouping3 &);

__qpu__ void seletoey3() {
  Grouping3 e3;
  fremtredende3(e3);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_seletoey3._Z9seletoey3v() attributes
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.struq<!quake.ref, !quake.veq<3>>
// CHECK:           call @_Z13fremtredende3R9Grouping3(%[[VAL_0]]) : (!quake.struq<!quake.ref, !quake.veq<3>>) -> ()
// CHECK:           return
// CHECK:         }

// Non-allocating, reference grouping.
struct Grouping4 {
  cudaq::qview<> ambiance;
  cudaq::qview<> warm;
};

__qpu__ void fremtredende4(Grouping4 &);

__qpu__ void seletoey4(cudaq::qvector<> &p1, cudaq::qvector<> &p2) {
  Grouping4 e4{p1, p2};
  fremtredende4(e4);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_seletoey4._Z9seletoey4RN5cudaq7qvectorILm2EEES2_(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.veq<?>) attributes
// CHECK:           %[[VAL_2:.*]] = quake.make_struq %[[VAL_0]], %[[VAL_1]] : (!quake.veq<?>, !quake.veq<?>) -> !quake.struq<!quake.veq<?>, !quake.veq<?>>
// CHECK:           call @_Z13fremtredende4R9Grouping4(%[[VAL_2]]) : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> ()
// CHECK:           return
// CHECK:         }

// Non-allocating, reference grouping. We cannot copy a qvector, but we can
// build a struct of references to them.
struct Grouping5 {
  cudaq::qvector<> &snow;
  cudaq::qvector<> &rain;
  cudaq::qvector<> &coat;
};

__qpu__ void fremtredende5(Grouping5 &);

__qpu__ void seletoey5(cudaq::qvector<> &p1, cudaq::qvector<> &p2,
                       cudaq::qvector<> &p3) {
  Grouping5 e5{p1, p2, p3};
  fremtredende5(e5);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_seletoey5._Z9seletoey5RN5cudaq7qvectorILm2EEES2_S2_(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.veq<?>, %[[VAL_2:.*]]: !quake.veq<?>) attributes
// CHECK:           %[[VAL_3:.*]] = quake.make_struq %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : (!quake.veq<?>, !quake.veq<?>, !quake.veq<?>) -> !quake.struq<!quake.veq<?>, !quake.veq<?>, !quake.veq<?>>
// CHECK:           call @_Z13fremtredende5R9Grouping5(%[[VAL_3]]) : (!quake.struq<!quake.veq<?>, !quake.veq<?>, !quake.veq<?>>) -> ()
// CHECK:           return
// CHECK:         }

