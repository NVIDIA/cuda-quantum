/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>
#include <vector>

struct VectorHandleAssign {
  void operator()() __qpu__ {
    cudaq::qvector qv(2);
    auto m = mz(qv);
    auto m_new = mz(qv);
    m = m_new;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorHandleAssign() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[QV:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[M:.*]] = quake.mz %[[QV]] name "m" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[M_ADDR:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %[[M]], %[[M_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           %[[M_NEW:.*]] = quake.mz %[[QV]] name "m_new" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[M_NEW_ADDR:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %[[M_NEW]], %[[M_NEW_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           %[[M_NEW_VAL:.*]] = cc.load %[[M_NEW_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           cc.store %[[M_NEW_VAL]], %[[M_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           return
// CHECK:         }


struct LoopCarriedReassign {
  void operator()(int nRounds) __qpu__ {
    cudaq::qvector qvec(3);
    auto m = mz(qvec);
    for (int round = 0; round < nRounds; round++) {
      auto m_new = mz(qvec);
      m = m_new;
    }
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__LoopCarriedReassign(
// CHECK:           %[[QVEC:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[M_OUTER:.*]] = quake.mz %[[QVEC]] name "m" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[M_ADDR:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %[[M_OUTER]], %[[M_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           cc.scope {
// CHECK:             cc.loop while {
// CHECK:               cc.condition
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[M_NEW:.*]] = quake.mz %[[QVEC]] name "m_new" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:                 %[[M_NEW_ADDR:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:                 cc.store %[[M_NEW]], %[[M_NEW_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 %[[M_NEW_VAL:.*]] = cc.load %[[M_NEW_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:                 cc.store %[[M_NEW_VAL]], %[[M_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:               }


struct ConditionalReassign {
  bool operator()(bool cond) __qpu__ {
    cudaq::qvector qv(2);
    auto m = mz(qv);
    if (cond) {
      cudaq::qvector qv2(2);
      auto m_new = mz(qv2);
      m = m_new;
    }
    return m[0];
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ConditionalReassign(
// CHECK:           %[[QV:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[M:.*]] = quake.mz %[[QV]] name "m" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[M_ADDR:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %[[M]], %[[M_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           cc.if(%{{.*}}) {
// CHECK:             %[[QV2:.*]] = quake.alloca !quake.veq<2>
// CHECK:             %[[M_NEW:.*]] = quake.mz %[[QV2]] name "m_new" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:             %[[M_NEW_ADDR:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:             cc.store %[[M_NEW]], %[[M_NEW_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:             %[[M_NEW_VAL:.*]] = cc.load %[[M_NEW_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:             cc.store %[[M_NEW_VAL]], %[[M_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           }
// CHECK:           %[[POST:.*]] = cc.load %[[M_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           %{{.*}} = cc.stdvec_data %[[POST]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.ptr<!cc.array<!cc.measure_handle x ?>>
// CHECK:           return %{{.*}} : i1


struct ChainedAssign {
  void operator()() __qpu__ {
    cudaq::qvector qv(2);
    auto a = mz(qv);
    auto b = mz(qv);
    auto c = mz(qv);
    a = b = c;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ChainedAssign() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[A:.*]] = quake.mz %{{.*}} name "a" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[A_ADDR:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %[[A]], %[[A_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           %[[B:.*]] = quake.mz %{{.*}} name "b" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[B_ADDR:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %[[B]], %[[B_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           %[[C:.*]] = quake.mz %{{.*}} name "c" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[C_ADDR:.*]] = cc.alloca !cc.stdvec<!cc.measure_handle>
// CHECK:           cc.store %[[C]], %[[C_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           %[[C_VAL:.*]] = cc.load %[[C_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           cc.store %[[C_VAL]], %[[B_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           %[[B_VAL:.*]] = cc.load %[[B_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           cc.store %[[B_VAL]], %[[A_ADDR]] : !cc.ptr<!cc.stdvec<!cc.measure_handle>>
// CHECK:           return
