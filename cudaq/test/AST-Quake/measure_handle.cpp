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

struct ScalarHandle {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h = mz(q);
    (void)h;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ScalarHandle()
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

struct ScalarBool {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    auto h = mz(q);
    return h;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ScalarBool() -> i1
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_HL:.*]] = cc.load %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %[[VAL_HL]] : (!cc.measure_handle) -> i1
// CHECK:           return %[[VAL_B]] : i1
// CHECK:         }

struct CopyConstructedBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h = mz(q);
    cudaq::measure_handle h2 = h;
    bool b = h2;
    (void)b;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CopyConstructedBound()
// CHECK:           quake.mz %{{.*}} name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           return
// CHECK:         }

struct CopyAssignedBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h = mz(q);
    cudaq::measure_handle h2;
    h2 = h;
    bool b = h2;
    (void)b;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CopyAssignedBound()
// CHECK:           quake.mz %{{.*}} name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           return
// CHECK:         }

struct ChainedAssignedBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h2;
    cudaq::measure_handle h3;
    h3 = h2 = mz(q);
    bool b = h3;
    (void)b;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ChainedAssignedBound()
// CHECK:           quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           return
// CHECK:         }

struct ArrayElementBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle hs[2];
    hs[0] = mz(q);
    bool b = hs[0];
    (void)b;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ArrayElementBound()
// CHECK:           quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           cc.cast %{{.*}} : (!cc.ptr<!cc.array<!cc.measure_handle x 2>>) -> !cc.ptr<!cc.measure_handle>
// CHECK:           quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           return
// CHECK:         }

struct Holder {
  cudaq::measure_handle h;
};

struct AggregateMemberBound {
  void operator()() __qpu__ {
    cudaq::qubit q;
    Holder holder;
    holder.h = mz(q);
    bool b = holder.h;
    (void)b;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__AggregateMemberBound()
// CHECK:           quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           return
// CHECK:         }

struct DirectIf {
  void operator()() __qpu__ {
    cudaq::qubit q;
    if (mz(q))
      x(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__DirectIf()
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_HL:.*]] = cc.load %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %[[VAL_HL]] : (!cc.measure_handle) -> i1
// CHECK:           cc.if(%[[VAL_B]]) {
// CHECK:             quake.x %[[VAL_Q]]
// CHECK:           }
// CHECK:           return
// CHECK:         }

struct ConditionalStoreAfterBind {
  bool operator()(bool cond) __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle h = mz(q);
    if (cond) {
      cudaq::qubit q2;
      h = mz(q2);
    }
    bool b = h;
    return b;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ConditionalStoreAfterBind(
// CHECK:           quake.mz %{{.*}} name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           cc.if(%{{.*}}) {
// CHECK:             quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           }
// CHECK:           quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           return %{{.*}} : i1
// CHECK:         }

struct WhileCond {
  void operator()() __qpu__ {
    cudaq::qubit q;
    while (mz(q))
      x(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__WhileCond()
// CHECK:           cc.loop while {
// CHECK:             %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[VAL_B:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:             cc.condition %[[VAL_B]]
// CHECK:           } do {
// CHECK:             quake.x

struct ForCond {
  void operator()() __qpu__ {
    cudaq::qubit q;
    for (int i = 0; mz(q); ++i)
      x(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ForCond()
// CHECK:           cc.loop while {
// CHECK:             %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[VAL_B:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:             cc.condition %[[VAL_B]]

struct RegisterHandle {
  void operator()() __qpu__ {
    cudaq::qvector v(4);
    auto hs = mz(v);
    (void)hs;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__RegisterHandle()
// CHECK:           %[[VAL_V:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %{{.*}} = quake.mz %[[VAL_V]] name "hs" : (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

struct RegisterBools {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector v(4);
    return cudaq::to_bools(mz(v));
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__RegisterBools() -> !cc.stdvec<i1>
// CHECK:           %[[VAL_V:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_V]] : (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[VAL_BV:.*]] = quake.discriminate %[[VAL_M]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.stdvec<i1>
// CHECK:           %{{.*}} = cc.stdvec_data %[[VAL_BV]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %{{.*}} = cc.stdvec_size %[[VAL_BV]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %{{.*}} = call @__nvqpp_vectorCopyCtor(
// CHECK:           return %{{.*}} : !cc.stdvec<i1>
// CHECK:         }

void sink(std::int64_t);

struct ToIntegerExplicit {
  void operator()() __qpu__ {
    cudaq::qvector q(8);
    sink(cudaq::to_integer(cudaq::to_bools(mz(q))));
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ToIntegerExplicit
// CHECK:           %[[BOOLS:.*]] = quake.discriminate %{{.*}} : (!cc.stdvec<!cc.measure_handle>) -> !cc.stdvec<i1>
// CHECK-NOT:       quake.discriminate
// CHECK:           %{{.*}} = call @__nvqpp_cudaqConvertToInteger(%[[BOOLS]]) : (!cc.stdvec<i1>) -> i64

struct MxMyHandles {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::measure_handle hx = mx(q);
    cudaq::measure_handle hy = my(q);
    (void)hx;
    (void)hy;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MxMyHandles()
// CHECK:           quake.mx %{{.*}} name "hx" : (!quake.ref) -> !cc.measure_handle
// CHECK:           quake.my %{{.*}} name "hy" : (!quake.ref) -> !cc.measure_handle
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

__qpu__ bool consume_handle_only(cudaq::measure_handle h) { return h; }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_consume_handle_only.
// CHECK-SAME:      attributes {"cudaq-kernel"
// CHECK:           return

// CHECK-NOT:     func.func {{.*}}@_Z{{[0-9]+}}consume_handle_only{{.*}}(

__qpu__ bool consume_handle(cudaq::qubit &q, const cudaq::measure_handle &h) {
  return h;
}

struct CrossFunctionCaller {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    auto h = mz(q);
    return consume_handle(q, h);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_consume_handle.
// CHECK-SAME:      (%{{.*}}: !quake.ref{{.*}}, %[[ARG_H:.*]]: !cc.ptr<!cc.measure_handle>{{.*}}) -> i1
// CHECK:           %[[VAL_HL:.*]] = cc.load %[[ARG_H]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %[[VAL_HL]] : (!cc.measure_handle) -> i1
// CHECK:           return %[[VAL_B]] : i1
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CrossFunctionCaller() -> i1
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_R:.*]] = call @__nvqpp__mlirgen__function_consume_handle.{{.*}}(%[[VAL_Q]], %[[VAL_HA]]) : (!quake.ref, !cc.ptr<!cc.measure_handle>) -> i1
// CHECK:           return %[[VAL_R]] : i1
// CHECK:         }

__qpu__ std::vector<cudaq::measure_handle> single_round(cudaq::qview<> qv) {
  return mz(qv);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_single_round.
// CHECK-SAME:      -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[C8:.*]] = arith.constant 8 : i64
// CHECK:           %[[M:.*]] = quake.mz %{{.*}} : (!quake.veq<?>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[D:.*]] = cc.stdvec_data %[[M]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.ptr<i8>
// CHECK:           %[[S:.*]] = cc.stdvec_size %[[M]] : (!cc.stdvec<!cc.measure_handle>) -> i64
// CHECK:           %[[H:.*]] = call @__nvqpp_vectorCopyCtor(%[[D]], %[[S]], %[[C8]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[V:.*]] = cc.stdvec_init %[[H]], %[[S]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           return %[[V]] : !cc.stdvec<!cc.measure_handle>

__qpu__ std::vector<cudaq::measure_handle>
stab_round(cudaq::qview<> ancz, cudaq::qview<> ancx) {
  return mz(ancz, ancx);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_stab_round.
// CHECK-SAME:      -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[C8:.*]] = arith.constant 8 : i64
// CHECK:           %[[M:.*]] = quake.mz %{{.*}}, %{{.*}} : (!quake.veq<?>, !quake.veq<?>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %{{.*}} = call @__nvqpp_vectorCopyCtor(%{{.*}}, %{{.*}}, %[[C8]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           return %{{.*}} : !cc.stdvec<!cc.measure_handle>

struct HandleEquality {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit r;
    auto h1 = mz(q);
    auto h2 = mz(r);
    return h1 == h2;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleEquality() -> i1
// CHECK:           %[[VAL_M1:.*]] = quake.mz %{{.*}} name "h1" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA1:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M1]], %[[VAL_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_M2:.*]] = quake.mz %{{.*}} name "h2" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA2:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M2]], %[[VAL_HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_HL1:.*]] = cc.load %[[VAL_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B1:.*]] = quake.discriminate %[[VAL_HL1]] : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_HL2:.*]] = cc.load %[[VAL_HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B2:.*]] = quake.discriminate %[[VAL_HL2]] : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_EQ:.*]] = arith.cmpi eq, %[[VAL_B1]], %[[VAL_B2]] : i1
// CHECK:           return %[[VAL_EQ]] : i1
// CHECK:         }

struct HandleNotEqual {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit r;
    auto h1 = mz(q);
    auto h2 = mz(r);
    return h1 != h2;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleNotEqual() -> i1
// CHECK:           %[[VAL_M1:.*]] = quake.mz %{{.*}} name "h1" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_M2:.*]] = quake.mz %{{.*}} name "h2" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_B1:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_B2:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_NE:.*]] = arith.cmpi ne, %[[VAL_B1]], %[[VAL_B2]] : i1
// CHECK:           return %[[VAL_NE]] : i1

struct LogicalNot {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    return !mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__LogicalNot() -> i1
// CHECK:           %[[VAL_FALSE:.*]] = arith.constant false
// CHECK:           %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_NOT:.*]] = arith.cmpi eq, %[[VAL_B]], %[[VAL_FALSE]] : i1
// CHECK:           return %[[VAL_NOT]] : i1

struct HandleAnd {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit r;
    return mz(q) && mz(r);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleAnd() -> i1
// CHECK:           %[[VAL_FALSE:.*]] = arith.constant false
// CHECK:           %[[VAL_M1:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA1:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M1]], %[[VAL_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_HL1:.*]] = cc.load %[[VAL_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_B1:.*]] = quake.discriminate %[[VAL_HL1]] : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_NOT:.*]] = arith.cmpi eq, %[[VAL_B1]], %[[VAL_FALSE]] : i1
// CHECK:           %[[VAL_RES:.*]] = cc.if(%[[VAL_NOT]]) -> i1 {
// CHECK:             cc.continue %[[VAL_FALSE]] : i1
// CHECK:           } else {
// CHECK:             %[[VAL_M2:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[VAL_HA2:.*]] = cc.alloca !cc.measure_handle
// CHECK:             cc.store %[[VAL_M2]], %[[VAL_HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[VAL_HL2:.*]] = cc.load %[[VAL_HA2]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[VAL_B2:.*]] = quake.discriminate %[[VAL_HL2]] : (!cc.measure_handle) -> i1
// CHECK:             cc.continue %[[VAL_B2]] : i1
// CHECK:           }
// CHECK:           return %[[VAL_RES]] : i1
// CHECK:         }

struct HandleOr {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit r;
    return mz(q) || mz(r);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleOr() -> i1
// CHECK:           %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_B1:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_R:.*]] = cc.if(%[[VAL_B1]]) -> i1 {
// CHECK:             cc.continue %[[VAL_B1]] : i1
// CHECK:           } else {
// CHECK:             %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:             %[[VAL_B2:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:             cc.continue %[[VAL_B2]] : i1
// CHECK:           }
// CHECK:           return %[[VAL_R]] : i1

struct HandleNamedAndRhs {
  bool operator()() __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    auto result0 = mz(q0);
    auto result1 = mz(q1);
    return result0 && result1;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleNamedAndRhs() -> i1
// CHECK:           %[[NA_M0:.*]] = quake.mz %{{.*}} name "result0" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[NA_HA0:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[NA_M0]], %[[NA_HA0]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[NA_M1:.*]] = quake.mz %{{.*}} name "result1" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[NA_HA1:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[NA_M1]], %[[NA_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[NA_HL0:.*]] = cc.load %[[NA_HA0]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[NA_D0:.*]] = quake.discriminate %[[NA_HL0]] : (!cc.measure_handle) -> i1
// CHECK:           %{{.*}} = cc.if(%{{.*}}) -> i1 {
// CHECK:             cc.continue %{{.*}} : i1
// CHECK:           } else {
// CHECK:             %[[NA_HL1:.*]] = cc.load %[[NA_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[NA_D1:.*]] = quake.discriminate %[[NA_HL1]] : (!cc.measure_handle) -> i1
// CHECK:             cc.continue %[[NA_D1]] : i1
// CHECK:           }

struct HandleNamedOrRhs {
  bool operator()() __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    auto result0 = mz(q0);
    auto result1 = mz(q1);
    return result0 || result1;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleNamedOrRhs() -> i1
// CHECK:           %[[NO_M0:.*]] = quake.mz %{{.*}} name "result0" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[NO_HA0:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[NO_M0]], %[[NO_HA0]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[NO_M1:.*]] = quake.mz %{{.*}} name "result1" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[NO_HA1:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[NO_M1]], %[[NO_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[NO_HL0:.*]] = cc.load %[[NO_HA0]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[NO_D0:.*]] = quake.discriminate %[[NO_HL0]] : (!cc.measure_handle) -> i1
// CHECK:           %{{.*}} = cc.if(%{{.*}}) -> i1 {
// CHECK:             cc.continue %{{.*}} : i1
// CHECK:           } else {
// CHECK:             %[[NO_HL1:.*]] = cc.load %[[NO_HA1]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[NO_D1:.*]] = quake.discriminate %[[NO_HL1]] : (!cc.measure_handle) -> i1
// CHECK:             cc.continue %[[NO_D1]] : i1
// CHECK:           }

struct HandleNamedDiscrimInsideIf {
  bool operator()(bool cond) __qpu__ {
    cudaq::qubit q;
    auto result0 = mz(q);
    bool b = false;
    if (cond)
      b = result0;
    return b;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__HandleNamedDiscrimInsideIf(
// CHECK:           %[[NI_M0:.*]] = quake.mz %{{.*}} name "result0" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[NI_HA0:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[NI_M0]], %[[NI_HA0]] : !cc.ptr<!cc.measure_handle>
// CHECK:           cc.if(%{{.*}}) {
// CHECK:             %[[NI_HL0:.*]] = cc.load %[[NI_HA0]] : !cc.ptr<!cc.measure_handle>
// CHECK:             %[[NI_D0:.*]] = quake.discriminate %[[NI_HL0]] : (!cc.measure_handle) -> i1
// CHECK:             cc.store %[[NI_D0]], %{{.*}} : !cc.ptr<i1>
// CHECK:           }

struct BoolInit {
  void operator()() __qpu__ {
    cudaq::qubit q;
    bool b = mz(q);
    if (b)
      x(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__BoolInit()
// CHECK:           %{{.*}} = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_B:.*]] = quake.discriminate %{{.*}} : (!cc.measure_handle) -> i1
// CHECK:           %[[VAL_S:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_B]], %[[VAL_S]] : !cc.ptr<i1>

struct CallableParamReturningHandleVec {
  void operator()(
      cudaq::qkernel<std::vector<cudaq::measure_handle>(std::size_t)> cb)
      __qpu__ {
    auto syn = cb(2);
    (void)syn;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CallableParamReturningHandleVec(
// CHECK-SAME:      %{{.*}}: !cc.indirect_callable<(i64) -> !cc.stdvec<!cc.measure_handle>>
// CHECK-SAME:      attributes {"cudaq-entrypoint", "cudaq-kernel"

struct CallableParamTakingHandle {
  void operator()(cudaq::qkernel<void(cudaq::measure_handle)> cb) __qpu__ {
    cudaq::qubit q;
    auto h = mz(q);
    cb(h);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CallableParamTakingHandle(
// CHECK-SAME:      %{{.*}}: !cc.indirect_callable<(!cc.measure_handle) -> ()>
// CHECK-SAME:      attributes {"cudaq-entrypoint", "cudaq-kernel"
