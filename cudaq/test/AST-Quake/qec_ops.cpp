/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>
#include <vector>

// ---------------------------------------------------------------------------
// `cudaq::detector(h)` — single scalar handle.
// ---------------------------------------------------------------------------

struct DetectorScalar {
  void operator()() __qpu__ {
    cudaq::qubit q;
    auto h = mz(q);
    cudaq::detector(h);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__DetectorScalar()
// CHECK:           %[[VAL_Q:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_M:.*]] = quake.mz %[[VAL_Q]] name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HA:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.store %[[VAL_M]], %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_HL:.*]] = cc.load %[[VAL_HA]] : !cc.ptr<!cc.measure_handle>
// CHECK:           qec.detector %[[VAL_HL]] : !cc.measure_handle
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// `cudaq::detector(h0, h1, h2)` — variadic scalar handles.
// ---------------------------------------------------------------------------

struct DetectorVariadic {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1, q2;
    auto h0 = mz(q0);
    auto h1 = mz(q1);
    auto h2 = mz(q2);
    cudaq::detector(h0, h1, h2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__DetectorVariadic()
// CHECK:           %[[VAL_H0:.*]] = cc.load %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_H1:.*]] = cc.load %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_H2:.*]] = cc.load %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           qec.detector %[[VAL_H0]], %[[VAL_H1]], %[[VAL_H2]] : !cc.measure_handle, !cc.measure_handle, !cc.measure_handle
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// `cudaq::detector(vec)` — single stdvec of handles.
// ---------------------------------------------------------------------------

struct DetectorVector {
  void operator()() __qpu__ {
    cudaq::qvector qs(4);
    auto handles = mz(qs);
    cudaq::detector(handles);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__DetectorVector()
// CHECK:           %[[VAL_QS:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_HS:.*]] = quake.mz %[[VAL_QS]] name "handles" : (!quake.veq<4>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           qec.detector %[[VAL_HS]] : !cc.stdvec<!cc.measure_handle>
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// `cudaq::detector(hs, h)` — mixed scalar + list operands.
// ---------------------------------------------------------------------------

struct DetectorMixed {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qvector qs(2);
    auto h = mz(q);
    auto hs = mz(qs);
    cudaq::detector(hs, h);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__DetectorMixed()
// CHECK:           %[[VAL_M:.*]] = quake.mz %{{.*}} name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HS:.*]] = quake.mz %{{.*}} name "hs" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           qec.detector %[[VAL_HS]], %{{.*}} : !cc.stdvec<!cc.measure_handle>, !cc.measure_handle
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// `cudaq::logical_observable(...)` — variadic default index.
// ---------------------------------------------------------------------------

struct ObservableVariadic {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    auto h0 = mz(q0);
    auto h1 = mz(q1);
    cudaq::logical_observable(h0, h1);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ObservableVariadic()
// CHECK:           %[[VAL_H0:.*]] = cc.load %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           %[[VAL_H1:.*]] = cc.load %{{.*}} : !cc.ptr<!cc.measure_handle>
// CHECK:           qec.observable %[[VAL_H0]], %[[VAL_H1]] : !cc.measure_handle, !cc.measure_handle
// CHECK-NOT:       index
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// `cudaq::logical_observable(vec)` — vector form, default index 0.
// ---------------------------------------------------------------------------

struct ObservableVectorDefault {
  void operator()() __qpu__ {
    cudaq::qvector qs(3);
    auto handles = mz(qs);
    cudaq::logical_observable(handles);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ObservableVectorDefault()
// CHECK:           %[[VAL_HS:.*]] = quake.mz %{{.*}} name "handles"
// CHECK:           qec.observable %[[VAL_HS]] : !cc.stdvec<!cc.measure_handle>
// CHECK-NOT:       index
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// `cudaq::logical_observable(vec, 2)` — vector form, explicit index.
// ---------------------------------------------------------------------------

struct ObservableVectorIndexed {
  void operator()() __qpu__ {
    cudaq::qvector qs(3);
    auto handles = mz(qs);
    cudaq::logical_observable(handles, 2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ObservableVectorIndexed()
// CHECK:           %[[VAL_HS:.*]] = quake.mz %{{.*}} name "handles"
// CHECK:           qec.observable %[[VAL_HS]] index 2 : !cc.stdvec<!cc.measure_handle>
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// `cudaq::logical_observable(vec, constexpr_var)` — index is any C++
// compile-time constant, evaluated via Clang's AST constant evaluator.
// Covers `constexpr` named values, simple expressions like `2 + 1`, and
// templated non-type parameters.
// ---------------------------------------------------------------------------

struct ObservableConstexprIndex {
  void operator()() __qpu__ {
    cudaq::qvector qs(3);
    auto handles = mz(qs);
    constexpr std::size_t kIdx = 5;
    cudaq::logical_observable(handles, kIdx);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ObservableConstexprIndex()
// CHECK:           qec.observable %{{.*}} index 5 : !cc.stdvec<!cc.measure_handle>
// CHECK:           return
// CHECK:         }

struct ObservableExpressionIndex {
  void operator()() __qpu__ {
    cudaq::qvector qs(3);
    auto handles = mz(qs);
    cudaq::logical_observable(handles, 2 + 1);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ObservableExpressionIndex()
// CHECK:           qec.observable %{{.*}} index 3 : !cc.stdvec<!cc.measure_handle>
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// `cudaq::logical_observable(hs, h)` — mixed list + scalar operands via
// the variadic overload (default `observable_index = 0`).
// ---------------------------------------------------------------------------

struct ObservableMixed {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qvector qs(2);
    auto h = mz(q);
    auto hs = mz(qs);
    cudaq::logical_observable(hs, h);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ObservableMixed()
// CHECK:           %[[VAL_M:.*]] = quake.mz %{{.*}} name "h" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_HS:.*]] = quake.mz %{{.*}} name "hs" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           qec.observable %[[VAL_HS]], %{{.*}} : !cc.stdvec<!cc.measure_handle>, !cc.measure_handle
// CHECK-NOT:       index
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// Rvalue measurement handles
// ---------------------------------------------------------------------------

struct DetectorRvalueScalar {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::detector(mz(q));
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__DetectorRvalueScalar()
// CHECK:           %[[VAL_M:.*]] = quake.mz %{{.*}} : (!quake.ref) -> !cc.measure_handle
// CHECK:           qec.detector %{{.*}} : !cc.measure_handle
// CHECK:           return
// CHECK:         }

struct ObservableRvalueVariadic {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    cudaq::logical_observable(mz(q0), mz(q1));
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ObservableRvalueVariadic()
// CHECK:           qec.observable %{{.*}}, %{{.*}} : !cc.measure_handle, !cc.measure_handle
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// `cudaq::detectors(prev, curr)` — paired stdvecs.
// ---------------------------------------------------------------------------

struct PairDetectors {
  void operator()() __qpu__ {
    cudaq::qvector ancA(3), ancB(3);
    auto prev = mz(ancA);
    auto curr = mz(ancB);
    cudaq::detectors(prev, curr);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__PairDetectors()
// CHECK:           %[[VAL_P:.*]] = quake.mz %{{.*}} name "prev" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[VAL_C:.*]] = quake.mz %{{.*}} name "curr" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           qec.pair_detectors %[[VAL_P]], %[[VAL_C]] : <!cc.measure_handle>, <!cc.measure_handle>
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// End-to-end: distance-3 bit-flip repetition code with cross-round detectors
// and a final logical observable.
// ---------------------------------------------------------------------------

struct RepCodeD3 {
  void operator()(int nRounds) __qpu__ {
    cudaq::qvector data(3);
    cudaq::qubit anc0, anc1;
    cudaq::measure_handle prev_s0, prev_s1;

    for (int r = 0; r < nRounds; r++) {
      cx(data[0], anc0);
      cx(data[1], anc0);
      cx(data[1], anc1);
      cx(data[2], anc1);

      auto s0 = mz(anc0);
      auto s1 = mz(anc1);
      reset(anc0);
      reset(anc1);

      if (r > 0) {
        cudaq::detector(prev_s0, s0);
        cudaq::detector(prev_s1, s1);
      }
      prev_s0 = s0;
      prev_s1 = s1;
    }
    auto readout = mz(data);
    cudaq::logical_observable(readout);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__RepCodeD3(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca i32
// CHECK:           cc.store %[[ARG0]], %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:           %[[ALLOCA_1:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[ALLOCA_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[ALLOCA_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[ALLOCA_4:.*]] = cc.alloca !cc.measure_handle
// CHECK:           %[[ALLOCA_5:.*]] = cc.alloca !cc.measure_handle
// CHECK:           cc.scope {
// CHECK:             %[[ALLOCA_6:.*]] = cc.alloca i32
// CHECK:             cc.store %[[CONSTANT_1]], %[[ALLOCA_6]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[LOAD_0:.*]] = cc.load %[[ALLOCA_6]] : !cc.ptr<i32>
// CHECK:               %[[LOAD_1:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_0]], %[[LOAD_1]] : i32
// CHECK:               cc.condition %[[CMPI_0]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_1]][0] : (!quake.veq<3>) -> !quake.ref
// CHECK:                 quake.x {{\[}}%[[EXTRACT_REF_0]]] %[[ALLOCA_2]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                 %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ALLOCA_1]][1] : (!quake.veq<3>) -> !quake.ref
// CHECK:                 quake.x {{\[}}%[[EXTRACT_REF_1]]] %[[ALLOCA_2]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                 %[[EXTRACT_REF_2:.*]] = quake.extract_ref %[[ALLOCA_1]][1] : (!quake.veq<3>) -> !quake.ref
// CHECK:                 quake.x {{\[}}%[[EXTRACT_REF_2]]] %[[ALLOCA_3]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                 %[[EXTRACT_REF_3:.*]] = quake.extract_ref %[[ALLOCA_1]][2] : (!quake.veq<3>) -> !quake.ref
// CHECK:                 quake.x {{\[}}%[[EXTRACT_REF_3]]] %[[ALLOCA_3]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                 %[[MZ_0:.*]] = quake.mz %[[ALLOCA_2]] name "s0" : (!quake.ref) -> !cc.measure_handle
// CHECK:                 %[[ALLOCA_7:.*]] = cc.alloca !cc.measure_handle
// CHECK:                 cc.store %[[MZ_0]], %[[ALLOCA_7]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[MZ_1:.*]] = quake.mz %[[ALLOCA_3]] name "s1" : (!quake.ref) -> !cc.measure_handle
// CHECK:                 %[[ALLOCA_8:.*]] = cc.alloca !cc.measure_handle
// CHECK:                 cc.store %[[MZ_1]], %[[ALLOCA_8]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 quake.reset %[[ALLOCA_2]] : (!quake.ref) -> ()
// CHECK:                 quake.reset %[[ALLOCA_3]] : (!quake.ref) -> ()
// CHECK:                 %[[LOAD_2:.*]] = cc.load %[[ALLOCA_6]] : !cc.ptr<i32>
// CHECK:                 %[[CMPI_1:.*]] = arith.cmpi sgt, %[[LOAD_2]], %[[CONSTANT_1]] : i32
// CHECK:                 cc.if(%[[CMPI_1]]) {
// CHECK:                   %[[LOAD_3:.*]] = cc.load %[[ALLOCA_4]] : !cc.ptr<!cc.measure_handle>
// CHECK:                   %[[LOAD_4:.*]] = cc.load %[[ALLOCA_7]] : !cc.ptr<!cc.measure_handle>
// CHECK:                   qec.detector %[[LOAD_3]], %[[LOAD_4]] : !cc.measure_handle, !cc.measure_handle
// CHECK:                   %[[LOAD_5:.*]] = cc.load %[[ALLOCA_5]] : !cc.ptr<!cc.measure_handle>
// CHECK:                   %[[LOAD_6:.*]] = cc.load %[[ALLOCA_8]] : !cc.ptr<!cc.measure_handle>
// CHECK:                   qec.detector %[[LOAD_5]], %[[LOAD_6]] : !cc.measure_handle, !cc.measure_handle
// CHECK:                 }
// CHECK:                 %[[LOAD_7:.*]] = cc.load %[[ALLOCA_7]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 cc.store %[[LOAD_7]], %[[ALLOCA_4]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 %[[LOAD_8:.*]] = cc.load %[[ALLOCA_8]] : !cc.ptr<!cc.measure_handle>
// CHECK:                 cc.store %[[LOAD_8]], %[[ALLOCA_5]] : !cc.ptr<!cc.measure_handle>
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[LOAD_9:.*]] = cc.load %[[ALLOCA_6]] : !cc.ptr<i32>
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[LOAD_9]], %[[CONSTANT_0]] : i32
// CHECK:               cc.store %[[ADDI_0]], %[[ALLOCA_6]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[MZ_2:.*]] = quake.mz %[[ALLOCA_1]] name "readout" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           qec.observable %[[MZ_2]] : !cc.stdvec<!cc.measure_handle>
// CHECK:           return
// CHECK:         }

// ---------------------------------------------------------------------------
// A user-defined `cudaq::qec::detector` must NOT be hijacked by the bridge:
// the QEC handlers gate on the IMMEDIATE enclosing namespace being `cudaq`
// (`isInDirectNamespace`), so sub-namespace symbols with the same name
// dispatch through the regular function-call path. The bridge would
// otherwise silently shadow QEC-side helpers a user might add.
// ---------------------------------------------------------------------------

namespace cudaq::qec {
inline void detector(cudaq::measure_handle h) {
  (void)h;
}
} // namespace cudaq::qec

struct NamespacedDetectorIsNotHijacked {
  void operator()() __qpu__ {
    cudaq::qubit q;
    auto h = mz(q);
    cudaq::qec::detector(h);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__NamespacedDetectorIsNotHijacked()
// CHECK:           call @_ZN5cudaq3qec8detectorENS_14measure_handleE(%{{.*}})
// CHECK-NOT:       qec.detector
// CHECK:           return
// CHECK:         }
