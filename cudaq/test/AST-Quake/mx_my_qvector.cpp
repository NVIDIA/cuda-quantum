/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct MeasureXRange {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(3);
    auto bits = mx(q);
    return cudaq::to_bools(bits);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MeasureXRange
// CHECK:           %[[Q:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[BITS:.*]] = quake.mx %[[Q]] name "bits" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           quake.discriminate
// CHECK:           return
// clang-format on

struct MeasureYRange {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(3);
    auto bits = my(q);
    return cudaq::to_bools(bits);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MeasureYRange
// CHECK:           %[[Q:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[BITS:.*]] = quake.my %[[Q]] name "bits" : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           quake.discriminate
// CHECK:           return
// clang-format on

struct MeasureViewsAndMixedArgs {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qubit head;
    cudaq::qvector q(4);
    auto front = q.front(2);
    auto back = q.back(1);
    cudaq::qubit tail;
    return cudaq::to_bools(mx(head, front, back, tail));
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MeasureViewsAndMixedArgs
// CHECK:           quake.mx %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!quake.ref, !quake.veq<2>, !quake.veq<1>, !quake.ref) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           return
// clang-format on

struct QECStyleRangeLoop {
  int operator()(int numQubits) __qpu__ {
    cudaq::qvector qubits(numQubits);
    h(qubits[0]);
    auto bits = mx(qubits);

    int result = 0;
    for (int i = 0; i < numQubits; ++i) {
      if (bits[i])
        result += 1;
    }
    return result;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__QECStyleRangeLoop
// CHECK:           %[[Q:.*]] = quake.alloca !quake.veq<?>
// CHECK:           %[[BITS:.*]] = quake.mx %[[Q]] name "bits" : (!quake.veq<?>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           quake.discriminate
// CHECK:           return
// clang-format on
