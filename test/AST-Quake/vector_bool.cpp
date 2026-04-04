/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

// Simple test using a std::vector<bool> operator.

#include <cudaq.h>

struct t1 {
  bool operator()(std::vector<double> d) __qpu__ {
    cudaq::qvector q(2);
    auto vec = mz(q);
    return vec[0];
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__t1(
// CHECK:           quake.mz
// CHECK:           quake.get_measure
// CHECK:           quake.discriminate
// CHECK:           return
// CHECK:         }
// CHECK-NOT:     func.func private @_ZNKSt14_Bit_referencecvbEv() -> i1
// clang-format on

struct VectorBoolReturn {
   std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(4);
    auto res = mz(q);
    return cudaq::to_bool_vector(res);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorBoolReturn()
// CHECK:           quake.mz
// CHECK:           quake.discriminate
// CHECK:           return
// CHECK:         }
// clang-format on

struct VectorMeasureResult {
   std::vector<cudaq::measure_result> operator()() __qpu__ {
    cudaq::qvector q(4);
    return mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorMeasureResult()
// CHECK-NOT:     cudaq-entrypoint
// CHECK:           quake.mz
// CHECK-NOT:       quake.discriminate
// CHECK:           return
// CHECK:         }
// clang-format on
