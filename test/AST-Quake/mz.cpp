/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct S {
  void operator()() __qpu__ {
    cudaq::qvector reg(20);
    mz(reg);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S() attributes
// CHECK:           quake.mz
// CHECK:           return
// CHECK:         }
// clang-format on

struct VectorOfStaticVeq {
  std::vector<cudaq::measure_result> operator()() __qpu__ {
    cudaq::qubit q1;
    cudaq::qvector reg1(4);
    cudaq::qvector reg2(2);
    cudaq::qubit q2;
    return mz(q1, reg1, reg2, q2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfStaticVeq()
// CHECK-NOT: cudaq-entrypoint
// CHECK:           quake.mz
// CHECK:           return
// CHECK:         }

struct VectorOfStaticVeq_Bool {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qubit q1;
    cudaq::qvector reg1(4);
    cudaq::qvector reg2(2);
    cudaq::qubit q2;
    auto res = mz(q1, reg1, reg2, q2);
    return cudaq::to_bool_vector(res);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfStaticVeq_Bool()
// CHECK:           quake.mz
// CHECK:           quake.discriminate
// CHECK:           return
// CHECK:         }

struct VectorOfDynamicVeq {
  std::vector<cudaq::measure_result> operator()(unsigned i, unsigned j) __qpu__ {
    cudaq::qubit q1;
    cudaq::qvector reg1(i);
    cudaq::qvector reg2(j);
    cudaq::qubit q2;
    return mz(q1, reg1, reg2, q2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfDynamicVeq(
// CHECK-NOT: cudaq-entrypoint
// CHECK:           quake.mz
// CHECK:           return
// CHECK:         }

struct MxTest {
  void operator()() __qpu__ {
    cudaq::qubit q;
    auto r = mx(q);
    bool b = r;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MxTest() attributes
// CHECK:           quake.mx
// CHECK:           quake.discriminate
// CHECK:           return
// CHECK:         }

struct MyTest {
  void operator()() __qpu__ {
    cudaq::qubit q;
    auto r = my(q);
    bool b = r;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MyTest() attributes
// CHECK:           quake.my
// CHECK:           quake.discriminate
// CHECK:           return
// CHECK:         }
