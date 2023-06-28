/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>
#include <vector>

namespace ns {
template <typename T>
struct test {
  using Type = T;
};
} // namespace ns

// CHECK-LABEL: func.func @__nvqpp__mlirgen__name1
// CHECK-SAME: (%{{.*}}: !cc.stdvec<f64>{{.*}}) attributes {
struct name1 {
  void operator()(std::vector<ns::test<double>::Type> variable) __qpu__ {}
};
