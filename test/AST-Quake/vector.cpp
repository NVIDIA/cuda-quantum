/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test for std::vector support

// RUN: cudaq-quake %cpp_std %s  | FileCheck %s

#include <cudaq.h>
#include <cudaq/algorithm.h>

// Define a quantum kernel
struct simple_double_rotation {
  // CHECK-LABEL: func.func @__nvqpp__mlirgen__simple_double_rotation
  // CHECK-SAME: ([[arg:.*]]: !cc.stdvec<f64>{{.*}}) attributes
  auto operator()(std::vector<double> theta) __qpu__ {
    auto size = theta.size();
    bool empty = theta.empty();
    cudaq::qvector q(1);
    int test = q.size();
    rx(theta[0], q[0]);
    mz(q);
  }
};

struct simple_float_rotation {
  // CHECK-LABEL: func.func @__nvqpp__mlirgen__simple_float_rotation
  // CHECK-SAME: ([[arg:.*]]: !cc.stdvec<f32>{{.*}}) attributes
  auto operator()(std::vector<float> theta) __qpu__ {
    int size = theta.size();
    bool empty = theta.empty();
    cudaq::qvector q(1);
    rx(theta[0], q[0]);
    mz(q);
  }
};

struct difficult_symphony {
  auto operator()(std::vector<float> theta) __qpu__ {
    float *firstData = theta.data();
    cudaq::qvector q(1);
    rx(firstData[0], q[0]);
    mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__difficult_symphony(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f32>{{.*}}) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f32>) -> !cc.ptr<!cc.array<f32 x ?>>
// clang-format on

int main() {
  std::vector<double> vec_args = {0.63};

  auto counts = cudaq::sample(simple_double_rotation{}, vec_args);
  counts.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.c_str(), count);
  }

  // can get <ZZ...Z> from counts too
  printf("Exp: %lf\n", counts.expectation());

  std::vector<float> float_args = {0.64};

  auto float_counts = cudaq::sample(simple_float_rotation{}, float_args);
  float_counts.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : float_counts) {
    printf("Observed: %s, %lu\n", bits.c_str(), count);
  }

  auto bob_counts = cudaq::sample(difficult_symphony{}, float_args);
  bob_counts.dump();

  // can get <ZZ...Z> from counts too
  printf("Exp: %lf\n", float_counts.expectation());
  return 0;
}
