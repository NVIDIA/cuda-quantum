/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s 2>&1 | FileCheck %s

#include <cudaq.h>

__qpu__ void kernel_as_func(double theta) {
  cudaq::qreg q(2);
  x(q[0]);
  ry(theta, q[1]);
  x<cudaq::ctrl>(q[1], q[0]);
}

template <std::size_t N>
struct kernel_as_struct {
  auto operator()() __qpu__ {
    cudaq::qreg<N> q;
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  // Sample
  {
    auto [kernel, value] = cudaq::make_kernel<float>();
    auto q = kernel.qalloc();
    kernel.x(q);
    // Calling sample but not passing along a concrete argument for `value`
    auto result = cudaq::sample(kernel);
    // CHECK: 'InvalidArgs<cudaq::Msg<{{[0-9]+}}>{"requires <float>, got <>"}>'
  }
  {
    // Calling sample but not passing along a concrete argument for `value`
    auto result = cudaq::sample(kernel_as_func);
    // CHECK: 'InvalidArgs<cudaq::Msg<{{[0-9]+}}>{"requires <double>, got <>"}>'
  }
  {
    // Provide an argument but not needed.
    auto result = cudaq::sample(kernel_as_struct<3>{}, 1.234);
    // CHECK: 'InvalidArgs<cudaq::Msg<{{[0-9]+}}>{"requires <>, got <double>"}>'
  }

  // Observe
  {
    cudaq::spin_op h = cudaq::spin::z(0);
    auto kernel = cudaq::make_kernel();
    auto q = kernel.qalloc();
    kernel.x(q);
    // Passing a double while none required
    auto result = cudaq::observe(kernel, h, 1.234);
    // CHECK NEXT: 'InvalidArgs<cudaq::Msg<{{[0-9]+}}>{"requires <>, got
    // <double>"}>'
  }
  {
    cudaq::spin_op h = cudaq::spin::z(0) * cudaq::spin::z(1);
    // Calling sample but not passing along a concrete argument for `value`
    auto result = cudaq::observe(kernel_as_func, h);
    // CHECK NEXT: 'InvalidArgs<cudaq::Msg<{{[0-9]+}}>{"requires <double>, got
    // <>"}>'
  }
}
