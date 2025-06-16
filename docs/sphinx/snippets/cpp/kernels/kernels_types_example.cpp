/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ kernels_types_example.cpp && ./a.out`

#include <cudaq.h>
#include <stdio.h>
#include <vector>

// [Begin Kernel Types C++]
// Entry point lambda kernel
auto my_first_entry_point_kernel_lambda = [](double x) __qpu__ {
  cudaq::qarray<1> q;
  ry(x, q[0]);
  mz(q);
  printf("Lambda kernel executed with x = %f\n", x);
};

// Entry point typed callable kernel
struct my_second_entry_point_kernel_struct {
  void operator()(double x, std::vector<double> params) __qpu__ {
    cudaq::qarray<1> q;
    ry(x * params[0], q[0]);
    mz(q);
    printf("Struct kernel executed with x = %f, param0 = %f\n", x, params[0]);
  }
};

// Pure device free function kernel
__qpu__ void my_first_pure_device_kernel(cudaq::qview<> qubits) {
  h(qubits[0]);
  printf("Pure device kernel executed on a qubit.\n");
}

// Entry point kernel to call the pure device kernel
auto caller_for_pure_device = []() __qpu__ {
  cudaq::qarray<1> q;
  my_first_pure_device_kernel(q);
  mz(q);
};
// [End Kernel Types C++]

int main() {
  // [Begin Kernel Types C++ Execution]
  my_first_entry_point_kernel_lambda(1.23);
  std::vector<double> params = {0.5};
  my_second_entry_point_kernel_struct{}(1.23, params);
  cudaq::sample(caller_for_pure_device); // Sample to execute the caller
  // [End Kernel Types C++ Execution]
  return 0;
}

