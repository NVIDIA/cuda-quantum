/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu %s -o %t  && %t | FileCheck %s
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void test_complex_constant_array() {
   cudaq::qvector v(std::vector<cudaq::complex>({ M_SQRT1_2, M_SQRT1_2, 0., 0.}));
}

__qpu__ void test_complex_constant_array2() {
   cudaq::qvector v1(std::vector<cudaq::complex>({ M_SQRT1_2, M_SQRT1_2, 0., 0.}));
   cudaq::qvector v2(std::vector<cudaq::complex>({ 0., 0., M_SQRT1_2, M_SQRT1_2}));
}

__qpu__ void test_complex_constant_array3() {
   cudaq::qvector v({
    cudaq::complex(M_SQRT1_2),
    cudaq::complex(M_SQRT1_2),
    cudaq::complex(0.0),
    cudaq::complex(0.0)
  });
}

__qpu__ void test_complex_array_param(std::vector<cudaq::complex> inState) {
  cudaq::qvector q1 = inState;
}

__qpu__ void test_real_constant_array() {
  cudaq::qvector v({ M_SQRT1_2, M_SQRT1_2, 0., 0.});
}

__qpu__ void test_real_array_param(std::vector<cudaq::real> inState) {
  cudaq::qvector q1 = inState;
}

__qpu__ void test_double_array_param(std::vector<double> inState) {
  cudaq::qvector q = inState;
}

__qpu__ void test_float_array_param(std::vector<float> inState) {
  cudaq::qvector q = inState;
}

void printCounts(cudaq::sample_result& result) {
  std::vector<std::string> values{};
  for (auto &&[bits, counts] : result) {
    values.push_back(bits);
  }

  std::sort(values.begin(), values.end());
  for (auto &&bits : values) {
    std::cout << bits << '\n';
  }
}

int main() {
    {
      auto counts = cudaq::sample(test_complex_constant_array);
      printCounts(counts);
    }

// CHECK: 00
// CHECK: 10

    {
      auto counts = cudaq::sample(test_complex_constant_array2);
      printCounts(counts);
    }

// CHECK: 0001
// CHECK: 0011
// CHECK: 1001
// CHECK: 1011

    {
      auto counts = cudaq::sample(test_complex_constant_array3);
      printCounts(counts);
    }

// CHECK: 00
// CHECK: 10

    {
      auto counts = cudaq::sample(test_real_constant_array);
      printCounts(counts);
    }

// CHECK: 00
// CHECK: 10

    {
      std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
      std::vector<cudaq::complex> vec1{0., 0., M_SQRT1_2, M_SQRT1_2};
      {
          // Passing state data as argument (kernel mode)
          auto counts = cudaq::sample(test_complex_array_param, vec);
          printCounts(counts);

          counts = cudaq::sample(test_complex_array_param, vec1);
          printCounts(counts);
      }

// CHECK: 00
// CHECK: 10

// CHECK: 01
// CHECK: 11

      {
          // Passing state data as argument (builder mode)
          auto [kernel, v] = cudaq::make_kernel<std::vector<cudaq::complex>>();
          auto qubits = kernel.qalloc(v);

          auto counts = cudaq::sample(kernel, vec);
          printCounts(counts);

          counts = cudaq::sample(kernel, vec1);
          printCounts(counts);
      }
    }

// CHECK: 00
// CHECK: 10

// CHECK: 01
// CHECK: 11

    {
      std::vector<cudaq::real> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
      std::vector<cudaq::real> vec1{0., 0., M_SQRT1_2, M_SQRT1_2};
      {
          // Passing state data as argument (kernel mode)
          auto counts = cudaq::sample(test_real_array_param, vec);
          printCounts(counts);

          counts = cudaq::sample(test_real_array_param, vec1);
          printCounts(counts);
      }

// CHECK: 00
// CHECK: 10

// CHECK: 01
// CHECK: 11

      {
          // Passing state data as argument (builder mode)
          auto [kernel, v] = cudaq::make_kernel<std::vector<cudaq::real>>();
          auto qubits = kernel.qalloc(v);

          auto counts = cudaq::sample(kernel, vec);
          printCounts(counts);

          counts = cudaq::sample(kernel, vec1);
          printCounts(counts);
      }

// CHECK: 00
// CHECK: 10

// CHECK: 01
// CHECK: 11
    }

    {
      std::vector<double> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
      std::vector<double> vec1{0., 0., M_SQRT1_2, M_SQRT1_2};

      // Passing state data as argument (kernel mode)
      auto counts = cudaq::sample(test_double_array_param, vec);
      printCounts(counts);

      counts = cudaq::sample(test_double_array_param, vec1);
      printCounts(counts);
    }

// CHECK: 00
// CHECK: 10

// CHECK: 01
// CHECK: 11

    {
      std::vector<float> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
      std::vector<float> vec1{0., 0., M_SQRT1_2, M_SQRT1_2};

      // Passing state data as argument (kernel mode)
      auto counts = cudaq::sample(test_float_array_param, vec);
      printCounts(counts);

      counts = cudaq::sample(test_float_array_param, vec1);
      printCounts(counts);
    }

// CHECK: 00
// CHECK: 10

// CHECK: 01
// CHECK: 11
}
