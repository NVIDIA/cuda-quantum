/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Simulators
// RUN: nvq++ %cpp_std --enable-mlir  %s                             -o %t && %t | FileCheck %s

// Quantum emulators
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target ionq                     --emulate %s -o %t && %t | FileCheck %s
// 2 different IQM machines for 2 different topologies
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Adonis --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Apollo --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ float test_const_prop_cast() {
  return M_SQRT1_2;
}

__qpu__ void test_const_prop_cast_caller() {
  auto c = test_const_prop_cast();
  cudaq::qvector v(std::vector<cudaq::complex>({ c, c, 0., 0.}));
}

__qpu__ void test_complex_constant_array() {
  cudaq::qvector v(std::vector<cudaq::complex>({ M_SQRT1_2, M_SQRT1_2, 0., 0.}));
}

#ifdef CUDAQ_SIMULATION_SCALAR_FP32
__qpu__ void test_complex_constant_array_floating_point() {
  cudaq::qvector v(std::vector<std::complex<float>>({ M_SQRT1_2, M_SQRT1_2, 0., 0.}));
}
#else
__qpu__ void test_complex_constant_array_floating_point() {
  cudaq::qvector v(std::vector<std::complex<double>>({ M_SQRT1_2, M_SQRT1_2, 0., 0.}));
}
#endif

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

#ifdef CUDAQ_SIMULATION_SCALAR_FP32
__qpu__ void test_complex_array_param_floating_point(std::vector<std::complex<float>> inState) {
  cudaq::qvector q1 = inState;
}
#else
__qpu__ void test_complex_array_param_floating_point(std::vector<std::complex<double>> inState) {
  cudaq::qvector q1 = inState;
}
#endif

__qpu__ void test_real_constant_array() {
  cudaq::qvector v({ M_SQRT1_2, M_SQRT1_2, 0., 0.});
}

#ifdef CUDAQ_SIMULATION_SCALAR_FP32
__qpu__ void test_real_constant_array_floating_point() {
  cudaq::qvector v(std::vector<float>({ M_SQRT1_2, M_SQRT1_2, 0., 0.}));
}
#else
__qpu__ void test_real_constant_array_floating_point() {
  cudaq::qvector v(std::vector<double>({ M_SQRT1_2, M_SQRT1_2, 0., 0.}));
}
#endif

__qpu__ void test_real_array_param(std::vector<cudaq::real> inState) {
  cudaq::qvector q1 = inState;
}

#ifdef CUDAQ_SIMULATION_SCALAR_FP32
__qpu__ void test_real_array_param_floating_point(std::vector<float> inState) {
  cudaq::qvector q1 = inState;
}
#else
__qpu__ void test_real_array_param_floating_point(std::vector<double> inState) {
  cudaq::qvector q1 = inState;
}
#endif

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
    auto counts = cudaq::sample(test_const_prop_cast_caller);
    printCounts(counts);
  }

// CHECK: 00
// CHECK: 10

  {
    auto counts = cudaq::sample(test_complex_constant_array);
    printCounts(counts);
  }

// CHECK: 00
// CHECK: 10

  {
    auto counts = cudaq::sample(test_complex_constant_array_floating_point);
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
    auto counts = cudaq::sample(test_real_constant_array_floating_point);
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

// CHECK: 00
// CHECK: 10

        counts = cudaq::sample(test_complex_array_param, vec1);
        printCounts(counts);

// CHECK: 01
// CHECK: 11
    }
    {
        // Passing state data as argument (kernel mode)
        auto counts = cudaq::sample(test_complex_array_param_floating_point, vec);
        printCounts(counts);

// CHECK: 00
// CHECK: 10

        counts = cudaq::sample(test_complex_array_param_floating_point, vec1);
        printCounts(counts);

// CHECK: 01
// CHECK: 11
    }

    {
        // Passing state data as argument (builder mode)
        auto [kernel, v] = cudaq::make_kernel<std::vector<cudaq::complex>>();
        auto qubits = kernel.qalloc(v);

        auto counts = cudaq::sample(kernel, vec);
        printCounts(counts);

// CHECK: 00
// CHECK: 10

        counts = cudaq::sample(kernel, vec1);
        printCounts(counts);

// CHECK: 01
// CHECK: 11
    }
  }

  {
    std::vector<cudaq::real> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
    std::vector<cudaq::real> vec1{0., 0., M_SQRT1_2, M_SQRT1_2};
    {
        // Passing state data as argument (kernel mode)
        auto counts = cudaq::sample(test_real_array_param, vec);
        printCounts(counts);

// CHECK: 00
// CHECK: 10

        counts = cudaq::sample(test_real_array_param, vec1);
        printCounts(counts);

// CHECK: 01
// CHECK: 11
    }
    {
        // Passing state data as argument (kernel mode)
        auto counts = cudaq::sample(test_real_array_param_floating_point, vec);
        printCounts(counts);

// CHECK: 00
// CHECK: 10

        counts = cudaq::sample(test_real_array_param_floating_point, vec1);
        printCounts(counts);

// CHECK: 01
// CHECK: 11
    }

    {
        // Passing state data as argument (builder mode)
        auto [kernel, v] = cudaq::make_kernel<std::vector<cudaq::real>>();
        auto qubits = kernel.qalloc(v);

        auto counts = cudaq::sample(kernel, vec);
        printCounts(counts);

// CHECK: 00
// CHECK: 10

        counts = cudaq::sample(kernel, vec1);
        printCounts(counts);

// CHECK: 01
// CHECK: 11
    }
  }
}
