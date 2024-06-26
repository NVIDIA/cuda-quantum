/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include "cudaq/builder/kernels.h"
#include <iostream>

__qpu__ void test(std::vector<cudaq::complex> inState) {
  cudaq::qvector q1 = inState;
}

void printCounts(cudaq::sample_result& result) {
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
}

int main() {
    std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
    std::vector<cudaq::complex> vec1{0., 0., M_SQRT1_2, M_SQRT1_2};
    {
        // Passing state data as argument (kernel mode)
        auto counts = cudaq::sample(test, vec);
        printCounts(counts);

        counts = cudaq::sample(test, vec1);
        printCounts(counts);
    }

    {
        // Passing state data as argument (builder mode)
        auto [kernel, v] = cudaq::make_kernel<std::vector<cudaq::complex>>();
        auto qubits = kernel.qalloc(v);
    
        auto counts = cudaq::sample(kernel, vec);
        printCounts(counts);
    }
}

// CHECK: 01
// CHECK: 00

// CHECK: 10
// CHECK: 10

// CHECK: 01
// CHECK: 00