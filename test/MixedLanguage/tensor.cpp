/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: nvcc

// clang-format off
// RUN: nvcc -I runtime -c -Xcompiler -fPIC %cpp_std %p/tensor.cu -o %t.o
// RUN: nvq++ %cpp_std --enable-mlir %s %t.o -L `dirname $(which nvcc)`/../lib64 -lcudart -o %t.x
// clang-format on

#include <cudaq.h>

struct CUDA_Quantum_Kernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    x(q);
  }
};

void cudaq_kernel() { CUDA_Quantum_Kernel{}(); }

void cuda_tensor();

int main() {
  cuda_tensor();
  cudaq_kernel();
}