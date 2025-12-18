/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: nvcc

// RUN: (nvcc -c -Xcompiler -fPIC %p/cuda-1.cu -o %t.o && \
// RUN: nvq++ --enable-mlir %s %t.o -L `dirname $(which nvcc)`/../lib64 -lcudart -o %t && echo "Success") | \
// RUN: FileCheck %s

// CHECK-LABEL: Success

#include <cudaq.h>

struct CUDA_Quantum_Kernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    x(q);
  }
};

void cudaq_kernel() {
  CUDA_Quantum_Kernel{}();
}

void cuda_gpu_kernel();

int main() {
  cuda_gpu_kernel();
  cudaq_kernel();
}
