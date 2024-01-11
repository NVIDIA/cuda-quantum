/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: nvcc

// RUN: (nvq++ -std=c++17 -c %s -o %t.1.o && \
// RUN: nvcc -std=c++17 cuda-1.cu -o %t.2.o && \
// RUN: nvq++ %t.1.o %t.2.o -o %t && echo "Success") | FileCheck %s

// CHECK-LABEL: Success

#include <cudaq.h>

__qpu__ void cudaq_kernel() {
  cudaq::qubit q;
  h(q);
  x(q);
}
