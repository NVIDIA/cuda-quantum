/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if [ command -v split-file ]; then \
// RUN: split-file %s %t && \
// RUN: nvq++ --target stim -c %t/gke-1.cpp \
// RUN:   %t/gke-2.cpp -o %t/gke.out && %t/gke.out ; else \
// RUN: echo "skipping" ; fi
// clang-format on

//--- gke-1.cpp

#include <cudaq.h>

// Will be defined in a separate file
__qpu__ int mytest(int x, std::vector<cudaq::measure_result> y);

__qpu__ int mykernel() {
  cudaq::qvector q(2);
  h(q);
  auto mzq = mz(q);
  int res = cudaq::device_call(mytest, 1, mzq);
  return res;
}

int main() {
  auto res = cudaq::run(1, mykernel);
  return 0;
}

//--- gke-2.cpp

#include <cudaq.h>

__qpu__ int mytest(int x, std::vector<cudaq::measure_result> y) {
  return x * 2;
}
