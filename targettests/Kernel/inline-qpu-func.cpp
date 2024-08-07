/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s

#include "cudaq.h"

// This function has no cudaq::qubit's in the parameter list, so it will be
// tagged as a possible cudaq-entrypoint kernel. Make sure we can still inline
// it if called from another kernel.
bool xor_result(const std::vector<cudaq::measure_result> &result_vec) __qpu__ {
  bool result = false;
  for (auto x : result_vec)
    result ^= x;
  return result;
}

bool kernel() __qpu__ {
  cudaq::qvector q(7);
  x(q);
  std::vector<cudaq::measure_result> mz_res = mz(q);
  bool res = xor_result(mz_res);
  return res;
}

int main(int argc, char *argv[]) {
  printf("Result: %d\n", static_cast<int>(kernel()));
  return 0;
}

// CHECK: Result: 1
