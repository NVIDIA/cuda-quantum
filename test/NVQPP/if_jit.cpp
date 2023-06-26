/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s --target quantinuum --emulate && ./a.out | FileCheck %s

#include <cudaq.h>

__qpu__ void foo(bool value) {
  cudaq::qubit q;
  if (value)
    x(q);
  mz(q);
}

int main() {
  auto result = cudaq::sample(100, foo, true);
  result.dump();
  return 0;
}

// CHECK: { 0:{{[0-9]+}} }
