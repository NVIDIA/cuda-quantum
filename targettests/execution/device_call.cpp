/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --opt-pass distributed-device-call %s -o %t && %t | FileCheck %s

#include <cstdio>
#include <cudaq.h>

int add(int a, int b) {
  printf("%d + %d\n", a, b);
  return a + b;
}

__qpu__ auto test() {
  cudaq::qubit q;
  h(q);
  auto result = cudaq::device_call(add, 1, 2);
  return result;
}

int main() {
  int r = test();
  printf("%d\n", r);
  return 0;
}

// CHECK: 1 + 2
// CHECK: 3
