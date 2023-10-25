/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This code is from Issue 296.

// RUN: nvq++ %s --target quantinuum --emulate -o %t.x && %t.x | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ void foo(bool value) {
  cudaq::qubit q;
  if (value)
    x(q);

  mz(q);
}

int main() {
  auto result = cudaq::sample(100, foo, true);

#ifndef SYNTAX_CHECK
  std::cout << result.most_probable() << '\n';
  assert("1" == result.most_probable());
#endif

  return 0;
}

// CHECK: 1