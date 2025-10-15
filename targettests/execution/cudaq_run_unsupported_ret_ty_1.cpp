/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std %s |& FileCheck %s -check-prefix=FAIL
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ std::vector<std::vector<int>> vec_of_vec() { return {{1, 2}, {3, 4}}; }

int main() {
  const auto results = cudaq::run(100, vec_of_vec);
  return 0;
}

// FAIL: unhandled vector element type is not yet supported
