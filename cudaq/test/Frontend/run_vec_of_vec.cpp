/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s
// expected-no-diagnostics

// The bridge accepts a vector of vectors as a kernel result. Whether the
// result of a `cudaq::run` of such a kernel can be decoded is not known to the
// bridge. That is reported when the kernel is run.

#include <cudaq.h>
#include <iostream>

__qpu__ std::vector<std::vector<int>> vec_of_vec() { return {{1, 2}, {3, 4}}; }

int main() {
  auto const result1 = cudaq::run(10, vec_of_vec);
  return 0;
}
