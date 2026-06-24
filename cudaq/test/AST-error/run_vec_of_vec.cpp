/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>
#include <iostream>

__qpu__ std::vector<std::vector<int>> vec_of_vec() { 
  // expected-error@+2 {{unhandled vector element type is not yet supported}}
  // expected-error@+1 {{statement not supported in qpu kernel}}
  return {{1, 2}, {3, 4}};
}

int main() {
  auto const result1 = cudaq::run(10, vec_of_vec); 
  return 0;
}
