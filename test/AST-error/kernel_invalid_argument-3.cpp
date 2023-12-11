/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>
#include <string>
#include <tuple>

// expected-error@+1{{kernel argument type not supported}}
void prepQubit(std::pair<int, double> basis, cudaq::qubit &q) __qpu__ {}

// expected-error@+1{{kernel argument type not supported}}
void RzArcTan2(bool input, std::pair<int, double> basis) __qpu__ {
  cudaq::qubit aux;
  cudaq::qubit resource;
  cudaq::qubit target;
  if (input) {
    x(target);
  }
  prepQubit(basis, target);
}

int main() {
  RzArcTan2(true, {});
  return 0;
}
