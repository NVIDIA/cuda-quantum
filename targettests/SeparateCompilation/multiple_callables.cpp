/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o %t && %t

#include "cudaq.h"

__qpu__ void entry(const cudaq::qkernel<void(cudaq::qvector<> &)> &o,
                   const cudaq::qkernel<void(cudaq::qvector<> &, int)> &p) {
  cudaq::qvector q(2);
  o(q);
  p(q, 1);
}

int main() {
  auto l = [](cudaq::qvector<> &q) __qpu__ { x(q[0]); };
  auto m = [](cudaq::qvector<> &q, int i) __qpu__ { y(q[i]); };

  entry(l, m);
  return 0;
}
