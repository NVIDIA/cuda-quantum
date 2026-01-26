/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --disable-mlir-links %s -o %s.x && ! ldd %s.x | grep -q libcudaq-mlir-runtime.so 

#include "cudaq.h"

__qpu__ void bell() {
  cudaq::qubit q, r;
  h(q);
  x<cudaq::ctrl>(q, r);
}

int main() {
  auto counts = cudaq::sample(bell);
  counts.dump();
  return 0;
}
