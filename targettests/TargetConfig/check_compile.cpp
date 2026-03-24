/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Check that we can compile all the targets
// i.e., all the config files are valid.
// RUN: for target in $(nvq++ --list-targets); do echo "Testing target: ${target}"; nvq++ --library-mode --target ${target} %s; done
// RUN: for target in $(nvq++ --list-targets); do echo "Testing target: ${target}"; nvq++ --enable-mlir --target ${target} %s; done

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
