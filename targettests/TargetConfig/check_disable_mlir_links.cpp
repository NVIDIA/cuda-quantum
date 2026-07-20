/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --disable-mlir-links %s -o %s.x && ! ldd %s.x | grep -q libcudaq-mlir-runtime.so
// RUN: nvq++ --disable-mlir-links %s -o %s.x && %s.x
// We expect a failure when emulating a target that requires JIT compilation.
// RUN: nvq++ --disable-mlir-links --target quantinuum --emulate %s -o %s.x && CUDAQ_LOG_LEVEL=info %s.x 2>&1 | FileCheck %s --check-prefix=FAIL
// clang-format on

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

// FAIL: JIT compilation is disabled. Compilation is a no-op.
// FAIL: QPU does not support launching a CompiledModule without MLIR artifacts
