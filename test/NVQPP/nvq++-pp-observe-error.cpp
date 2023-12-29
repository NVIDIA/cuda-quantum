/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"

// RUN: nvq++-pp %s -verify
// RUN: nvq++ %s 2>&1 >/dev/null | FileCheck %s --check-prefix=COMPILEERROR

__qpu__ void kernel(std::vector<double> x) { // expected-error {{CUDA Quantum kernel passed to cudaq::observe cannot have measurements specified}}
  cudaq::qubit q;
  mz(q);
}

int main() {
  cudaq::spin_op h;
  cudaq::observe(kernel, h, std::vector<double>{1.2}); 
}

// COMPILEERROR: CUDA Quantum kernel passed to cudaq::observe cannot have measurements specified