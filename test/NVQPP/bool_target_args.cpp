/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test that Bool-type target arguments are correctly passed to the backend
// configuration. Bool args can be specified without a value (implicit true).

// clang-format off
// Test 1: Implicit true (--ionq-debias without value means true)
// RUN: nvq++ -v --target ionq --ionq-debias %s -o %t |& FileCheck --check-prefix=IMPLICIT %s

// Test 2: Explicit true
// RUN: nvq++ -v --target ionq --ionq-debias true %s -o %t |& FileCheck --check-prefix=EXPLICIT %s

// Test 3: Explicit false using = syntax
// RUN: nvq++ -v --target ionq --ionq-debias=false %s -o %t |& FileCheck --check-prefix=FALSE %s
// clang-format on

// IMPLICIT: -DNVQPP_TARGET_BACKEND_CONFIG
// IMPLICIT-SAME: debias;true

// EXPLICIT: -DNVQPP_TARGET_BACKEND_CONFIG
// EXPLICIT-SAME: debias;true

// FALSE: -DNVQPP_TARGET_BACKEND_CONFIG
// FALSE-SAME: debias;false

#include <cudaq.h>

struct test_kernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
  }
};

int main() { return 0; }
