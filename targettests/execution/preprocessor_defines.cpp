/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir -DCUDAQ_HELLO_WORLD %s -o %t && %t | FileCheck --check-prefixes=DEFINE_ON %s
// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: if [ $(echo %cpp_std | cut -c4- ) -ge 20 ]; then \
// RUN:   nvq++ --enable-mlir %s -o %t && %t | FileCheck %s; \
// RUN: fi

#include "cudaq.h"

#ifdef CUDAQ_HELLO_WORLD
struct test {
    void operator()() __qpu__ {
        cudaq::qubit q;
        h(q);
        mz(q);
    }
};
#else 
struct test {
    void operator()() __qpu__ {
        cudaq::qubit q;
        x(q);
        mz(q);
    }
};
#endif 

int main() {
    cudaq::sample(test{}).dump();
}

// DEFINE_ON: { [[B0:.*]]:[[C0:.*]] [[B1:.*]]:[[C1:.*]] }
// CHECK: { [[B0:.*]]:[[C0:.*]] }
