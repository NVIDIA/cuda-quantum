/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o out_defines.x && ./out_defines.x | FileCheck %s && rm out_defines.x
// RUN: nvq++ --enable-mlir -DCUDAQ_HELLO_WORLD %s -o out_defines2.x && ./out_defines2.x | FileCheck --check-prefixes=DEFINE_ON %s && rm out_defines2.x

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
