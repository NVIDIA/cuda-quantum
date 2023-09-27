/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Simple test of vector front/back on different std::vector<> types.

#include <cudaq.h>

__qpu__ void test() {
    std::vector<float> vec_float{0.0, 1.0, 2.0, 3.14};
    auto front = vec_float.front();
    float back = vec_float.back();

    std::vector<bool> vec_bool{0,1,0,1,0,1};
    bool zero = vec_bool.front();
    bool one = vec_bool.back();

    // Testing our front/back bools as indices for the register here.
    cudaq::qreg q(2);
    rx(front, q[zero]);
    rx(back, q[one]);
    mz(q[zero]);
    mz(q[one]);
}

// TODO: Will update this to a filecheck test once the output Quake is
// finalized. This is just to ensure it executes as I'd expect it to.
int main() {
    auto result = cudaq::sample(test);
    result.dump();
}