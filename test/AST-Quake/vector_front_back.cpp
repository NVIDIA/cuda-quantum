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
    std::vector<float> vec_float{0.0, 1.0, 2.0, 3.0, 4.0};
    auto front_0 = vec_float.front();
    // float back_0 = vec_float.back();

    // std::vector<bool> vec_bool{0,1,0,1,0,1};
    // bool front_1 = vec_bool.front();
    // bool back_1 = vec_bool.back();
}