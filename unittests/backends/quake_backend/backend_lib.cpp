/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This file implements some basic functions, deployed on the mock server, to
// test `device_call`.

extern "C" {

int add_op(int a, int b) { return a + b; }

int mul_op(int a, int b) { return a * b; }

int sub_op(int a, int b) { return a - b; }

}
