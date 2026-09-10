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

// A stand-in for a backend-implemented operation. The mock server erases the
// call, so this body never runs, but the host object has to link.
void __qm__wait_function(double duration, void *q) {}

int add_op(int a, int b) { return a + b; }

int mul_op(int a, int b) { return a * b; }

int sub_op(int a, int b) { return a - b; }
}
