/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --target anyon --emulate %s -o %t && %t | FileCheck %s

#include "mid_circuit_measurement.inc"

// CHECK: 1
// CHECK: 1
// CHECK: 1
// CHECK: 1

// CHECK: 1
// CHECK: 1
// CHECK: 1
// CHECK: 1

// CHECK: done
