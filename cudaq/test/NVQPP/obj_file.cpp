/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -c %s -o %t 2>&1 | FileCheck %s
// RUN: [ -e %t ]

#include <cudaq.h>

int plain_old_function() { return 0; }

// CHECK: warning: The CUDA-Q `sample` and `observe` algorithmic primitives will change in a future release.
