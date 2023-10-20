/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: ( cudaq-quake %s || true ) 2>&1 | FileCheck %s

// CHECK: cannot call variadic function (printf) from quantum kernel

#include <cudaq.h>

__qpu__ void invalid_code() {
  double d = sqrt(2.0); // this should be allowed
  printf("Hello\n");    // this should not
}
