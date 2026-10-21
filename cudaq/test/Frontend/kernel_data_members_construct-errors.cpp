/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: not cudaq-quake %s 2>&1 | FileCheck %s

#include <cudaq.h>

// The object of a kernel class with data members does not exist on the device,
// so the kernel cannot make another one.

struct N {
  int i = 1;
  void operator()() __qpu__ {
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: cannot create an object of a kernel class that has data members
    N other;
  }
};

struct P {
  int i = 1;
  void operator()() __qpu__ {
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: cannot create an object of a kernel class that has data members
    P copy(*this);
  }
};
